/* Bản TUẦN TỰ (C thuần, một luồng, không thư viện ngoài): huấn luyện chồng khối MLP bằng SGD.
 * Dùng làm (1) đối chứng tính đúng cho bản CUDA và (2) mốc thời gian tuần tự.
 * Biên dịch: make seq   (cc -O3 -march=native -o mlp_seq mlp_seq.c -lm)
 * Chạy:      ./mlp_seq --H 256 --L 4 --B 64 --steps 3
 *            ./mlp_seq --H 64 --L 2 --B 32 --check        (in loss và chuẩn gradient để so với bản CUDA) */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "common.h"

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* C[M×N] (+)= A[M×K] · B[K×N]  (row-major) */
static void gemm_nn(int M, int N, int K, const float *A, const float *B, float *C, int acc) {
    for (int i = 0; i < M; i++) {
        float *c = C + (size_t)i * N;
        if (!acc) memset(c, 0, sizeof(float) * (size_t)N);
        for (int k = 0; k < K; k++) {
            float a = A[(size_t)i * K + k];
            const float *b = B + (size_t)k * N;
            for (int j = 0; j < N; j++) c[j] += a * b[j];
        }
    }
}
/* C[K×N] += Aᵀ · B,  A: M×K, B: M×N   (gradient trọng số: dW = Xᵀ · dZ) */
static void gemm_tn_acc(int M, int K, int N, const float *A, const float *B, float *C) {
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++) {
            float a = A[(size_t)i * K + k];
            const float *b = B + (size_t)i * N;
            float *c = C + (size_t)k * N;
            for (int j = 0; j < N; j++) c[j] += a * b[j];
        }
}
/* C[M×K] = A · Bᵀ,  A: M×N, B: K×N   (lan truyền ngược qua trọng số: dU = dY · W2ᵀ) */
static void gemm_nt(int M, int N, int K, const float *A, const float *B, float *C) {
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++) {
            const float *a = A + (size_t)i * N, *b = B + (size_t)k * N;
            float s = 0.f;
            for (int j = 0; j < N; j++) s += a[j] * b[j];
            C[(size_t)i * K + k] = s;
        }
}

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "hết bộ nhớ (%zu byte)\n", n); exit(1); }
    return p;
}

int main(int argc, char **argv) {
    MpConfig c = { 256, 0, 4, 64, 1, 1e-3f, 0, 0, 0.1f };
    int steps = 3, warmup = 1, check = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--H") && i + 1 < argc) c.H = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--L") && i + 1 < argc) c.L = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--B") && i + 1 < argc) c.B = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) c.seed = (uint64_t)atoll(argv[++i]);
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lr") && i + 1 < argc) c.lr = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--check")) check = 1;
        else { fprintf(stderr, "đối số không hợp lệ: %s\n", argv[i]); return 2; }
    }
    c.F = 4 * c.H;
    c.scale_w1 = 1.0f / sqrtf((float)c.H);
    c.scale_w2 = 1.0f / sqrtf((float)c.F);
    const int H = c.H, F = c.F, L = c.L, B = c.B;
    const size_t nW1 = (size_t)H * F, nW2 = (size_t)F * H;

    float *W1 = xmalloc(sizeof(float) * nW1 * L), *B1 = xmalloc(sizeof(float) * (size_t)F * L);
    float *W2 = xmalloc(sizeof(float) * nW2 * L);
    float *dW1 = xmalloc(sizeof(float) * nW1 * L), *dB1 = xmalloc(sizeof(float) * (size_t)F * L);
    float *dW2 = xmalloc(sizeof(float) * nW2 * L);
    float *X = xmalloc(sizeof(float) * (size_t)B * H * (L + 1));      /* X[l]: đầu vào khối l, X[L]: đầu ra */
    float *U = xmalloc(sizeof(float) * (size_t)B * F * L);            /* U[l] = ReLU(X[l]·W1 + b1) */
    float *dY = xmalloc(sizeof(float) * (size_t)B * H), *dX = xmalloc(sizeof(float) * (size_t)B * H);
    float *dU = xmalloc(sizeof(float) * (size_t)B * F);

    for (int l = 0; l < L; l++) {
        for (size_t i = 0; i < nW1; i++) W1[l * nW1 + i] = c.scale_w1 * mp_urand(c.seed, MP_TAG_W1, l, i);
        for (int i = 0; i < F; i++) B1[(size_t)l * F + i] = c.scale_b1 * mp_urand(c.seed, MP_TAG_B1, l, i);
        for (size_t i = 0; i < nW2; i++) W2[l * nW2 + i] = c.scale_w2 * mp_urand(c.seed, MP_TAG_W2, l, i);
    }
    for (size_t i = 0; i < (size_t)B * H; i++) X[i] = mp_urand(c.seed, MP_TAG_X, 0, i);

    double loss = 0, gnorm = 0, total_s = 0;
    for (int it = 0; it < warmup + steps; it++) {
        double t0 = now_s();
        /* ---- forward ---- */
        for (int l = 0; l < L; l++) {
            float *Xl = X + (size_t)l * B * H, *Ul = U + (size_t)l * B * F, *Xn = X + (size_t)(l + 1) * B * H;
            gemm_nn(B, F, H, Xl, W1 + l * nW1, Ul, 0);
            for (int i = 0; i < B; i++)
                for (int f = 0; f < F; f++) {
                    float z = Ul[(size_t)i * F + f] + B1[(size_t)l * F + f];
                    Ul[(size_t)i * F + f] = z > 0.f ? z : 0.f;
                }
            gemm_nn(B, H, F, Ul, W2 + l * nW2, Xn, 0);
            for (size_t i = 0; i < (size_t)B * H; i++) Xn[i] += Xl[i];
        }
        const float *XL = X + (size_t)L * B * H;
        double ss = 0;
        for (size_t i = 0; i < (size_t)B * H; i++) ss += (double)XL[i] * XL[i];
        loss = 0.5 * ss / ((double)B * H);
        /* ---- backward ---- */
        memset(dW1, 0, sizeof(float) * nW1 * L); memset(dB1, 0, sizeof(float) * (size_t)F * L);
        memset(dW2, 0, sizeof(float) * nW2 * L);
        float inv = 1.0f / ((float)B * (float)H);
        for (size_t i = 0; i < (size_t)B * H; i++) dY[i] = XL[i] * inv;
        for (int l = L - 1; l >= 0; l--) {
            const float *Xl = X + (size_t)l * B * H, *Ul = U + (size_t)l * B * F;
            gemm_tn_acc(B, F, H, Ul, dY, dW2 + l * nW2);                       /* dW2 += Uᵀ·dY */
            gemm_nt(B, H, F, dY, W2 + l * nW2, dU);                            /* dU = dY·W2ᵀ  */
            for (size_t i = 0; i < (size_t)B * F; i++) if (!(Ul[i] > 0.f)) dU[i] = 0.f;   /* dZ */
            for (int i = 0; i < B; i++)
                for (int f = 0; f < F; f++) dB1[(size_t)l * F + f] += dU[(size_t)i * F + f];
            gemm_tn_acc(B, H, F, Xl, dU, dW1 + l * nW1);                       /* dW1 += Xᵀ·dZ */
            gemm_nt(B, F, H, dU, W1 + l * nW1, dX);                            /* dX = dZ·W1ᵀ  */
            for (size_t i = 0; i < (size_t)B * H; i++) dY[i] += dX[i];         /* + nhánh residual */
        }
        gnorm = 0;
        for (size_t i = 0; i < nW1 * L; i++) gnorm += (double)dW1[i] * dW1[i];
        for (size_t i = 0; i < (size_t)F * L; i++) gnorm += (double)dB1[i] * dB1[i];
        for (size_t i = 0; i < nW2 * L; i++) gnorm += (double)dW2[i] * dW2[i];
        gnorm = sqrt(gnorm);
        if (check) break;                                         /* chỉ một lượt forward+backward, không cập nhật */
        /* ---- SGD ---- */
        for (size_t i = 0; i < nW1 * L; i++) W1[i] -= c.lr * dW1[i];
        for (size_t i = 0; i < (size_t)F * L; i++) B1[i] -= c.lr * dB1[i];
        for (size_t i = 0; i < nW2 * L; i++) W2[i] -= c.lr * dW2[i];
        if (it >= warmup) total_s += now_s() - t0;
    }
    if (check) {
        printf("CHECK impl=seq H=%d L=%d B=%d loss=%.9e gradnorm=%.9e\n", H, L, B, loss, gnorm);
        return 0;
    }
    double step_s = total_s / steps;
    printf("RESULT impl=seq mode=seq gpus=0 micro=0 H=%d L=%d B=%d params_M=%.1f step_ms=%.3f samples_s=%.1f "
           "gflops=%.2f loss=%.6e\n", H, L, B, mp_param_count(&c) / 1e6, step_s * 1e3, B / step_s,
           mp_step_flops(&c) / step_s / 1e9, loss);
    return 0;
}
