/* Bản SONG SONG (CUDA, nhiều GPU): huấn luyện chồng khối MLP bằng SGD với MODEL PARALLELISM.
 *
 *   --mode pipeline   Pipeline parallelism (GPipe): các khối chia thành S stage liên tiếp, mỗi stage một GPU;
 *                     batch chia thành M micro-batch chạy gối đầu nhau, activation chuyển giữa các GPU bằng
 *                     cudaMemcpyPeerAsync. --gpus 1 là bản một GPU.
 *   --mode tensor     Tensor parallelism (kiểu Megatron): mỗi khối chia theo chiều F: W1 theo cột, W2 theo hàng;
 *                     mỗi khối cần một all-reduce ở forward và một ở backward (tự viết, recursive doubling).
 *
 * Toàn bộ kernel tự cài đặt, KHÔNG dùng cuBLAS, cuDNN, NCCL hay MPI: GEMM chia ô, bias+ReLU, residual, mặt nạ ReLU,
 * tổng cột, SGD, tổng bình phương, all-reduce. Chỉ dùng CUDA runtime API (cấp phát, stream, event, copy giữa GPU).
 *
 * Biên dịch: make cuda          (nvcc -O3 -arch=sm_90 -o mlp_cuda mlp_cuda.cu)
 * Chạy:      ./mlp_cuda --mode pipeline --gpus 4 --micro 8 --H 4096 --L 8 --B 2048
 *            ./mlp_cuda --mode tensor   --gpus 4 --H 4096 --L 8 --B 2048
 *            ./mlp_cuda --mode pipeline --gpus 2 --H 64 --L 4 --B 32 --check   (so với ./mlp_seq --check) */
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "common.h"

#define CK(x)                                                                                    \
    do {                                                                                         \
        cudaError_t e_ = (x);                                                                    \
        if (e_ != cudaSuccess) {                                                                 \
            fprintf(stderr, "Lỗi CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
            exit(1);                                                                             \
        }                                                                                        \
    } while (0)

static char g_desc[256] = "";    /* mô tả cấu hình, in kèm kết quả OOM */

/* Cấp phát trên thiết bị hiện tại. Hết bộ nhớ là kết quả hợp lệ (mô hình không vừa GPU): in RESULT status=OOM. */
static float *dmalloc(size_t n) {
    float *p = nullptr;
    cudaError_t e = cudaMalloc((void **)&p, n * sizeof(float));
    if (e != cudaSuccess) {
        if (e == cudaErrorMemoryAllocation) {
            printf("RESULT %s status=OOM(alloc) failed_alloc_GB=%.1f\n", g_desc, (double)n * 4 / 1e9);   /* kích thước mảng bị từ chối, không phải tổng nhu cầu */
            fflush(stdout);
            exit(0);
        }
        fprintf(stderr, "Lỗi cudaMalloc: %s\n", cudaGetErrorString(e));
        exit(1);
    }
    return p;
}

/* ============================== KERNEL ============================== */

#define BM 64
#define BN 64
#define BK 16

/* C[M×N] = op(A)[M×K] · op(B)[K×N] + beta·C.  Row-major.
 *   TA=false: A(m,k) = A[m·lda + k];  TA=true: A(m,k) = A[k·lda + m]
 *   TB=false: B(k,n) = B[k·ldb + n];  TB=true: B(k,n) = B[n·ldb + k]
 * Mỗi block 16×16 luồng tính ô 64×64 của C (mỗi luồng 4×4 phần tử), duyệt K theo lát BK=16 qua shared memory. */
template <bool TA, bool TB>
__global__ void gemm_kernel(int M, int N, int K, const float *A, int lda, const float *B, int ldb, float *C, int ldc,
                            float beta) {
    __shared__ float As[BK][BM + 1];
    __shared__ float Bs[BK][BN + 1];
    const int tx = threadIdx.x, ty = threadIdx.y, tid = ty * 16 + tx;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.f;

    for (int k0 = 0; k0 < K; k0 += BK) {
#pragma unroll
        for (int r = 0; r < 4; r++) {
            int e = tid + r * 256;                       /* 0..1023 = BM·BK = BK·BN */
            int mm, kk;
            if (!TA) { kk = e % BK; mm = e / BK; } else { mm = e % BM; kk = e / BM; }
            int gm = m0 + mm, gk = k0 + kk;
            float av = 0.f;
            if (gm < M && gk < K) av = !TA ? A[(size_t)gm * lda + gk] : A[(size_t)gk * lda + gm];
            As[kk][mm] = av;
            int nn, kb;
            if (!TB) { nn = e % BN; kb = e / BN; } else { kb = e % BK; nn = e / BK; }
            int gn = n0 + nn, gkb = k0 + kb;
            float bv = 0.f;
            if (gn < N && gkb < K) bv = !TB ? B[(size_t)gkb * ldb + gn] : B[(size_t)gn * ldb + gkb];
            Bs[kb][nn] = bv;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK; k++) {
            float a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; i++) a[i] = As[k][ty + 16 * i];
#pragma unroll
            for (int j = 0; j < 4; j++) b[j] = Bs[k][tx + 16 * j];
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int gm = m0 + ty + 16 * i, gn = n0 + tx + 16 * j;
            if (gm < M && gn < N) {
                size_t o = (size_t)gm * ldc + gn;
                C[o] = acc[i][j] + (beta != 0.f ? beta * C[o] : 0.f);
            }
        }
}

template <bool TA, bool TB>
static void gemm(cudaStream_t st, int M, int N, int K, const float *A, int lda, const float *B, int ldb, float *C,
                 int ldc, float beta) {
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM), block(16, 16);
    gemm_kernel<TA, TB><<<grid, block, 0, st>>>(M, N, K, A, lda, B, ldb, C, ldc, beta);
}

#define GRID1D(n) (int)(((n) + 255) / 256 > 65535 ? 65535 : ((n) + 255) / 256)
#define FOR_EACH(i, n) for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < (n); i += (size_t)gridDim.x * blockDim.x)

/* p[i][j] = scale · urand(giá trị của phần tử toàn cục (row0+i, col0+j) trong ma trận ld_global cột) */
__global__ void k_init_mat(float *p, int rows, int cols, size_t ld_global, size_t row0, size_t col0, uint64_t seed,
                           uint32_t tag, uint32_t layer, float scale) {
    FOR_EACH(i, (size_t)rows * cols) {
        size_t r = i / cols, c = i % cols;
        p[i] = scale * mp_urand(seed, tag, layer, (row0 + r) * ld_global + col0 + c);
    }
}
__global__ void k_init_vec(float *p, size_t n, size_t off0, uint64_t seed, uint32_t tag, uint32_t layer, float scale) {
    FOR_EACH(i, n) p[i] = scale * mp_urand(seed, tag, layer, off0 + i);
}
__global__ void k_bias_relu(float *z, const float *b, int rows, int cols) {
    FOR_EACH(i, (size_t)rows * cols) {
        float v = z[i] + b[i % cols];
        z[i] = v > 0.f ? v : 0.f;
    }
}
__global__ void k_add(float *y, const float *x, size_t n) { FOR_EACH(i, n) y[i] += x[i]; }
__global__ void k_scale(float *dst, const float *src, float s, size_t n) { FOR_EACH(i, n) dst[i] = src[i] * s; }
__global__ void k_relu_mask(float *d, const float *u, size_t n) { FOR_EACH(i, n) if (!(u[i] > 0.f)) d[i] = 0.f; }
__global__ void k_sgd(float *p, const float *g, float lr, size_t n) { FOR_EACH(i, n) p[i] -= lr * g[i]; }
/* db[j] += Σ_i dZ[i][j] (mỗi luồng một cột, đọc liên tiếp theo cột nên gộp được truy cập bộ nhớ) */
__global__ void k_colsum_acc(const float *dz, float *db, int rows, int cols) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= cols) return;
    float s = 0.f;
    for (int i = 0; i < rows; i++) s += dz[(size_t)i * cols + j];
    db[j] += s;
}
__global__ void k_sumsq(const float *p, size_t n, double *out) {
    double s = 0;
    FOR_EACH(i, n) { double v = p[i]; s += v * v; }
    __shared__ double sh[256];
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if (threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, sh[0]);
}

static double dev_sumsq(int dev, cudaStream_t st, const float *p, size_t n) {
    CK(cudaSetDevice(dev));
    double *d, h = 0;
    CK(cudaMalloc((void **)&d, sizeof(double)));
    CK(cudaMemsetAsync(d, 0, sizeof(double), st));
    k_sumsq<<<GRID1D(n), 256, 0, st>>>(p, n, d);
    CK(cudaMemcpyAsync(&h, d, sizeof(double), cudaMemcpyDeviceToHost, st));
    CK(cudaStreamSynchronize(st));
    CK(cudaFree(d));
    return h;
}

/* ============================ THAM SỐ MỘT THIẾT BỊ ============================ */

struct Params {      /* các khối [l0, l1) trên một GPU; Fl = số cột F cục bộ, F0 = cột F đầu tiên của lát */
    int H, l0, l1, Fl, F0;
    float *W1, *b1, *W2, *dW1, *db1, *dW2;
    float *w1(int l) const { return W1 + (size_t)(l - l0) * H * Fl; }
    float *bb1(int l) const { return b1 + (size_t)(l - l0) * Fl; }
    float *w2(int l) const { return W2 + (size_t)(l - l0) * Fl * H; }
    float *g1(int l) const { return dW1 + (size_t)(l - l0) * H * Fl; }
    float *gb1(int l) const { return db1 + (size_t)(l - l0) * Fl; }
    float *g2(int l) const { return dW2 + (size_t)(l - l0) * Fl * H; }
    size_t n1() const { return (size_t)(l1 - l0) * H * Fl; }
    size_t nb() const { return (size_t)(l1 - l0) * Fl; }
};

static void alloc_params(const MpConfig &c, Params &p, int dev, int l0, int l1, int Fl, int F0) {
    CK(cudaSetDevice(dev));
    p.H = c.H; p.l0 = l0; p.l1 = l1; p.Fl = Fl; p.F0 = F0;
    p.W1 = dmalloc(p.n1()); p.dW1 = dmalloc(p.n1());
    p.W2 = dmalloc(p.n1()); p.dW2 = dmalloc(p.n1());
    p.b1 = dmalloc(p.nb()); p.db1 = dmalloc(p.nb());
    for (int l = l0; l < l1; l++) {
        k_init_mat<<<GRID1D((size_t)c.H * Fl), 256>>>(p.w1(l), c.H, Fl, c.F, 0, F0, c.seed, MP_TAG_W1, l, c.scale_w1);
        k_init_vec<<<GRID1D((size_t)Fl), 256>>>(p.bb1(l), Fl, F0, c.seed, MP_TAG_B1, l, c.scale_b1);
        k_init_mat<<<GRID1D((size_t)Fl * c.H), 256>>>(p.w2(l), Fl, c.H, c.H, F0, 0, c.seed, MP_TAG_W2, l, c.scale_w2);
    }
    CK(cudaGetLastError());
}

static void zero_grads(const Params &p, cudaStream_t st) {
    CK(cudaMemsetAsync(p.dW1, 0, p.n1() * sizeof(float), st));
    CK(cudaMemsetAsync(p.dW2, 0, p.n1() * sizeof(float), st));
    CK(cudaMemsetAsync(p.db1, 0, p.nb() * sizeof(float), st));
}
static void sgd_update(const Params &p, float lr, cudaStream_t st) {
    k_sgd<<<GRID1D(p.n1()), 256, 0, st>>>(p.W1, p.dW1, lr, p.n1());
    k_sgd<<<GRID1D(p.n1()), 256, 0, st>>>(p.W2, p.dW2, lr, p.n1());
    k_sgd<<<GRID1D(p.nb()), 256, 0, st>>>(p.b1, p.db1, lr, p.nb());
}
static double grad_sumsq(int dev, cudaStream_t st, const Params &p) {
    return dev_sumsq(dev, st, p.dW1, p.n1()) + dev_sumsq(dev, st, p.dW2, p.n1()) + dev_sumsq(dev, st, p.db1, p.nb());
}

/* Forward một khối trên `rows` dòng:  U = ReLU(X·W1 + b1);  Y = U·W2   (chưa cộng nhánh residual). */
static void fwd_block(cudaStream_t st, const Params &p, int l, const float *X, float *U, float *Y, int rows) {
    gemm<false, false>(st, rows, p.Fl, p.H, X, p.H, p.w1(l), p.Fl, U, p.Fl, 0.f);
    k_bias_relu<<<GRID1D((size_t)rows * p.Fl), 256, 0, st>>>(U, p.bb1(l), rows, p.Fl);
    gemm<false, false>(st, rows, p.H, p.Fl, U, p.Fl, p.w2(l), p.H, Y, p.H, 0.f);
}
/* Backward một khối. dY: gradient theo đầu ra (rows×H). Cộng dồn dW1, db1, dW2; ghi dX = dZ·W1ᵀ (chưa cộng residual). */
static void bwd_block(cudaStream_t st, const Params &p, int l, const float *X, const float *U, const float *dY,
                      float *dX, float *tmpDU, int rows) {
    gemm<true, false>(st, p.Fl, p.H, rows, U, p.Fl, dY, p.H, p.g2(l), p.H, 1.f);        /* dW2 += Uᵀ·dY  */
    gemm<false, true>(st, rows, p.Fl, p.H, dY, p.H, p.w2(l), p.H, tmpDU, p.Fl, 0.f);    /* dU = dY·W2ᵀ   */
    k_relu_mask<<<GRID1D((size_t)rows * p.Fl), 256, 0, st>>>(tmpDU, U, (size_t)rows * p.Fl);   /* dZ */
    k_colsum_acc<<<(p.Fl + 255) / 256, 256, 0, st>>>(tmpDU, p.gb1(l), rows, p.Fl);       /* db1 += Σ dZ   */
    gemm<true, false>(st, p.H, p.Fl, rows, X, p.H, tmpDU, p.Fl, p.g1(l), p.Fl, 1.f);    /* dW1 += Xᵀ·dZ  */
    gemm<false, true>(st, rows, p.H, p.Fl, tmpDU, p.Fl, p.w1(l), p.Fl, dX, p.H, 0.f);   /* dX = dZ·W1ᵀ   */
}

/* ============================ PIPELINE PARALLELISM ============================ */

struct Pipeline {
    MpConfig c;
    int S, M, Bmb;
    std::vector<cudaStream_t> st;
    std::vector<Params> P;
    std::vector<float *> X, U, dYr, bufA, bufB, tmpDU;
    std::vector<std::vector<cudaEvent_t>> evF, evB;     /* evF[s][m]: s đã gửi xong activation m cho s+1; evB: gửi gradient cho s-1 */

    float *Xp(int s, int j, int m) { return X[s] + ((size_t)j * M + m) * Bmb * c.H; }
    float *Up(int s, int j, int m) { return U[s] + ((size_t)j * M + m) * Bmb * c.F; }
    float *dYrp(int s, int m) { return dYr[s] + (size_t)m * Bmb * c.H; }
    int nb(int s) const { return P[s].l1 - P[s].l0; }

    void setup(const MpConfig &cfg, int S_, int M_) {
        c = cfg; S = S_; M = M_; Bmb = c.B / M;
        st.resize(S); P.resize(S); X.resize(S); U.resize(S); dYr.assign(S, nullptr); bufA.resize(S); bufB.resize(S);
        tmpDU.resize(S);
        evF.assign(S, std::vector<cudaEvent_t>(M)); evB.assign(S, std::vector<cudaEvent_t>(M));
        int base = c.L / S, rem = c.L % S, l = 0;
        for (int s = 0; s < S; s++) {
            int cnt = base + (s < rem ? 1 : 0);
            CK(cudaSetDevice(s));
            CK(cudaStreamCreate(&st[s]));
            alloc_params(c, P[s], s, l, l + cnt, c.F, 0);
            l += cnt;
            X[s] = dmalloc((size_t)(cnt + 1) * M * Bmb * c.H);
            U[s] = dmalloc((size_t)cnt * M * Bmb * c.F);
            if (s < S - 1) dYr[s] = dmalloc((size_t)M * Bmb * c.H);
            bufA[s] = dmalloc((size_t)Bmb * c.H); bufB[s] = dmalloc((size_t)Bmb * c.H);
            tmpDU[s] = dmalloc((size_t)Bmb * c.F);
            for (int m = 0; m < M; m++) {
                CK(cudaEventCreateWithFlags(&evF[s][m], cudaEventDisableTiming));
                CK(cudaEventCreateWithFlags(&evB[s][m], cudaEventDisableTiming));
            }
        }
        CK(cudaSetDevice(0));      /* đầu vào: dòng toàn cục [m·Bmb, (m+1)·Bmb) của batch, sinh bằng cùng hàm băm như bản tuần tự */
        for (int m = 0; m < M; m++)
            k_init_vec<<<GRID1D((size_t)Bmb * c.H), 256>>>(Xp(0, 0, m), (size_t)Bmb * c.H, (size_t)m * Bmb * c.H, c.seed, MP_TAG_X, 0, 1.0f);
        for (int s = 0; s < S; s++) { CK(cudaSetDevice(s)); CK(cudaDeviceSynchronize()); }
    }

    void step(bool update, double *loss, double *gnorm) {
        const size_t bh = (size_t)Bmb * c.H;
        for (int s = 0; s < S; s++) { CK(cudaSetDevice(s)); zero_grads(P[s], st[s]); }
        /* ---- forward: micro-batch m lần lượt đi qua các stage; GPU s xử lý m+1 trong lúc GPU s+1 xử lý m ---- */
        for (int m = 0; m < M; m++)
            for (int s = 0; s < S; s++) {
                CK(cudaSetDevice(s));
                if (s > 0) CK(cudaStreamWaitEvent(st[s], evF[s - 1][m], 0));
                for (int j = 0; j < nb(s); j++) {
                    fwd_block(st[s], P[s], P[s].l0 + j, Xp(s, j, m), Up(s, j, m), Xp(s, j + 1, m), Bmb);
                    k_add<<<GRID1D(bh), 256, 0, st[s]>>>(Xp(s, j + 1, m), Xp(s, j, m), bh);      /* + residual */
                }
                if (s < S - 1) {
                    CK(cudaMemcpyPeerAsync(Xp(s + 1, 0, m), s + 1, Xp(s, nb(s), m), s, bh * sizeof(float), st[s]));
                    CK(cudaEventRecord(evF[s][m], st[s]));
                }
            }
        /* ---- backward: micro-batch ngược lại, từ stage cuối về stage đầu ---- */
        const float inv = 1.0f / ((float)c.B * (float)c.H);
        for (int m = M - 1; m >= 0; m--)
            for (int s = S - 1; s >= 0; s--) {
                CK(cudaSetDevice(s));
                const float *cur;
                if (s == S - 1) {
                    k_scale<<<GRID1D(bh), 256, 0, st[s]>>>(bufA[s], Xp(s, nb(s), m), inv, bh);
                    cur = bufA[s];
                } else {
                    CK(cudaStreamWaitEvent(st[s], evB[s + 1][m], 0));
                    cur = dYrp(s, m);
                }
                float *other = (cur == bufA[s]) ? bufB[s] : bufA[s];
                for (int j = nb(s) - 1; j >= 0; j--) {
                    bwd_block(st[s], P[s], P[s].l0 + j, Xp(s, j, m), Up(s, j, m), cur, other, tmpDU[s], Bmb);
                    k_add<<<GRID1D(bh), 256, 0, st[s]>>>(other, cur, bh);                        /* + nhánh residual */
                    cur = other;
                    other = (cur == bufA[s]) ? bufB[s] : bufA[s];
                }
                if (s > 0) {
                    CK(cudaMemcpyPeerAsync(dYrp(s - 1, m), s - 1, cur, s, bh * sizeof(float), st[s]));
                    CK(cudaEventRecord(evB[s][m], st[s]));
                }
            }
        if (update)
            for (int s = 0; s < S; s++) { CK(cudaSetDevice(s)); sgd_update(P[s], c.lr, st[s]); }
        for (int s = 0; s < S; s++) { CK(cudaSetDevice(s)); CK(cudaStreamSynchronize(st[s])); CK(cudaGetLastError()); }
        if (loss) {
            double ss = 0;
            for (int m = 0; m < M; m++) ss += dev_sumsq(S - 1, st[S - 1], Xp(S - 1, nb(S - 1), m), bh);
            *loss = 0.5 * ss / ((double)c.B * c.H);
        }
        if (gnorm) {
            double g = 0;
            for (int s = 0; s < S; s++) g += grad_sumsq(s, st[s], P[s]);
            *gnorm = std::sqrt(g);
        }
    }
};

/* ============================ TENSOR PARALLELISM ============================ */

struct Tensor {
    MpConfig c;
    int T;
    std::vector<cudaStream_t> st;
    std::vector<Params> P;
    std::vector<float *> X, U, bufA, bufB, tmpDU, recv;

    float *Xp(int d, int l) { return X[d] + (size_t)l * c.B * c.H; }
    float *Up(int d, int l) { return U[d] + (size_t)l * c.B * (c.F / T); }

    void sync_all() { for (int d = 0; d < T; d++) { CK(cudaSetDevice(d)); CK(cudaStreamSynchronize(st[d])); } }

    /* All-reduce (tổng) của T bản `buf[d]` (n float) bằng recursive doubling: log2(T) vòng, mỗi vòng GPU d trao đổi
     * toàn bộ buffer với GPU d^(1<<r) rồi cộng. Đồng bộ host giữa các pha để không có đua dữ liệu. */
    void all_reduce(std::vector<float *> &buf, size_t n) {
        if (T == 1) return;
        sync_all();
        for (int r = 0; (1 << r) < T; r++) {
            for (int d = 0; d < T; d++) {
                int q = d ^ (1 << r);
                CK(cudaSetDevice(d));
                CK(cudaMemcpyPeerAsync(recv[q], q, buf[d], d, n * sizeof(float), st[d]));
            }
            sync_all();
            for (int d = 0; d < T; d++) {
                CK(cudaSetDevice(d));
                k_add<<<GRID1D(n), 256, 0, st[d]>>>(buf[d], recv[d], n);
            }
            sync_all();
        }
    }

    void setup(const MpConfig &cfg, int T_) {
        c = cfg; T = T_;
        const int Fl = c.F / T;
        st.resize(T); P.resize(T); X.resize(T); U.resize(T); bufA.resize(T); bufB.resize(T); tmpDU.resize(T); recv.resize(T);
        for (int d = 0; d < T; d++) {
            CK(cudaSetDevice(d));
            CK(cudaStreamCreate(&st[d]));
            alloc_params(c, P[d], d, 0, c.L, Fl, d * Fl);
            X[d] = dmalloc((size_t)(c.L + 1) * c.B * c.H);
            U[d] = dmalloc((size_t)c.L * c.B * Fl);
            bufA[d] = dmalloc((size_t)c.B * c.H); bufB[d] = dmalloc((size_t)c.B * c.H);
            tmpDU[d] = dmalloc((size_t)c.B * Fl);
            recv[d] = dmalloc((size_t)c.B * c.H);
            k_init_vec<<<GRID1D((size_t)c.B * c.H), 256>>>(Xp(d, 0), (size_t)c.B * c.H, 0, c.seed, MP_TAG_X, 0, 1.0f);
            CK(cudaDeviceSynchronize());
        }
    }

    void step(bool update, double *loss, double *gnorm) {
        const size_t bh = (size_t)c.B * c.H;
        const int Fl = c.F / T;
        std::vector<float *> ptr(T);
        for (int d = 0; d < T; d++) { CK(cudaSetDevice(d)); zero_grads(P[d], st[d]); }
        for (int l = 0; l < c.L; l++) {                                            /* forward */
            for (int d = 0; d < T; d++) {
                CK(cudaSetDevice(d));
                fwd_block(st[d], P[d], l, Xp(d, l), Up(d, l), Xp(d, l + 1), c.B);   /* Y riêng phần của GPU d */
                ptr[d] = Xp(d, l + 1);
            }
            all_reduce(ptr, bh);                                                   /* Y = Σ_d Y_d */
            for (int d = 0; d < T; d++) {
                CK(cudaSetDevice(d));
                k_add<<<GRID1D(bh), 256, 0, st[d]>>>(Xp(d, l + 1), Xp(d, l), bh);   /* + residual */
            }
        }
        const float inv = 1.0f / ((float)c.B * (float)c.H);
        std::vector<float *> cur(T), other(T);
        for (int d = 0; d < T; d++) {
            CK(cudaSetDevice(d));
            k_scale<<<GRID1D(bh), 256, 0, st[d]>>>(bufA[d], Xp(d, c.L), inv, bh);
            cur[d] = bufA[d]; other[d] = bufB[d];
        }
        for (int l = c.L - 1; l >= 0; l--) {                                       /* backward */
            for (int d = 0; d < T; d++) {
                CK(cudaSetDevice(d));
                bwd_block(st[d], P[d], l, Xp(d, l), Up(d, l), cur[d], other[d], tmpDU[d], c.B);
            }
            all_reduce(other, bh);                                                 /* dX = Σ_d dX_d */
            for (int d = 0; d < T; d++) {
                CK(cudaSetDevice(d));
                k_add<<<GRID1D(bh), 256, 0, st[d]>>>(other[d], cur[d], bh);         /* + nhánh residual */
                std::swap(cur[d], other[d]);
            }
        }
        if (update) for (int d = 0; d < T; d++) { CK(cudaSetDevice(d)); sgd_update(P[d], c.lr, st[d]); }
        sync_all();
        for (int d = 0; d < T; d++) { CK(cudaSetDevice(d)); CK(cudaGetLastError()); }
        (void)Fl;
        if (loss) *loss = 0.5 * dev_sumsq(0, st[0], Xp(0, c.L), bh) / ((double)c.B * c.H);
        if (gnorm) {
            double g = 0;                               /* các lát rời nhau nên tổng bình phương = của toàn bộ mô hình */
            for (int d = 0; d < T; d++) g += grad_sumsq(d, st[d], P[d]);
            *gnorm = std::sqrt(g);
        }
    }
};

/* ================================== MAIN ================================== */

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
    MpConfig c = {256, 0, 4, 64, 1, 1e-3f, 0.f, 0.f, 0.1f};
    std::string mode = "pipeline";
    int gpus = 1, micro = 1, steps = 3, warmup = 1, check = 0;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto nexti = [&](void) { if (i + 1 >= argc) { fprintf(stderr, "thiếu giá trị cho %s\n", a.c_str()); exit(2); } return atoi(argv[++i]); };
        if (a == "--mode") { if (i + 1 >= argc) return 2; mode = argv[++i]; }
        else if (a == "--gpus") gpus = nexti();
        else if (a == "--micro") micro = nexti();
        else if (a == "--H") c.H = nexti();
        else if (a == "--L") c.L = nexti();
        else if (a == "--B") c.B = nexti();
        else if (a == "--seed") c.seed = (uint64_t)nexti();
        else if (a == "--steps") steps = nexti();
        else if (a == "--warmup") warmup = nexti();
        else if (a == "--lr") { if (i + 1 >= argc) return 2; c.lr = (float)atof(argv[++i]); }
        else if (a == "--check") check = 1;
        else { fprintf(stderr, "đối số không hợp lệ: %s\n", a.c_str()); return 2; }
    }
    c.F = 4 * c.H;
    c.scale_w1 = 1.0f / sqrtf((float)c.H);
    c.scale_w2 = 1.0f / sqrtf((float)c.F);
    if (mode != "pipeline" && mode != "tensor") { fprintf(stderr, "--mode phải là pipeline hoặc tensor\n"); return 2; }
    if (mode == "tensor") micro = 0;
    int ndev = 0;
    CK(cudaGetDeviceCount(&ndev));
    if (gpus < 1 || gpus > ndev) { fprintf(stderr, "cần %d GPU nhưng chỉ thấy %d\n", gpus, ndev); return 2; }
    if (mode == "pipeline") {
        if (micro < 1 || c.B % micro) { fprintf(stderr, "B=%d phải chia hết cho --micro=%d\n", c.B, micro); return 2; }
        if (c.L < gpus) { fprintf(stderr, "cần L >= số GPU (stage)\n"); return 2; }
    } else {
        if ((gpus & (gpus - 1)) || c.F % gpus) { fprintf(stderr, "tensor parallel cần số GPU là luỹ thừa của 2 và chia hết F\n"); return 2; }
    }
    snprintf(g_desc, sizeof g_desc, "impl=cuda mode=%s gpus=%d micro=%d H=%d L=%d B=%d params_M=%.1f", mode.c_str(), gpus,
             micro, c.H, c.L, c.B, mp_param_count(&c) / 1e6);
    for (int i = 0; i < gpus; i++)                       /* truy cập ngang hàng giữa các GPU (nếu phần cứng cho phép) */
        for (int j = 0; j < gpus; j++) {
            if (i == j) continue;
            int can = 0;
            CK(cudaDeviceCanAccessPeer(&can, i, j));
            if (can) { CK(cudaSetDevice(i)); cudaError_t e = cudaDeviceEnablePeerAccess(j, 0); if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) CK(e); cudaGetLastError(); }
        }

    Pipeline pp; Tensor tp;
    if (mode == "pipeline") pp.setup(c, gpus, micro); else tp.setup(c, gpus);
    auto run = [&](bool update, double *loss, double *gnorm) {
        if (mode == "pipeline") pp.step(update, loss, gnorm); else tp.step(update, loss, gnorm);
    };
    double loss = 0, gnorm = 0;
    if (check) {                                          /* một lượt forward+backward, không cập nhật */
        run(false, &loss, &gnorm);
        printf("CHECK impl=cuda mode=%s gpus=%d micro=%d H=%d L=%d B=%d loss=%.9e gradnorm=%.9e\n", mode.c_str(), gpus, micro,
               c.H, c.L, c.B, loss, gnorm);
        return 0;
    }
    for (int i = 0; i < warmup; i++) run(true, nullptr, nullptr);
    double t0 = now_s();
    for (int i = 0; i < steps; i++) run(true, nullptr, nullptr);
    double step_s = (now_s() - t0) / steps;
    run(false, &loss, &gnorm);
    double mem_max = 0, mem_min = 1e30;
    for (int d = 0; d < gpus; d++) {
        size_t fr, tot;
        CK(cudaSetDevice(d)); CK(cudaMemGetInfo(&fr, &tot));
        double u = (double)(tot - fr) / 1e9;
        if (u > mem_max) mem_max = u;
        if (u < mem_min) mem_min = u;
    }
    /* Lượng dữ liệu chuyển giữa GPU mỗi bước: pipeline = activation + gradient qua mỗi ranh giới stage;
     * tensor = 2L all-reduce, mỗi all-reduce mỗi GPU gửi log2(T) lần B·H float. */
    double comm_mb = 0;
    if (mode == "pipeline") comm_mb = 2.0 * (gpus - 1) * (double)c.B * c.H * 4 / 1e6;
    else { int rounds = 0; while ((1 << rounds) < gpus) rounds++; comm_mb = 2.0 * c.L * rounds * (double)c.B * c.H * 4 / 1e6 * gpus; }
    printf("RESULT %s step_ms=%.3f samples_s=%.1f tflops=%.2f mem_GB_max=%.2f mem_GB_min=%.2f comm_MB=%.1f loss=%.6e status=ok\n",
           g_desc, step_s * 1e3, c.B / step_s, mp_step_flops(&c) / step_s / 1e12, mem_max, mem_min, comm_mb, loss);
    return 0;
}
