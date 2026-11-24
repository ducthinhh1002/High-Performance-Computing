/* Phần dùng chung cho bản tuần tự (C) và bản song song (CUDA).
 *
 * Mô hình: L khối  x_{l+1} = x_l + W2_l · ReLU(W1_l · x_l + b1_l),  W1: H×F, W2: F×H, F = 4H,
 * loss = 0.5 · mean(x_L²). Mọi trọng số và đầu vào được sinh bằng một hàm băm không trạng thái, nên
 * CPU và GPU, và mọi cách chia mô hình ra nhiều GPU, đều bắt đầu từ đúng cùng một giá trị. */
#ifndef MP_COMMON_H
#define MP_COMMON_H

#include <stddef.h>
#include <stdint.h>

#ifdef __CUDACC__
#define MP_HD __host__ __device__
#else
#define MP_HD
#endif

/* splitmix64: hàm băm 64 bit. Chỉ dùng số nguyên nên cho kết quả giống hệt trên CPU và GPU. */
static MP_HD inline uint64_t mp_mix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

enum { MP_TAG_W1 = 1, MP_TAG_B1 = 2, MP_TAG_W2 = 3, MP_TAG_X = 4 };

/* Số thực đều trong [-1, 1), xác định bởi (seed, tag, layer, idx). idx là chỉ số TOÀN CỤC của phần tử
 * trong ma trận gốc, nên một lát (shard) của ma trận nhận đúng giá trị của nó trong ma trận đầy đủ. */
static MP_HD inline float mp_urand(uint64_t seed, uint32_t tag, uint32_t layer, uint64_t idx) {
    uint64_t key = idx + ((uint64_t)layer << 36) + ((uint64_t)tag << 56);
    uint64_t h = mp_mix64(key ^ (seed * 0xD1B54A32D192ED03ULL));
    uint32_t m = (uint32_t)(h >> 40);                 /* 24 bit cao: chuyển sang float không làm tròn */
    return (float)m * (1.0f / 8388608.0f) - 1.0f;
}

typedef struct {
    int H;                 /* chiều ẩn */
    int F;                 /* chiều trong của khối, F = 4H */
    int L;                 /* số khối */
    int B;                 /* batch toàn cục (số dòng) */
    uint64_t seed;
    float lr;              /* learning rate SGD */
    float scale_w1;        /* 1/sqrt(H)  */
    float scale_w2;        /* 1/sqrt(F)  */
    float scale_b1;        /* 0.1 */
} MpConfig;

/* FLOP của một bước huấn luyện: forward 2 GEMM (4·B·H·F), backward 4 GEMM (8·B·H·F), mỗi khối. */
static inline double mp_step_flops(const MpConfig *c) {
    return 12.0 * (double)c->B * (double)c->H * (double)c->F * (double)c->L;
}
static inline double mp_param_count(const MpConfig *c) {
    return (double)c->L * (2.0 * (double)c->H * (double)c->F + (double)c->F);
}

#endif
