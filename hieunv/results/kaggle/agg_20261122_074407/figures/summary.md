# Tóm tắt kết quả — kaggle / agg_20261122_074407

- CPU: Intel(R) Xeon(R) CPU @ 2.00GHz (torch dùng tối đa 2 luồng)
- GPU: Tesla T4 · CUDA 13.0 · cuDNN 92700
- PyTorch 2.11.0+cu130 · Python 3.13.16 · Linux-6.18.55+-x86_64-with-glibc2.39

## Thí nghiệm 1 — lớp convolution, cài đặt kiểu Caffe (so với Hình 1–2 của paper)

| Cấu hình | Độ dốc (ms/ảnh) | R² | Tỷ lệ độ dốc so với baseline |
|---|---|---|---|
| *Paper: Xeon E5-2670 (CPU)* | *6.45* | | *1×* |
| *Paper: GRID K520 (GPU)* | *0.16* | | *40.3×* |
| CPU · 1 luồng | 1.879 | 0.9999 | 1.0× |
| GPU · Tesla T4 | 0.0677 | 0.9999 | 27.8× |
| CPU · 2 luồng | 1.265 | 0.9999 | 1.5× |

Speedup theo batch size (thời gian baseline / thời gian cấu hình):

| Batch | CPU · 2 luồng | GPU · Tesla T4 |
|---|---|---|
| 1 | 1.46× | 18.62× |
| 8 | 1.39× | 21.43× |
| 16 | 1.47× | 26.14× |
| 32 | 1.48× | 26.95× |
| 64 | 1.49× | 27.92× |
| 96 | 1.49× | 27.82× |
| 128 | 1.50× | 27.18× |
| 160 | 1.51× | 27.74× |
| 192 | 1.49× | 27.69× |
| 224 | 1.48× | 27.66× |
| 256 | 1.48× | 27.60× |

## Thí nghiệm 2 — lớp fully-connected (so với Hình 3–4 của paper)

| Cấu hình | Phép | Số mũ k (t ∝ nᵏ) | k nếu khớp từ n ≥ 1024 | Thời gian ở n lớn nhất (ms) | GFLOP/s | GB/s |
|---|---|---|---|---|---|---|
| CPU · 1 luồng | matvec (lý thuyết k=2) | 2.07 | 2.21 | 87.302 | 6.1 | 12.3 |
| GPU · Tesla T4 | matvec (lý thuyết k=2) | 1.94 | 1.75 | 4.016 | 133.7 | 267.4 |
| CPU · 2 luồng | matvec (lý thuyết k=2) | 2.09 | 2.21 | 45.763 | 11.7 | 23.5 |
| CPU · 1 luồng | gemm (lý thuyết k=3) | 3.00 | 3.00 | 1066.685 | 128.9 | 0.2 |
| GPU · Tesla T4 | gemm (lý thuyết k=3) | 3.13 | 3.13 | 43.909 | 3148.1 | 4.6 |
| CPU · 2 luồng | gemm (lý thuyết k=3) | 2.99 | 2.99 | 559.698 | 245.6 | 0.4 |

Speedup matvec theo n:

| n | CPU · 2 luồng | GPU · Tesla T4 |
|---|---|---|
| 256 | 0.86× | 0.43× |
| 512 | 1.91× | 1.48× |
| 1024 | 1.94× | 6.84× |
| 2048 | 1.83× | 8.78× |
| 4096 | 1.98× | 18.42× |
| 8192 | 1.93× | 20.30× |
| 12288 | 1.91× | 21.24× |
| 16384 | 1.91× | 21.74× |

Ghi chú: trục Hình 1 của paper là Time (ms) theo batch size, nên độ dốc là ms/ảnh. Con số tuyệt đối khác paper vì phần cứng khác đời; điều cần so là xu hướng.
