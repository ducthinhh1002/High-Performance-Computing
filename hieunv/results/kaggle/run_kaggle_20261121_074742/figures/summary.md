# Tóm tắt kết quả — kaggle / run_kaggle_20261121_074742

- CPU: Intel(R) Xeon(R) CPU @ 2.00GHz (torch dùng tối đa 2 luồng)
- GPU: Tesla T4 · CUDA 13.0 · cuDNN 92700
- PyTorch 2.11.0+cu130 · Python 3.13.16 · Linux-6.18.55+-x86_64-with-glibc2.39

## Thí nghiệm 1 — lớp convolution, cài đặt kiểu Caffe (so với Hình 1–2 của paper)

| Cấu hình | Độ dốc (ms/ảnh) | R² | Tỷ lệ độ dốc so với baseline |
|---|---|---|---|
| *Paper: Xeon E5-2670 (CPU)* | *6.45* | | *1×* |
| *Paper: GRID K520 (GPU)* | *0.16* | | *40.3×* |
| CPU · 1 luồng | 2.012 | 0.9999 | 1.0× |
| GPU · Tesla T4 | 0.06553 | 0.9766 | 30.7× |
| CPU · 2 luồng | 1.291 | 0.9996 | 1.6× |

Speedup theo batch size (thời gian baseline / thời gian cấu hình):

| Batch | CPU · 2 luồng | GPU · Tesla T4 |
|---|---|---|
| 1 | 1.60× | 11.60× |
| 8 | 1.54× | 15.33× |
| 16 | 1.54× | 17.18× |
| 32 | 1.57× | 18.61× |
| 64 | 1.43× | 17.96× |
| 96 | 1.58× | 21.04× |
| 128 | 1.52× | 28.61× |
| 160 | 1.54× | 28.74× |
| 192 | 1.53× | 29.08× |
| 224 | 1.58× | 28.40× |
| 256 | 1.55× | 29.07× |

## Thí nghiệm 2 — lớp fully-connected (so với Hình 3–4 của paper)

| Cấu hình | Phép | Số mũ k (t ∝ nᵏ) | Thời gian ở n lớn nhất (ms) | GFLOP/s | GB/s |
|---|---|---|---|---|---|
| CPU · 1 luồng | matvec (lý thuyết k=2) | 2.09 | 86.777 | 6.2 | 12.4 |
| GPU · Tesla T4 | matvec (lý thuyết k=2) | 1.95 | 4.021 | 133.5 | 267.0 |
| CPU · 2 luồng | matvec (lý thuyết k=2) | 2.07 | 45.946 | 11.7 | 23.4 |
| CPU · 1 luồng | gemm (lý thuyết k=3) | 2.98 | 1106.381 | 124.2 | 0.2 |
| GPU · Tesla T4 | gemm (lý thuyết k=3) | 2.67 | 35.261 | 3897.8 | 5.7 |
| CPU · 2 luồng | gemm (lý thuyết k=3) | 2.99 | 576.224 | 238.5 | 0.3 |

Speedup matvec theo n:

| n | CPU · 2 luồng | GPU · Tesla T4 |
|---|---|---|
| 256 | 1.14× | 0.48× |
| 512 | 1.61× | 1.47× |
| 1024 | 2.02× | 7.87× |
| 2048 | 2.08× | 9.59× |
| 4096 | 1.84× | 17.77× |
| 8192 | 1.88× | 20.66× |
| 12288 | 1.91× | 21.48× |
| 16384 | 1.89× | 21.58× |

Ghi chú: paper không ghi đơn vị độ dốc; ở đây hiểu là ms/ảnh. Con số tuyệt đối khác paper vì phần cứng khác đời; điều cần so là xu hướng.
