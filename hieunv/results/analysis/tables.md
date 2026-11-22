# Bảng số cho phân tích giai đoạn 5 (sinh bởi code/analyze.py)

## 1. Speedup CPU-1 luồng / GPU theo batch (conv kiểu Caffe, Kaggle) và bài báo

| batch | tỷ lệ đo lại (trung vị 3 lần) | nhỏ nhất–lớn nhất | đo lại / batch 256 | bài báo (Hình 2) | bài báo / batch 256 |
|---|---|---|---|---|---|
| 1 | 18.67 | 17.9–19.3 | 0.68 |  |  |
| 8 | 24.92 | 14.2–25.2 | 0.89 |  |  |
| 16 | 26.66 | 24.9–26.9 | 0.95 | 37.01 | 0.91 |
| 32 | 27.10 | 26.1–27.6 | 0.98 | 37.67 | 0.92 |
| 64 | 27.47 | 27.5–28.8 | 1.01 | 38.28 | 0.94 |
| 96 | 27.68 | 26.8–29.0 | 1.01 |  |  |
| 128 | 27.43 | 26.3–27.8 | 0.98 | 39.76 | 0.97 |
| 160 | 27.24 | 27.0–29.0 | 1.00 |  |  |
| 192 | 27.38 | 27.1–28.6 | 1.00 |  |  |
| 224 | 27.29 | 26.9–28.8 | 1.00 |  |  |
| 256 | 27.39 | 27.1–28.3 | 1.00 | 40.80 | 1.00 |

## 2. Độ dốc conv theo từng lần chạy (Kaggle)

| run | cpu_slope | gpu_slope | ratio | r2_cpu | r2_gpu |
|---|---|---|---|---|---|
| run1 | 1.8552 | 0.0648 | 28.6386 | 0.9996 | 0.9999 |
| run2 | 1.8870 | 0.0689 | 27.3737 | 0.9999 | 0.9999 |
| run3 | 1.8957 | 0.0694 | 27.3241 | 0.9999 | 0.9995 |

Bài báo: 6.45 và 0.16 ms/ảnh, tỷ lệ 40.3. Tốc độ FLOP của conv: CPU bài báo 10.5, GPU bài báo 423 GFLOP/s; CPU-1 luồng đo lại 36.2, GPU đo lại 1001 GFLOP/s.

## 3. Mô hình t_GPU(B) = a + b·B cho conv kiểu Caffe (khớp theo sai số tương đối)

| run | gpu_fixed_us | gpu_us_per_img | cpu_ms_per_img | max_err_pct |
|---|---|---|---|---|
| run1 | 22.7 | 64.8 | 1.84 | 6.6 |
| run2 | 23.7 | 68.8 | 1.88 | 4.05 |
| run3 | 30 | 71.6 | 1.87 | 5.56 |

max_err_pct: sai lệch lớn nhất giữa speedup do mô hình dự đoán và đo được (không tính lần 3 batch 8 bị hạ xung).

## 4. Speedup matvec CPU-1 luồng / GPU theo n (Kaggle) và bài báo

| n | đo lại (trung vị 3 lần) | nhỏ nhất–lớn nhất | bài báo (Hình 4) |
|---|---|---|---|
| 256 | 0.42 | 0.40–0.48 | 3.14 |
| 512 | 1.43 | 1.33–1.69 | 7.74 |
| 1024 | 7.85 | 4.82–7.85 | 44.45 |
| 2048 | 8.86 | 8.42–9.07 | 28.65 |
| 4096 | 18.34 | 18.13–18.79 | 52.50 |
| 8192 | 20.28 | 20.23–20.39 | 56.61 |
| 12288 | 21.22 | 21.14–21.35 |  |
| 16384 | 21.74 | 21.68–21.80 |  |

## 5. Matvec: số tuyệt đối so với bài báo (GB/s tính theo 4 byte mỗi phần tử ma trận)

| n | bài báo CPU (ms) | bài báo CPU ns/phần tử | bài báo CPU GB/s | bài báo GPU GB/s (suy ra) | đo lại CPU-1 luồng ms | đo lại CPU GB/s | đo lại GPU GB/s | laptop CPU GB/s |
|---|---|---|---|---|---|---|---|---|
| 2048 | 62.6 | 14.9 | 0.268 | 7.68 | 0.775 | 21.7 | 190 | 34 |
| 4096 | 230 | 13.7 | 0.292 | 15.3 | 4.98 | 13.5 | 248 | 28.4 |
| 8192 | 967 | 14.4 | 0.278 | 15.7 | 20.8 | 12.9 | 262 | 27.9 |

## 5b. Giả thuyết: FC của bài báo là (n×n)·(n×B) ("matrix-matrix"), với B vector

| n | CPU GFLOP/s nếu B = 1 | B để CPU bằng conv (10,5 GFLOP/s) | CPU GFLOP/s nếu B = 64 | GPU GFLOP/s nếu B = 64 | CPU GFLOP/s nếu B = 76 | GPU GFLOP/s nếu B = 76 |
|---|---|---|---|---|---|---|
| 2048 | 0.134 | 78.4 | 8.58 | 246 | 10.2 | 292 |
| 4096 | 0.146 | 71.9 | 9.35 | 491 | 11.1 | 583 |
| 8192 | 0.139 | 75.6 | 8.88 | 503 | 10.5 | 597 |

Trần danh định của K520 (1 GPU) là 2289 GFLOP/s FP32.

## 6. Băng thông matvec theo n

| n | Kaggle CPU-1 luồng GB/s | Kaggle CPU-2 luồng GB/s | Kaggle GPU GB/s | laptop CPU-1 luồng GB/s | laptop CPU-8 luồng GB/s |
|---|---|---|---|---|---|
| 256 | 23.0 | 19.7 | 9.9 | 40.7 | 42.9 |
| 512 | 24.7 | 45.4 | 36.0 | 52.1 | 52.2 |
| 1024 | 20.0 | 38.7 | 136.0 | 40.5 | 39.4 |
| 2048 | 21.7 | 39.6 | 190.4 | 34.0 | 36.1 |
| 4096 | 13.5 | 26.6 | 248.2 | 28.4 | 29.5 |
| 8192 | 12.9 | 24.9 | 262.3 | 27.9 | 28.8 |
| 12288 | 12.6 | 24.0 | 266.7 | 28.1 | 28.8 |
| 16384 | 12.3 | 23.5 | 267.4 | 28.1 | 28.7 |

## 7. Chu kỳ GPU mỗi ảnh theo cách cài đặt (batch ≥ 64, trung vị 3 lần; xung nhịp trong đúng khoảng đo)

| impl | us_per_img | mhz | kcycles_per_img | gflops | peak_pct |
|---|---|---|---|---|---|
| caffe | 69.2 | 1377.4 | 95.3 | 979.4 | 13.9 |
| batched | 74.1 | 1042.5 | 77.2 | 913.9 | 17.1 |
| conv2d | 56.2 | 782.1 | 45.2 | 1204.9 | 29.6 |

### Từng lần

| run | impl | us_per_img | mhz | kcycles_per_img | gflops | peak_pct |
|---|---|---|---|---|---|---|
| run1 | caffe | 64.9 | 1467.6 | 95.3 | 1043.9 | 13.9 |
| run1 | batched | 67.8 | 1158.5 | 78.5 | 998.7 | 16.8 |
| run1 | conv2d | 49.7 | 919.5 | 45.8 | 1364.3 | 29.0 |
| run2 | caffe | 69.2 | 1377.4 | 95.5 | 979.4 | 13.9 |
| run2 | batched | 74.1 | 1042.5 | 77.2 | 913.9 | 17.1 |
| run2 | conv2d | 56.2 | 782.1 | 45.2 | 1204.9 | 30.1 |
| run3 | caffe | 69.8 | 1362.0 | 95.0 | 969.9 | 13.9 |
| run3 | batched | 75.6 | 1008.8 | 76.3 | 895.7 | 17.3 |
| run3 | conv2d | 59.2 | 756.0 | 44.8 | 1145.1 | 29.6 |

## 8. GEMM trên GPU: tỷ lệ so với trần FP32 ở đúng xung nhịp đo được

| run | n | mhz | value | peak_pct |
|---|---|---|---|---|
| run1 | 2048 | 757.5 | 3703.3 | 95.5 |
| run1 | 3072 | 685.3 | 3251.9 | 92.7 |
| run1 | 4096 | 704.6 | 3493.6 | 96.8 |
| run2 | 2048 | 687.0 | 3150.7 | 89.6 |
| run2 | 3072 | 585.0 | 2857.2 | 95.4 |
| run2 | 4096 | 604.6 | 2985.9 | 96.5 |
| run3 | 2048 | 756.0 | 3232.3 | 83.5 |
| run3 | 3072 | 586.4 | 2847.3 | 94.8 |
| run3 | 4096 | 598.7 | 2964.6 | 96.7 |

## 9. Matvec GPU: GB/s ở các xung nhịp SM khác nhau (xung nhịp bộ nhớ trong các khoảng đo đều 5000 MHz)

| run | n | mhz | value |
|---|---|---|---|
| run1 | 8192 | 1590.0 | 262.8 |
| run1 | 12288 | 1380.0 | 267.4 |
| run1 | 16384 | 780.0 | 268.1 |
| run2 | 8192 | 1095.0 | 262.1 |
| run2 | 12288 | 630.0 | 266.2 |
| run2 | 16384 | 630.0 | 267.3 |
| run3 | 8192 | 1500.0 | 261.9 |
| run3 | 12288 | 930.0 | 266.4 |
| run3 | 16384 | 667.5 | 266.9 |

## 10. Tăng tốc CPU: Kaggle 2 luồng (p=2)

| phép tính | S | E = S/p | f (Amdahl) |
|---|---|---|---|
| conv kiểu Caffe | 1.49 | 0.743 | 0.346 |
| conv batched | 1.74 | 0.87 | 0.15 |
| conv2d (oneDNN) | 1.93 | 0.965 | 0.0363 |
| GEMM n=4096 | 1.91 | 0.953 | 0.0494 |
| matvec n=16384 | 1.91 | 0.954 | 0.0484 |

## 11. Tăng tốc CPU: laptop 8 luồng (p=8)

| phép tính | S | E = S/p | f (Amdahl) |
|---|---|---|---|
| conv kiểu Caffe | 3.8 | 0.475 | 0.158 |
| conv batched | 3.19 | 0.399 | 0.215 |
| conv2d (oneDNN) | 3.06 | 0.383 | 0.23 |
| GEMM n=4096 | 4.5 | 0.563 | 0.111 |
| matvec n=16384 | 1.03 | 0.128 | 0.971 |

