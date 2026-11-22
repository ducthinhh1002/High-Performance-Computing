# Tóm tắt kết quả — laptop / agg_20261122_152118

- CPU: AMD Ryzen 7 5800H with Radeon Graphics (torch dùng tối đa 8 luồng)
- GPU: không có
- PyTorch 2.15.0+cpu · Python 3.12.10 · Windows-10-10.0.19045-SP0

## Thí nghiệm 1 — lớp convolution, cài đặt kiểu Caffe (so với Hình 1–2 của paper)

| Cấu hình | Độ dốc (ms/ảnh) | R² | Tỷ lệ độ dốc so với baseline |
|---|---|---|---|
| *Paper: Xeon E5-2670 (CPU)* | *6.45* | | *1×* |
| *Paper: GRID K520 (GPU)* | *0.16* | | *40.3×* |
| CPU · 1 luồng | 2.002 | 1.0000 | 1.0× |
| CPU · 8 luồng | 0.5264 | 0.9998 | 3.8× |

Speedup theo batch size (thời gian baseline / thời gian cấu hình):

| Batch | CPU · 8 luồng |
|---|---|
| 1 | 4.06× |
| 8 | 4.03× |
| 16 | 3.78× |
| 32 | 3.80× |
| 64 | 3.84× |
| 96 | 3.72× |
| 128 | 3.77× |
| 160 | 3.72× |
| 192 | 3.79× |
| 224 | 3.83× |
| 256 | 3.82× |

## Thí nghiệm 2 — lớp fully-connected (so với Hình 3–4 của paper)

| Cấu hình | Phép | Số mũ k (t ∝ nᵏ) | k nếu khớp từ n ≥ 1024 | Thời gian ở n lớn nhất (ms) | GFLOP/s | GB/s |
|---|---|---|---|---|---|---|
| CPU · 1 luồng | matvec (lý thuyết k=2) | 2.01 | 2.13 | 38.370 | 14.0 | 28.1 |
| CPU · 8 luồng | matvec (lý thuyết k=2) | 2.02 | 2.12 | 37.398 | 14.4 | 28.7 |
| CPU · 1 luồng | gemm (lý thuyết k=3) | 3.01 | 3.01 | 1301.809 | 105.6 | 0.2 |
| CPU · 8 luồng | gemm (lý thuyết k=3) | 2.89 | 2.89 | 289.207 | 475.4 | 0.7 |

Speedup matvec theo n:

| n | CPU · 8 luồng |
|---|---|
| 256 | 1.06× |
| 512 | 1.00× |
| 1024 | 0.97× |
| 2048 | 1.08× |
| 4096 | 1.04× |
| 8192 | 1.03× |
| 12288 | 1.03× |
| 16384 | 1.03× |

Ghi chú: trục Hình 1 của paper là Time (ms) theo batch size, nên độ dốc là ms/ảnh. Con số tuyệt đối khác paper vì phần cứng khác đời; điều cần so là xu hướng.
