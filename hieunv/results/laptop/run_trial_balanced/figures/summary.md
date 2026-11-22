# Tóm tắt kết quả — laptop / run_trial_balanced

- CPU: AMD Ryzen 7 5800H with Radeon Graphics (torch dùng tối đa 8 luồng)
- GPU: không có
- PyTorch 2.15.0+cpu · Python 3.12.10 · Windows-10-10.0.19045-SP0

## Thí nghiệm 1 — lớp convolution, cài đặt kiểu Caffe (so với Hình 1–2 của paper)

| Cấu hình | Độ dốc (ms/ảnh) | R² | Tỷ lệ độ dốc so với baseline |
|---|---|---|---|
| *Paper: Xeon E5-2670 (CPU)* | *6.45* | | *1×* |
| *Paper: GRID K520 (GPU)* | *0.16* | | *40.3×* |
| CPU · 1 luồng | 2.025 | 0.9961 | 1.0× |
| CPU · 8 luồng | 0.5692 | 0.9999 | 3.6× |

Speedup theo batch size (thời gian baseline / thời gian cấu hình):

| Batch | CPU · 8 luồng |
|---|---|
| 1 | 3.87× |
| 8 | 3.84× |
| 16 | 3.46× |
| 32 | 3.66× |
| 64 | 3.74× |
| 96 | 4.14× |
| 128 | 3.83× |
| 160 | 3.58× |
| 192 | 3.59× |
| 224 | 3.55× |
| 256 | 3.56× |

## Thí nghiệm 2 — lớp fully-connected (so với Hình 3–4 của paper)

| Cấu hình | Phép | Số mũ k (t ∝ nᵏ) | Thời gian ở n lớn nhất (ms) | GFLOP/s | GB/s |
|---|---|---|---|---|---|
| CPU · 1 luồng | matvec (lý thuyết k=2) | 2.08 | 52.534 | 10.2 | 20.4 |
| CPU · 8 luồng | matvec (lý thuyết k=2) | 2.11 | 51.850 | 10.4 | 20.7 |
| CPU · 1 luồng | gemm (lý thuyết k=3) | 2.99 | 1294.203 | 106.2 | 0.2 |
| CPU · 8 luồng | gemm (lý thuyết k=3) | 2.79 | 277.810 | 494.7 | 0.7 |

Speedup matvec theo n:

| n | CPU · 8 luồng |
|---|---|
| 256 | 1.06× |
| 512 | 1.06× |
| 1024 | 1.07× |
| 2048 | 1.12× |
| 4096 | 1.06× |
| 8192 | 1.01× |
| 12288 | 1.00× |
| 16384 | 1.01× |

Ghi chú: paper không ghi đơn vị độ dốc; ở đây hiểu là ms/ảnh. Con số tuyệt đối khác paper vì phần cứng khác đời; điều cần so là xu hướng.
