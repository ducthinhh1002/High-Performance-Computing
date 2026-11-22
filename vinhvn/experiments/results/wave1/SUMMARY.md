# Đợt 1: E2 và E1 (20/11/2026)

- GPU: 1× H100 80GB HBM3, 8 CPU cho phía host. Job `wave1-gpu`, 58 giây.
- CPU: Xeon Platinum 8462Y+, 32 luồng (= quota cgroup), MKL. Job `wave1-cpu`, 3 phút.
- PyTorch 2.5.1, CUDA 12.4, cuDNN 9.1. FP32 với TF32 tắt, trừ khi ghi rõ.
- File gốc: `gpu/` và `cpu/` (CSV + meta.json), log: `wave1-gpu.log`, `wave1-cpu.log`.
  Trên hostPath: `/var/lib/hpc-dl-exp/results/{e1,e2}/`.

## E2: GPU so với CPU

| Phép đo | Kết quả | Đối chiếu bài gốc |
|---|---|---|
| Lớp conv của bài, hệ số góc theo batch | CPU 0,186 ms/ảnh, GPU 0,0028 ms/ảnh → **66×** | Bài: 6,45 / 0,16 → 40× (K520) |
| Lớp conv, tăng tốc theo batch | 18× (batch 1) → 65–69× (batch ≥ 256) | Cùng xu hướng: batch lớn tăng tốc nhiều hơn |
| Nhân ma trận × vector (FC như bài mô tả) | 0,6× (n=256) → 16× (n=16384); bị giới hạn băng thông bộ nhớ, GPU đạt 3,0 TB/s | Bài: khoảng 40×, tăng theo n |
| Nhân ma trận × ma trận | 4× → 19× (n=2048) → 9,5× (n=16384); GPU 53 TFLOPS FP32 | Bài không đo |
| Bước huấn luyện FP32 | 9–30×, tăng theo batch | Bài không đo |
| Bước huấn luyện TF32 / BF16 | tới 62× / 73× (ResNet-18, batch 512) | Bài không đo |
| CPU→GPU | 28 GB/s thường, 55 GB/s pinned | Bài không đo |

Bất thường cần kiểm tra: **VGG-16 với BF16 chậm hơn FP32 khoảng 2 lần** ở mọi batch. Giả thuyết
(chưa kiểm chứng): backward của `AdaptiveAvgPool2d` (phóng 1×1 lên 7×7 với ảnh 32×32) dùng atomic add trên BF16.

## E1: Convolution và GEMM

Tỉ lệ self time theo nhóm phép tính trong một bước huấn luyện (batch 128; chỉ đọc tỉ lệ, profiler làm tăng thời gian tuyệt đối):

| Mô hình | GPU | CPU |
|---|---|---|
| AlexNet (bài) | conv 80%, act/pool 7%, gemm 6%, optimizer 3% | conv 69%, act/pool 12%, gemm 7% |
| ResNet-18 | conv 84%, norm 11% | conv 80%, norm 12% |
| VGG-16 | conv 62%, gemm 17%, optimizer 11% | conv 49%, gemm 22%, elementwise 11%, optimizer 10% |

im2col tự cài (unfold + matmul), lớp conv của bài:

| | Phần matmul trong im2col | conv2d so với im2col tự cài |
|---|---|---|
| GPU | 10–18% (batch ≥ 8); unfold chiếm phần lớn | cuDNN nhanh hơn khoảng 6× |
| CPU | 23–31% | oneDNN nhanh hơn khoảng 2× (batch ≥ 8) |

Kernel cuDNN thực sự chạy cho AlexNet: `implicit_convolve_sgemm`, `precomputed_convolve_sgemm`,
`xmma fprop implicit_gemm`, và có `fft2d`. Không có bước im2col tường minh.

Ma trận sau unfold là 121 × **2916** (54 × 54 vị trí), không phải 54 × 121 như bài ghi.

Backward + optimizer chiếm 52–61% bước huấn luyện trên GPU, 63–80% trên CPU.

## Kết luận về các khẳng định của bài

1. "Convolution chiếm nhiều thời gian nhất": **đúng**, 49–84% tùy mô hình và thiết bị.
2. "Phép nhân ma trận trong im2col là phần tốn nhất" (§3.1): **không đúng** khi đo. Bước unfold tốn hơn,
   và thư viện hiện đại không dùng im2col tường minh.
3. "GPU nhanh hơn CPU khoảng 40×": tùy phép đo. Lớp conv 66×, FC dạng matrix-vector chỉ 16×,
   bước huấn luyện FP32 9–30×. Không có một con số chung.
