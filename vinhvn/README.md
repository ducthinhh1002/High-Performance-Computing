# Model parallelism trên nhiều GPU: bản tuần tự (C) và bản song song (CUDA) tự cài đặt

Phần của `vinhvn` cho đề tài "Parallel and Distributed Deep Learning" (Hegde & Usmani): **§2, model parallelism**. Các thư mục khác của nhóm làm data
parallelism, allreduce và đo CPU/GPU; thư mục này làm phần chưa ai làm: chia **mô hình** (không chia dữ liệu) ra nhiều GPU và đo xem được gì, mất gì.
Bài báo nói model parallelism chỉ nên dùng khi mô hình không vừa một máy, không phải để tăng tốc. Phần này kiểm chứng điều đó bằng số đo.

Theo yêu cầu của đề, mọi thuật toán song song **tự cài đặt, không dùng thư viện có sẵn**: không cuBLAS, cuDNN, NCCL, MPI, PyTorch. Chỉ dùng CUDA runtime
(cấp phát bộ nhớ, stream, event, copy giữa GPU).

| Yêu cầu của đề | Ở đâu |
|---|---|
| Mã C (tuần tự) | `src/mlp_seq.c` |
| Mã song song (CUDA) | `src/mlp_cuda.cu` |
| README hướng dẫn biên dịch và chạy cả hai | tài liệu này, mục 3 và 4 |
| Mô hình toán, thuật toán, thiết kế song song | mục 1 và 2 |
| Kết quả đo | mục 5, dữ liệu gốc trong `results/` |

## 1. Bài toán

Huấn luyện một chồng `L` khối bằng SGD. Mỗi khối là khối MLP có nhánh residual:

```
Z = X·W1 + b1            W1: H×F,  F = 4H,  X: B×H
U = ReLU(Z)
X' = X + U·W2            W2: F×H
loss = 0.5 · mean(X_L²)   (X_L: đầu ra của khối cuối)
```

Lan truyền ngược (viết tay, 4 phép nhân ma trận mỗi khối): `dW2 = Uᵀ·dY`, `dU = dY·W2ᵀ`, `dZ = dU ⊙ (U>0)`, `db1 = Σ dZ`, `dW1 = Xᵀ·dZ`, `dX = dY + dZ·W1ᵀ`.
Số FLOP mỗi bước: `12·B·H·F·L`. Dữ liệu và trọng số được sinh bằng một hàm băm không trạng thái (splitmix64), nên **CPU và GPU, và mọi cách chia mô hình,
bắt đầu từ đúng cùng một giá trị**; nhờ đó so được kết quả với bản tuần tự. Dữ liệu là tổng hợp (không có tập dữ liệu thật): mục tiêu là đo song song, không phải độ chính xác học.

## 2. Thiết kế song song (`src/mlp_cuda.cu`)

- **GEMM tự viết**: ô 64×64 mỗi block, mỗi luồng tính 4×4 phần tử, tải lát K=16 vào shared memory. Ba biến thể (A·B, Aᵀ·B, A·Bᵀ) từ một template.
- **Pipeline parallelism** (`--mode pipeline`, lịch GPipe): `L` khối chia thành `S` stage liên tiếp, mỗi stage một GPU. Batch chia thành `M` micro-batch;
  GPU `s` xử lý micro-batch `m+1` trong lúc GPU `s+1` xử lý `m`. Activation và gradient chuyển giữa các stage bằng `cudaMemcpyPeerAsync`, đồng bộ bằng
  `cudaEvent` giữa các stream (không đồng bộ host trong lúc chạy). Bong bóng pipeline lý thuyết: `(S-1)/(M+S-1)`.
- **Tensor parallelism** (`--mode tensor`, kiểu Megatron): mỗi khối chia theo chiều `F`: `W1` theo cột, `W2` theo hàng, nên mỗi GPU giữ 1/T trọng số và
  mỗi khối cần **một all-reduce ở forward và một ở backward**. All-reduce tự viết theo recursive doubling (log₂T vòng trao đổi toàn bộ buffer giữa các cặp GPU rồi cộng).
- Bản một GPU là pipeline với `--gpus 1`.

## 3. Biên dịch

Cần: trình biên dịch C (gcc/clang) và make cho bản tuần tự; thêm CUDA toolkit (`nvcc`) và GPU NVIDIA cho bản song song (thử với CUDA 12.4, H100).

```bash
make seq                  # bản tuần tự  -> bin/mlp_seq
make cuda                 # bản song song -> bin/mlp_cuda   (mặc định ARCH=sm_90; A100: ARCH=sm_80; T4: ARCH=sm_75)
make                      # cả hai
```

## 4. Chạy

Bản tuần tự (chạy trên CPU, một luồng):

```bash
bin/mlp_seq --H 512 --L 4 --B 128 --steps 2          # in "RESULT ... step_ms=..."
bin/mlp_seq --H 64 --L 2 --B 32 --check              # in loss và chuẩn gradient của một lượt forward+backward
```

Bản song song (cần ≥ số GPU yêu cầu):

```bash
bin/mlp_cuda --mode pipeline --gpus 4 --micro 8 --H 4096 --L 8 --B 2048    # pipeline, 4 stage, 8 micro-batch
bin/mlp_cuda --mode tensor   --gpus 4          --H 4096 --L 8 --B 2048    # tensor parallelism trên 4 GPU
bin/mlp_cuda --mode pipeline --gpus 1 --micro 1 --H 512 --L 4 --B 128     # một GPU
```

Tham số: `--H` chiều ẩn, `--L` số khối, `--B` batch toàn cục (chia hết cho `--micro`), `--gpus` số GPU, `--micro` số micro-batch (pipeline),
`--steps`/`--warmup` số bước đo/khởi động, `--lr`, `--seed`, `--check`. Điều kiện: pipeline cần `L ≥ gpus`; tensor cần `gpus` là luỹ thừa của 2.

**Kiểm tra tính đúng** (so bản song song với bản tuần tự; cần 4 GPU): `scripts/check_correctness.sh`.
**Kiểm tra bản tuần tự bằng autograd** (công cụ ngoài, cần python3 + numpy + torch; không thuộc mã nộp): `scripts/verify_seq_autograd.sh`, kết quả trong `results/verify_autograd.log`.
**Benchmark đầy đủ** (cần 4 GPU): `scripts/bench.sh > results/bench.log; python3 scripts/results_to_csv.py results/bench.log > results/bench.csv`.
Chạy trên cụm Kubernetes: `KUBECONFIG=... scripts/cluster_run.sh <tên> <số GPU> "make && scripts/bench.sh"` (đóng gói mã vào ConfigMap, dùng image
`pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` chỉ để có `nvcc`; mã không gọi thư viện của PyTorch).

## 5. Kết quả

Đo trên 4× NVIDIA H100 80GB (một node), CUDA 12,4, FP32 (không Tensor Core), `--steps 3 --warmup 1`, **một lần chạy mỗi cấu hình**. Dữ liệu gốc: `results/bench.log`, `results/bench.csv`. Kiểm tra tính đúng (`results/check.log`): **12/12 cấu hình** (pipeline S=1,2,4 với M=1,2,4,8; tensor T=1,2,4) khớp bản tuần tự, sai số tương đối của loss ≤ 8·10⁻⁹ và của chuẩn gradient ≤ 1,4·10⁻⁸; bản tuần tự lại khớp PyTorch autograd (fp64) đến 8 chữ số của loss (`results/verify_autograd.log`). TFLOPS là tổng FLOP của mô hình chia thời gian bước (hiệu dụng, gộp mọi GPU).

### 5.1 Tuần tự (C, CPU một luồng) so với CUDA một GPU (H=512, L=4, B=128)

| | bước (ms) | hiệu năng |
|---|---|---|
| C tuần tự (CPU) | 744,1 | 8,7 GFLOPS |
| CUDA, 1 GPU | 2,01 | 3,2 TFLOPS |

CUDA nhanh hơn bản tuần tự khoảng **371×** (mô hình nhỏ, GPU chưa được dùng hết).

### 5.2 Mô hình vừa một GPU (H=4096, L=8, B=2048, 1,07 tỉ tham số; 1 GPU: 553 ms/bước, 23,8 TFLOPS, 10,7 GB)

Tăng tốc = thời gian 1 GPU / thời gian bước. Trần lý thuyết của pipeline là `S·M/(M+S-1)` (chỉ tính bong bóng, bỏ qua mọi chi phí khác).

| Chế độ | GPU | M | bước (ms) | tăng tốc | hiệu suất (tăng tốc/GPU) | trần pipeline | bộ nhớ/GPU (GB) | dữ liệu chuyển/bước (MB) |
|---|---|---|---|---|---|---|---|---|
| pipeline | 2 | 1 | 554 | 1,00× | 50% | 1,00× | 5,8 | 67 |
| pipeline | 2 | 2 | 424 | 1,30× | 65% | 1,33× | 5,7 | 67 |
| pipeline | 2 | 4 | 363 | 1,52× | 76% | 1,60× | 5,6 | 67 |
| pipeline | 2 | 8 | 342 | 1,62× | 81% | 1,78× | 5,6 | 67 |
| pipeline | 2 | 16 | 402 | 1,38× | 69% | 1,88× | 5,6 | 67 |
| pipeline | 4 | 1 | 554 | 1,00× | 25% | 1,00× | 3,3 | 201 |
| pipeline | 4 | 2 | 354 | 1,56× | 39% | 1,60× | 3,2 | 201 |
| pipeline | 4 | 4 | 255 | 2,17× | 54% | 2,29× | 3,2 | 201 |
| pipeline | 4 | 8 | 209 | 2,64× | 66% | 2,91× | 3,1 | 201 |
| pipeline | 4 | 16 | 225 | 2,46× | 61% | 3,37× | 3,1 | 201 |
| tensor | 2 | — | 285 | 1,94× | 97% | — | 5,9 | 1074 |
| tensor | 4 | — | 148 | 3,73× | 93% | — | 3,4 | 4295 |

Quan sát (từ chính các số trên):

- **Model parallelism thuần (M=1) không tăng tốc**: S=2 và S=4 đều cho 1,00× (các GPU chạy lần lượt, mỗi lúc chỉ một GPU làm việc). Đúng điều bài báo nói.
- **Chia micro-batch làm pipeline có tăng tốc**: tốt nhất S=2 đạt 1,62× (M=8) và S=4 đạt 2,64× (M=8); hiệu suất thấp hơn trần lý thuyết và **giảm khi M quá lớn** (M=16 chậm hơn M=8). Nguyên nhân chưa được tách riêng: giả thuyết là micro-batch nhỏ làm GEMM kém hiệu quả hơn.
- **Tensor parallelism ở cấu hình này gần tuyến tính**: T=2 đạt 1,94×, T=4 đạt 3,73× (hiệu suất 93–97%) dù mỗi bước có 16 all-reduce, và tốt hơn pipeline. Cần nhớ mô hình này có chiều ẩn lớn (H=4096) nên tính toán chiếm phần lớn thời gian; kết luận không tự áp dụng cho mô hình nhỏ hơn.
- **Bộ nhớ mỗi GPU giảm xấp xỉ 1/số GPU** (10,7 GB → 3,1 GB với S=4; 3,4 GB với T=4), đo bằng `cudaMemGetInfo` nên gồm cả bộ nhớ của CUDA context (cộng thủ công trọng số, gradient và activation cho thấy phần còn lại khoảng 0,5 GB).
- Một GPU đạt 23,8 TFLOPS bằng GEMM tự viết, khoảng 45% của cuBLAS (53 TFLOPS, đo ở `reference_pytorch/results/wave1/SUMMARY.md`).

### 5.3 Mô hình KHÔNG vừa một GPU (H=8192, L=24, B=256, M=4, 12,9 tỉ tham số; trọng số + gradient FP32 ≈ 103 GB > 80 GB)

| GPU (S) | kết quả | bước (ms) | bộ nhớ đỉnh/GPU (GB) |
|---|---|---|---|
| 1 | **hết bộ nhớ khi cấp phát** | — | — |
| 2 | chạy được | 793 | 52,6 |
| 4 | chạy được | 549 | 26,6 |

Đây là trường hợp bài báo nói model parallelism có lý do tồn tại: một GPU không chứa nổi, chia ra 2 hoặc 4 GPU thì chạy được.



## 6. Giới hạn

- **Không dùng MPI**: bản song song là một tiến trình điều khiển nhiều GPU trong cùng một máy qua CUDA. Chưa đo model parallelism qua nhiều máy.
- All-reduce của tensor parallelism đồng bộ host giữa các pha (đơn giản, không có đua dữ liệu) nên có chi phí cố định cho mỗi lần; bản tối ưu sẽ chồng lấp với tính toán.
- GEMM tự viết đạt khoảng 45% hiệu năng của cuBLAS (xem mục 5); chỉ FP32, không dùng Tensor Core.
- Dữ liệu tổng hợp và loss là `½·mean(x²)`: đo hiệu năng song song, không phải độ chính xác một bài toán học thật.
- Một lần chạy cho mỗi cấu hình; chênh lệch nhỏ hơn vài phần trăm chưa đủ kết luận.

## 7. Thư mục

```
Makefile              biên dịch (make seq / make cuda)
src/common.h          cấu hình mô hình và hàm băm dùng chung CPU/GPU
src/mlp_seq.c         bản tuần tự (C)
src/mlp_cuda.cu       bản song song (CUDA): GEMM, pipeline, tensor parallel
scripts/              kiểm tra tính đúng, benchmark, chạy trên cụm, đổi log sang CSV, kiểm tra bằng autograd
results/              log và CSV gốc
reference_pytorch/    thực nghiệm đối chứng bằng PyTorch/NCCL (KHÔNG thuộc phần nộp, xem README bên trong)
```
