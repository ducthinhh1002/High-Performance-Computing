# Tổng hợp Đợt 2 — CPU + GPU + MPI trên máy mục tiêu

Đối chiếu kết quả với dự đoán ghi trong [`../PLAN.md`](../PLAN.md) (mục "Đợt 2"), kể cả các dự đoán
sai. Cách các dự đoán được ghi lại được nói rõ ở đầu mục đó.

- **Máy**: Intel Core i5-12600K (6 P-core + 4 E-core, 16 luồng), 15,8 GB RAM vật lý, trong đó **WSL2 chỉ
  cấp 7,6 GB**; NVIDIA RTX 2060 6 GB (sm_75), driver Windows 610.88. Ubuntu 24.04 trên WSL2
  (kernel 5.15.146.1-microsoft-standard), GCC 13.3, CUDA Toolkit 12.0 (gói apt), OpenMPI 4.1.6.
- **Điều kiện đo**: power plan Windows **High performance** (đặt tạm trong lúc chạy rồi trả về
  Balanced), tải nền của Windows khi bắt đầu mỗi thí nghiệm 0–7%. Cả hai được ghi trong
  `manifest.json` của từng lần chạy (`environment.windows_host`).
- **Dữ liệu**: MNIST (LeNet, 61 706 tham số) và CIFAR-10 (CNN width 4, riêng `gpu_mpi`).
- **Ngày chạy**: 23/11/2026, 00:33–01:29 (8 thí nghiệm, 149 lần chạy, **0 lỗi**) và 01:31 (đối chứng
  không ghim, 18 lần chạy). Mỗi thư mục `dot2_*/<thời điểm>/` có `raw/*.json`, `logs/`,
  `analysis/*.csv`, `figures/`, `report.md` tự sinh.
- **Mã nguồn**: build từ working tree ngay trước commit `Ghi lại điều kiện máy chủ Windows…`, nên
  `manifest.json` ghi `git.dirty = true` (kết quả chưa được commit lúc chạy). Thay đổi duy nhất của
  `runner.py` sau đó (giữ `--bind-to none` khi np > 8) chỉ ảnh hưởng thí nghiệm đối chứng E4b.

## Hai lỗi đã sửa trước khi có bộ số này

Lần chạy đầu tiên (cùng ngày, 23:33–00:29, đã bỏ) cho số **sai** ở hai chỗ. Cả hai đã sửa và chạy
lại toàn bộ; chi tiết trong `code/docs/ai_effort_log.csv`.

| Lỗi | Số đo sai (lần đầu) | Số đo đúng (bộ này) | Nguyên nhân |
|---|---|---|---|
| `cpu-omp` chạy 1 luồng sau khi `--verify` | GEMM 1024: 31 GFLOP/s (bằng `cpu-blocked`) | 190 GFLOP/s (5,3×) | `omp_set_num_threads(1)` toàn cục từ context tham chiếu 1 luồng |
| GPU đo lúc xung nhịp còn thấp | cuBLAS n = 512…2048: 0,38 / 0,51 / 0,65 TFLOP/s | 3,07 / 5,29 / 5,83 TFLOP/s | warm-up 2 lần (< 1 ms) quá ngắn, GPU nghỉ lâu trong lúc CPU verify |

Hệ quả: nếu không kiểm tra lại, báo cáo sẽ kết luận "OpenMP không tăng tốc GEMM" và "cuBLAS chậm
gấp 10 lần mức hợp lý", cả hai đều sai.

## Tóm tắt

| Mã | Lặp | Kết luận chính | Dự đoán |
|---|---|---|---|
| E0 | 5 lần chạy | 105 kiểm tra, 0 lỗi; CUDA khớp CPU (sai số ≤ 1e-6 với GEMM, 0 với kernel CNN) | đúng |
| E1 | 3 lần × 5 lần đo | cuBLAS 6,23 TFLOP/s (96,6% đỉnh lý thuyết), OpenMP 190 GFLOP/s (5,3× so với 1 luồng) | đúng một phần |
| E2 | 3 lần × 5 lần đo | Conv batch 256: GPU nhanh hơn 1 luồng **55×**, nhanh hơn 12 luồng **13,2×**; có chép dữ liệu: 11× và 2,7× | đúng một phần (13× thấp hơn dự đoán 20–60×) |
| E3 | 3 lần × 20 lần đo | Message lớn: ring nhanh nhất từ k = 4; recursive doubling sụp ở k ≥ 8 | đúng |
| E4 | 3 lần | Speedup 1,79 / 2,91 / 3,12 / 3,16 / 3,45× ở k = 2 / 4 / 6 / 8 / 10 | **sai** (đoán 5× ở k = 6, 6–7× ở k = 8–10) |
| E4b | 3 lần | Không ghim tiến trình: cùng kết quả (3,46× ở k = 10) | — |
| E5 | 2 seed | Sync = tuần tự (98,9%); PSGD không IID: 94,6% (k = 4) và 88,4% (k = 8) | đúng 3/4 ý |
| E6 | 3 lần | Lai MPI×OpenMP: tốt nhất 4×3 và 6×2 (1,37× so với 1×12) | — |
| E7 | 3 lần | GPU 60× nhanh hơn CPU 1 luồng; nhiều tiến trình dùng chung GPU **chậm hơn** | đúng |

## E0 — tính đúng (`dot2_e0_correctness`)

| Kiểm tra | Kết quả |
|---|---|
| `ctest` (gồm `check_cuda`) | 13/13 đạt |
| `pdl_check --device both` | 35/35 (CPU và CUDA) |
| `mpirun -np 4 pdl_check` | 67/67 (allreduce bitwise, sync = tuần tự) |
| Gradient (4 mạng) | sai số tương đối ≤ 1,03e-4 |
| Kernel CUDA so với CPU | GEMM naive/tiled/cuBLAS: ≤ 2,8e-7 (ngưỡng 1e-3); pipeline CNN: 0 |

**Lần đầu CUDA chạy thật, không có lỗi code.** Có một lỗi môi trường (`libcuda` của apt che thư
viện driver của WSL, xem effort log) khiến `pdl_check` bỏ qua CUDA; sau khi sửa `LD_LIBRARY_PATH`
cả 35 kiểm tra CUDA đều đạt.

## E1 — GEMM (`dot2_e1_gemm`)

GFLOP/s, trung bình 3 lần chạy:

| N | cpu-naive (1 luồng) | cpu-blocked (1 luồng) | cpu-omp (12 luồng) | cuda-naive | cuda-tiled | cuBLAS |
|---|---|---|---|---|---|---|
| 64 | 6,6 | 34,3 | 9,5 | 13,4 | 8,0 | 10,5 |
| 256 | 4,3 | 38,5 | 53,4 | 366 | 476 | 613 |
| 1024 | 1,0 | 36,0 | 190 | 551 | 1 145 | 5 292 |
| 4096 | — | 25,9 | 135 | 558 | 1 198 | 6 231 |

- **GEMM 1 luồng**: 34–40 GFLOP/s khi N ≤ 1024, giảm còn 25,9 ở N = 4096. Dự đoán 25–35: đúng ở N
  lớn, cao hơn một chút ở N nhỏ. Blocked nhanh hơn naive **36×** ở N = 1024 (đợt 1: 25×).
- **OpenMP 12 luồng**: gấp **5,3×** một luồng ở N = 1024 (5,2× ở N = 4096). Dự đoán 5–8×: ở mép
  dưới. 12 luồng chỉ chạy trên 6 nhân vật lý (mỗi nhân hai luồng HT), nên không thể đạt 12×.
  OpenMP **thua** blocked ở N = 64 (9,5 so với 34,3), có thể vì chi phí đánh thức luồng lớn hơn việc tính
  (chưa đo riêng).
- **GPU**: cuBLAS đạt 6,23 TFLOP/s ở N = 4096, bằng 96,6% đỉnh lý thuyết FP32 của RTX 2060
  (≈ 6,45 TFLOP/s). Kernel tiled viết tay đạt 1,2 TFLOP/s (19% đỉnh), naive 0,56 TFLOP/s: thứ tự
  naive < tiled < cuBLAS giống như trên CPU, và khoảng cách cuBLAS/tiled là **5,2×**.
- **GPU thua CPU ở N nhỏ**: ở N = 64 cuBLAS chỉ 10,5 GFLOP/s (khoảng 50 µs cho 0,5 MFLOP), thấp hơn CPU
  1 luồng (34,3). Việc này xảy ra khi thời gian tính quá ngắn so với chi phí cố định mỗi lần gọi (khởi chạy
  kernel, đồng bộ); chưa đo riêng từng thành phần của chi phí đó.

## E2 — §3 của bài, có GPU (`dot2_e2_paper_cpu_gpu`)

Lớp conv đầu của bài (ảnh 224×224 xám, 96 bộ lọc 11×11, stride 4), thời gian forward [ms]:

| batch | 1 | 8 | 32 | 128 | 256 |
|---|---|---|---|---|---|
| CPU 1 luồng (im2col + GEMM) | 1,97 | 17,5 | 69,6 | 282 | 551 |
| CPU OpenMP 12 luồng | 0,78 | 4,75 | 18,5 | 71,7 | 134 |
| GPU cuBLAS (dữ liệu đã ở GPU) | 0,072 | 0,347 | 1,30 | 5,28 | 10,1 |
| GPU cuBLAS + chép host↔device | 0,324 | 1,67 | 6,91 | 25,1 | 49,3 |

- **Tuyến tính theo batch** ở mọi cách cài (R² ≥ 0,9988): CPU 1 luồng 2,16 ms/ảnh (đợt 1 trên
  Mac: 5,78), OpenMP 0,52, GPU 0,040 (có chép dữ liệu: 0,192). GPU cũng tuyến tính, vì lớp này đã
  đủ lớn để chiếm hết GPU ngay từ batch 1.
- **GPU nhanh hơn CPU ở batch 256**: **54,5×** so với 1 luồng, **13,2×** so với 12 luồng OpenMP.
  Bài báo đo ≈ 40× (K520 so với 1 nhân Xeon): kết quả ở đây cùng bậc, cao hơn. Con số này **không
  tính chép dữ liệu**; khi tính cả chép host↔device thì còn 11,2× và 2,7×.
- **Dự đoán**: "20–60× so với CPU 12 luồng OpenMP" (PLAN.md) **sai**: đo được 13,2×. Dự đoán
  "50–150× so với 1 luồng" đúng ở mép dưới (54,5×). Nguyên nhân dự đoán sai: OpenMP chỉ tăng tốc
  conv 4,1× ở batch 256 (không phải 5–8× như GEMM), vì chạy song song theo từng ảnh (xem dưới).
- **OpenMP theo batch không đều**: batch 1: 0,78 ms (2,5× so với 1 luồng, song song trong GEMM);
  batch 2: 3,43 ms (**1,1×**); batch 8: 4,75 ms (3,7×); batch 256: 4,1×. Với batch > 1 code song song
  theo ảnh (bài §2, "embarrassingly parallel"), nên chỉ min(batch, 12) luồng làm việc và batch 2–4
  dùng 2–4 luồng. Đây là hạn chế của cài đặt, chưa sửa.
- **im2col + GEMM so với conv trực tiếp**: đợt 1 nhanh hơn 12–15×; ở đây chỉ **1,4–2,7×** trên CPU
  và 2,5× trên GPU. Dự đoán đợt 1 "≥ 5×" **không còn đúng trên máy này**. Chưa xác định nguyên nhân
  (có thể do cache lớn hơn và GCC tự vector hoá tốt hơn Apple Clang); cần đo thêm mới kết luận được.
- **FC (nhân ma trận–vector, n×n)**: khi trọng số đã nằm trên GPU, cuBLAS ở n = 4096 mất 0,24 ms,
  nhanh hơn CPU 1 luồng (3,30 ms) 13,8× và OpenMP (1,66 ms) 6,9×. **Khi tính cả chép dữ liệu lên GPU
  thì chậm hơn CPU 2,3× (7,68 ms)** và thua CPU ở mọi n ≤ 4096: ma trận–vector bị giới hạn bởi băng
  thông, nên GPU chỉ có lợi khi trọng số ở lại trên GPU.
  Điểm bất thường: `cpu-omp` ở n = 512 đo 0,43 ms (chậm 27× so với 1 luồng, 0,016 ms) trong khi n = 1024 là
  0,015 ms; có thể là nhiễu hoặc chi phí đánh thức luồng, chưa điều tra.

## E3 — giao tiếp (`dot2_e3_comm`)

Thời gian allreduce, trung bình 3 lần chạy × 20 lần đo:

| thuật toán | 4 B, k=4 | 4 KiB, k=4 | 16 MiB, k=2 | 16 MiB, k=4 | 16 MiB, k=8 | 16 MiB, k=10 |
|---|---|---|---|---|---|---|
| mpi (OpenMPI) | **1,0 µs** | 4,3 µs | 4,5 ms | **12,8 ms** | 31,0 ms | 46,9 ms |
| ring | 2,1 µs | **4,0 µs** | **4,5 ms** | **12,8 ms** | **27,8 ms** | **33,2 ms** |
| rd | 1,1 µs | 4,2 µs | 6,0 ms | 23,1 ms | 71,8 ms | 85,2 ms |
| tree | 1,5 µs | 5,7 µs | 5,3 ms | 16,6 ms | 36,3 ms | 48,0 ms |
| naive | 3,4 µs | 7,5 µs | 5,3 ms | 22,5 ms | 44,7 ms | 59,2 ms |
| reduce_bcast | 1,5 µs | 7,4 µs | 5,3 ms | 22,5 ms | 34,5 ms | 42,7 ms |

- **Độ trễ và băng thông** (đoán: ≈ 1 µs và 8–15 GB/s): đúng. `MPI_Allreduce` 4 B mất 0,3 µs ở k = 2
  và 1,0 µs ở k = 4. Ping-pong đạt đỉnh 21,9 GB/s ở 4 MiB và 8,4 GB/s ở 64 MiB (vượt cache); khớp α-β cho
  α = 0,43 µs, băng thông 14,3 GB/s.
- **Message lớn**: ring nhanh nhất hoặc đồng hạng ở mọi k, đúng như đợt 1 và đúng dự đoán "ring và MPI tốt
  nhất". Khác với đợt 1, ring hơn `MPI_Allreduce` rõ ràng khi k ≥ 6 (33,2 so với 46,9 ms ở k = 10): bản
  tự viết thắng thư viện OpenMPI mặc định ở message lớn.
- **Message nhỏ**: `mpi` nhanh nhất ở 4 B (1,0 µs), ring **chậm nhất** (2,1 µs ở k = 4 và 10,2 µs
  ở k = 10, vì 2(k−1) vòng). Ở 4 KiB, ring bằng `mpi`.
- **Recursive doubling sụp ở k ≥ 8** với message lớn (71,8 ms ở k = 8 và 85,2 ms ở k = 10, gấp 2,6×
  ring). k = 8 là lũy thừa của 2 nên không phải do xử lý số tiến trình lẻ; nguyên nhân chưa rõ (có thể là
  mẫu truyền cả vector ở mỗi vòng cộng với băng thông shared memory bị chia), chưa có phép đo tách riêng.
- **Mô hình α-β sai nhiều hơn đợt 1**: trung vị 43% (ring 25%, mpi 38%, naive 104%), so với 14% / 22% /
  50–165% ở đợt 1. Mô hình bỏ qua tranh chấp băng thông bộ nhớ giữa các rank, và chúng nặng hơn khi k lên 10.

## E4 — strong scaling sync SGD (`dot2_e4_strong_scaling`, `dot2_e4b_…`)

3 epoch LeNet/MNIST, batch tổng 256, 1 luồng mỗi tiến trình, ghim vào nhân (`--bind-to core`):

| k | thời gian 1 epoch [s] | speedup | hiệu suất | tính toán [s] | tính toán lý tưởng [s] | chờ đồng bộ [s] | allreduce [s] | test acc |
|---|---|---|---|---|---|---|---|---|
| 1 | 7,19 | 1,00 | 100% | 21,58 | 21,58 | 0 | 0 | 98,5% |
| 2 | 4,01 | 1,79 | 90% | 11,85 | 10,79 | 0,13 | 0,04 | 98,5% |
| 4 | 2,47 | 2,91 | 73% | 7,17 | 5,40 | 0,16 | 0,07 | 98,5% |
| 6 | 2,30 | 3,12 | 52% | 6,02 | 3,60 | 0,77 | 0,13 | 98,5% |
| 8 | 2,28 | 3,16 | 40% | 5,53 | 2,70 | 1,17 | 0,13 | 98,6% |
| 10 | 2,09 | 3,45 | 34% | 5,35 | 2,16 | 0,71 | 0,20 | 98,5% |

(Các cột "tính toán" và "chờ" là tổng 3 epoch, trung bình theo rank.)

- **Dự đoán sai.** Đoán hiệu suất 85–90% ở k = 2–6 và speedup 5× ở k = 6, 6–7× ở k = 8–10. Đo được:
  90% ở k = 2 nhưng chỉ 73% ở k = 4 và 52% ở k = 6; speedup bão hoà ở khoảng **3,1–3,5×** từ k = 6 trở đi.
  Dự đoán trong PLAN ("k > 6 làm `sync_wait` tăng rõ") đúng về hướng nhưng sai về chỗ: `sync_wait`
  nhảy từ 0,16 s (k = 4) lên 0,77 s ngay ở **k = 6**, rồi 1,17 s (k = 8), và **giảm** về 0,71 s ở k = 10.
- **Giới hạn nằm ở phần tính toán, không phải giao tiếp**, giống đợt 1. Thời gian forward + backward
  lệch khỏi lý tưởng T₁/k theo +10%, +33%, +67%, +105%, +148% ở k = 2, 4, 6, 8, 10. Allreduce chỉ
  0,04–0,20 s trong tổng 6–12 s.
- **Giả thuyết về E-core không kiểm chứng được trên WSL2.** Đo thử tốc độ GEMM 1 luồng trên 10 CPU
  (0, 1, 2, 4, 10–15) cho kết quả như nhau (29,4–31,3 GFLOP/s), tức VM không phân biệt được P-core và E-core: Windows xếp
  vCPU lên nhân vật lý. Chạy **không ghim** (E4b) cho cùng kết quả (3,46× ở k = 10, so với 3,45×), nên ghim
  vào "nhân" trong VM không có tác dụng. Vì vậy sự suy giảm hiệu suất ở k ≥ 6 không thể quy cho E-core;
  các nguyên nhân khả dĩ còn lại là xung nhịp toàn nhân thấp hơn, tranh chấp bộ nhớ/L3, và lập lịch của
  Windows host, nhưng chưa có phép đo nào tách được từng nguyên nhân.
- Tuần tự 1 tiến trình: **7,19 s/epoch**, nhanh hơn Mac đợt 1 (19,2 s) 2,7×. Dự đoán 8–11 s: nhanh hơn
  dự đoán (chặn dưới quá thận trọng).

## E5 — so sánh chiến lược (`dot2_e5_strategies`)

10 epoch, local batch 32, momentum 0,9, trung bình 2 seed. Downpour: 1 server + (k−1) worker.

| chiến lược | k | IID acc | không IID acc | thời gian IID [s] | speedup | byte gửi / rank |
|---|---|---|---|---|---|---|
| tuần tự (batch 128) | 1 | 98,9% | — | 69,2 | 1,00× | 0 |
| sync SGD | 4 / 8 | 98,9% / 98,8% | — | 22,0 / 22,4 | 3,14× / 3,10× | 1,73 GB / 1,01 GB |
| PSGD, TB mỗi epoch | 4 / 8 | 99,0% / 99,0% | **94,6 ± 1,6%** / **88,4 ± 4,0%** | 19,4 / 17,9 | 3,58× / 3,86× | 3,7 MB / 4,3 MB |
| PSGD + ShuffleSGD | 4 / 8 | 99,1% / 99,0% | 99,0% / 99,0% | 19,4 / 17,7 | 3,58× / 3,91× | 83 MB / 51 MB |
| PSGD, TB mỗi 10 bước | 4 / 8 | 98,9% / 98,9% | 98,6% / 98,3% | 20,5 / 19,8 | 3,37× / 3,49× | 174 MB / 104 MB |
| ADMM (ρ = 0,05, 20 bước/vòng) | 4 / 8 | 99,1% / 99,0% | 96,9% / **87,4%** | 20,8 / 20,2 | 3,33× / 3,42× | 89 MB / 52 MB |
| Downpour, push/fetch mỗi 1 bước | 4 / 8 | 98,9% / 98,8% | 97,3% / 98,4% | 27,9 / 27,3 | 2,48× / 2,54× | 1,71 GB / 0,78 GB |
| Downpour, push/fetch mỗi 4 bước | 4 / 8 | 98,8% / 98,9% | 98,1% / 96,5% | 27,0 / 22,9 | 2,56× / 3,02× | 0,58 GB / 0,28 GB |

- (a) **Đúng.** Sync SGD ≈ tuần tự (98,9% cả hai), nhanh hơn 3,1×. Tăng từ 4 lên 8 tiến trình **không
  nhanh thêm** (22,0 → 22,4 s), cùng hiện tượng bão hoà như E4.
- (b) **Đúng, và mạnh hơn khi k tăng.** Trên dữ liệu không IID, PSGD chỉ lấy trung bình mỗi epoch rơi
  xuống 94,6% (k = 4) và **88,4% (k = 8)**, trong khi IID đạt 99,0%. ShuffleSGD đưa về 99,0%. Lấy trung
  bình mỗi 10 bước cũng đưa về 98,6% / 98,3% mà không chuyển dữ liệu. Ở đợt 1 (3 epoch) PSGD không IID rơi
  xuống 77,7%; con số lần này cao hơn vì huấn luyện 10 epoch với siêu tham số khác, nên **không so
  sánh trực tiếp** hai đợt.
- (c) **Sai** (đã sai từ đợt 1, vẫn sai): ADMM không hội tụ chậm hơn PSGD (99,1% so với 99,0%) và
  không giao tiếp ít nhất (89 MB so với 3,7 MB của PSGD-mỗi-epoch). Trên dữ liệu không IID ADMM
  **xấu đi khi k tăng**: 96,9% (k = 4) xuống 87,4% (k = 8), giống PSGD-mỗi-epoch.
- (d) Downpour **không** cho độ chính xác thấp nhất trên IID (98,8–98,9%, ngang các phương pháp khác
  sau 10 epoch), khác với dự đoán đợt 1. Nó vẫn **chậm nhất** (2,5–3,0×): tỉ lệ giao tiếp 15% ở k = 4 và
  34% ở k = 8 khi push mỗi bước. Push mỗi 4 bước giảm byte gửi 2,8× và rút ngắn thời gian 16% ở k = 8, nhưng
  độ chính xác không IID giảm từ 98,4% xuống 96,5%.
- Độ lệch chuẩn của PSGD-mỗi-epoch trên dữ liệu không IID lớn (± 1,6 và ± 4,0 điểm với 2 seed): chênh
  lệch dưới vài điểm giữa các phương pháp trong nhóm đó chưa có ý nghĩa thống kê.

## E6 — lai MPI × OpenMP (`dot2_e6_hybrid`)

12 luồng phần cứng chia thành np tiến trình × (12/np) luồng OpenMP, sync SGD, 2 epoch, batch tổng 256:

| np × luồng | 1×12 | 2×6 | 3×4 | 4×3 | 6×2 | 12×1 |
|---|---|---|---|---|---|---|
| giây / epoch | 2,66 | 2,19 | 2,03 | 1,96 | 1,95 | 2,02 |
| tỉ lệ giao tiếp + chờ | 0% | 3,0% | 4,4% | 5,2% | 7,6% | 10,7% |

- Cấu hình tốt nhất là **4×3 và 6×2** (1,37× so với 1×12). Chỉ dùng OpenMP (1×12) là **tệ nhất** vì
  OpenMP trong LeNet chỉ tăng tốc 2,7× so với 1 luồng (7,19 → 2,66 s/epoch): các lớp nhỏ không đủ việc
  cho 12 luồng. Chỉ dùng MPI (12×1) lại tốn nhiều giao tiếp và chờ hơn (10,7%).
- Đây là phép đo thứ hai của giới hạn "tính toán dùng chung": cả ba cách (E4 k = 10, E6 12×1, 4×3)
  đều dừng quanh 2,0 s/epoch (3,4–3,7× so với 1 luồng).

## E7 — GPU + MPI (`dot2_e7_gpu_mpi`)

CIFAR-10 (CNN width 4, 20 000 ảnh), 1 epoch, batch tổng 256, sync SGD:

| thiết bị | np | thời gian [s] | speedup so với cùng loại, np = 1 | tỉ lệ giao tiếp | ghi chú |
|---|---|---|---|---|---|
| CPU 1 luồng/rank | 1 | 101,1 | 1,00× | 0% | |
| CPU | 4 | 31,7 | 3,19× | 4,7% | |
| CPU | 8 | 27,4 | 3,69× | 21,7% | |
| GPU (cuBLAS) | 1 | **1,69** | 1,00× | 0% | **59,8× so với CPU np = 1** |
| GPU | 2 | 3,15 | **0,54×** | 26,0% | chép gradient qua host 0,41 s |
| GPU | 4 | 4,29 | **0,40×** | 42,7% | chép gradient qua host 0,77 s |

- **GPU cho CNN lớn hơn nhanh hơn rất nhiều**: 60× so với 1 luồng CPU và 16× so với 8 tiến trình CPU.
  Dự đoán "GPU chỉ nhanh hơn 2–5× với LeNet/MNIST" **không kiểm chứng được**: thí nghiệm này dùng CNN và dữ
  liệu lớn hơn, và chưa chạy LeNet trên GPU.
- **Nhiều tiến trình dùng chung một GPU làm chậm đi**, đúng dự đoán ("gần như không nhanh hơn"), thậm chí
  tệ hơn: 2 tiến trình chậm gần gấp đôi, 4 tiến trình gấp 2,5×. Mỗi rank có context CUDA riêng, các kernel
  chia thời gian GPU, và gradient phải đi qua bộ nhớ host (tăng với np) trước khi allreduce.
- Độ chính xác không đổi theo thiết bị (47,5–48,0% sau 1 epoch), nên GPU và CPU tính cùng kết quả.

## Đã chứng minh và chưa chứng minh

- **Đã chứng minh** (trên máy này): CUDA backend tính đúng như CPU; cuBLAS đạt 96,6% đỉnh lý thuyết FP32;
  GPU nhanh hơn CPU 1 luồng 55× (conv, batch 256) và 60× (huấn luyện CNN CIFAR-10), cùng bậc với ≈ 40×
  của bài; GPU thua CPU với phép tính nhỏ và với ma trận–vector khi phải chép dữ liệu; ring thắng
  `MPI_Allreduce` mặc định của OpenMPI ở message lớn; strong scaling bão hoà ở khoảng 3,1–3,5× với LeNet
  và nguyên nhân nằm ở phần tính toán dùng chung chứ không phải giao tiếp; dùng chung một GPU cho nhiều
  tiến trình là phản tác dụng.
- **Chưa chứng minh**: hiệu ứng E-core/P-core (không phân biệt được trong WSL2); mạng giữa nhiều máy;
  LeNet trên GPU; nguyên nhân của việc conv trực tiếp chỉ chậm hơn im2col 1,4–2,7× (đợt 1: 12–15×);
  nguyên nhân recursive doubling sụp ở k ≥ 8.
- **Giới hạn của phép đo**: WSL2 (vCPU do Windows xếp lịch, chỉ 7,6 GB RAM); `manifest.json` ghi
  `git.dirty = true`; số lần lặp 2–3. Mô hình nhỏ làm giao tiếp rẻ.
