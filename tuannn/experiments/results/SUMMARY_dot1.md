# Tổng hợp Đợt 1 — CPU + MPI trên máy phát triển

Đối chiếu kết quả với dự đoán ghi **trước** khi chạy trong [`../PLAN.md`](../PLAN.md), kể cả các
dự đoán sai.

- **Máy**: Intel Core i5-8257U (4 nhân, 8 luồng), 16 GB RAM, macOS 13.7, Apple Clang 14 (không có
  OpenMP), MPICH 5.0.2. Không có GPU.
- **Dữ liệu**: MNIST 60k ảnh train và 10k ảnh test, LeNet 61 706 tham số (247 KB gradient).
  Riêng E2 dùng hình học ảnh của bài báo.
- **Ngày chạy**: 22/11/2026. Mỗi thư mục `dot1_*/<thời điểm>/` có `raw/*.json` (số đo gốc),
  `logs/`, `analysis/*.csv`, `figures/` và `report.md` tự sinh.

| Mã | Lặp | Kết luận chính | Dự đoán |
|---|---|---|---|
| E0 | 1 | Bản song song cho đúng kết quả của bản tuần tự | đúng |
| E1 | 5 lần đo / điểm | GEMM blocked 14,5–17,2 GFLOP/s, nhanh hơn naive 7–25× | sai về độ lớn (đoán 3–10×) |
| E2 | 5 lần đo / điểm | Conv tuyến tính theo batch: 5,78 ms/ảnh, R² = 0,9987 | đúng |
| E3 | 20 lần đo / điểm | Message nhỏ: hàm thư viện MPI nhanh nhất; message lớn: ring/mpi nhanh nhất, naive tăng tuyến tính theo k | đúng một phần |
| E4 | 3 lần | Speedup 1,93× / 3,29× / 3,48× ở k = 2 / 4 / 8 | đúng |
| E5 | 2 seed | Sync = tuần tự (98,6%). PSGD trên dữ liệu không IID rơi xuống 77,7%; ShuffleSGD kéo lại 97,9% | đúng 3/4 ý |

## E0 — tính đúng (`dot1_e0_correctness`, `dot1_e0_equivalence`)

| Kiểm tra | Kết quả |
|---|---|
| Gradient (4 mạng, 277 tham số được thử) | sai số tương đối lớn nhất 1,1e-4 (ngưỡng 2e-2) |
| Sync SGD k = 2 và k = 4 so với tuần tự (cùng batch tổng) | sai lệch tham số 5,7e-8 và 2,9e-8 (ngưỡng 1e-4) |
| Replica sau 6 thuật toán allreduce | trùng từng bit ở cả k = 2 và k = 4 |
| `seq_train` (C thuần) so với `pdl_train --strategy sequential` | loss và accuracy **giống hệt** ở mọi epoch (0,44458018 rồi 0,11733646; test 94,50% rồi 96,48%) |

Dự đoán đúng. Sai lệch sync ↔ tuần tự còn nhỏ hơn dự đoán khoảng 1000 lần: chỉ khác thứ tự cộng số
thực trong một bước. Hai chương trình cùng làm 2 epoch trên 12k ảnh mất 8,08 s (`seq_train`) và
8,45 s (`pdl_train`). Lớp MPI và khung đo pha chỉ làm chậm khoảng 4%.

## E1 — GEMM (`dot1_e1_gemm`)

| N | naive GFLOP/s | blocked GFLOP/s | blocked / naive |
|---|---|---|---|
| 64 | 2,10 | 14,48 | 6,9× |
| 256 | 1,30 | 16,32 | 12,5× |
| 512 | 0,77 | 16,03 | 20,9× |
| 1024 | 0,65 | 16,57 | 25,5× |

- Blocked giữ ổn định 14,5–17,2 GFLOP/s, cao hơn khoảng đoán 5–15 một chút.
- **Dự đoán sai về độ lớn.** Khoảng cách tăng theo N, lên tới 25×, vì naive đọc B theo cột: khi
  N ≥ 512, mỗi cột B (N × 4 byte, bước nhảy N phần tử) trượt cache ở mọi phần tử. Đây là lý do
  phải tối ưu cache trước khi nói tới song song.

## E2 — §3 của bài trên CPU (`dot1_e2_paper_cpu`)

Lớp conv đầu của bài: ảnh 224×224 xám, 96 bộ lọc 11×11, stride 4 → 54×54.

| batch | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| im2col + GEMM [ms] | 5,1 | 10,8 | 21,3 | 41,9 | 89,0 | 172,0 | 371,7 |
| conv trực tiếp [ms] | 79,3 | 138,2 | 273,4 | 545,8 | 1101,6 | 2581,9 | 4702,3 |

- (a) **Đúng.** Thời gian tuyến tính theo batch: độ dốc 5,78 ms/ảnh, R² = 0,9987. Như bài nói, CPU
  xử lý từng ảnh nên O(n). *Đính chính PLAN.md*: bài chỉ ghi "độ dốc của đường CPU là 6,45" (Xeon
  E5-2670), phần văn bản không nêu đơn vị. Nếu đơn vị là ms/ảnh thì kết quả cùng bậc, nhưng không
  so sánh trực tiếp hai con số được.
- (b) **Đúng.** Conv trực tiếp chậm hơn im2col + GEMM 12,4–15,5×, vì GEMM dùng lại dữ liệu trong
  cache còn conv trực tiếp thì không. Đây đúng là lý do Caffe dùng im2col (bài §3.1).
- (c) **Đúng.** FC n×n với batch 1: mỗi lần n tăng 2× thì thời gian tăng 4,25×; 4,3×; 5,9×; 4,65×.
  Tức là O(n²). Bước nhảy 5,9× ở n = 2048 là lúc ma trận trọng số (16 MB) vượt L3 (6 MB), phép
  tính chuyển sang bị giới hạn bởi băng thông bộ nhớ.
- **Chưa kiểm chứng**: tỉ lệ GPU/CPU khoảng 40× của bài, vì cần GPU (đợt 2).

## E3 — giao tiếp (`dot1_e3_comm`)

Thời gian allreduce [µs], max theo rank, trung bình 20 lần:

| thuật toán | 4 B, k=4 | 64 KiB, k=4 | 16 MiB, k=2 | 16 MiB, k=4 | 16 MiB, k=8 |
|---|---|---|---|---|---|
| mpi (MPI_Allreduce) | **1,0** | **49,8** | 9 198 | **15 590** | 33 647 |
| ring | 16,2 | 88,0 | 9 597 | 16 540 | **30 598** |
| rd (recursive doubling) | 5,8 | 53,2 | **8 119** | 21 127 | 102 606 |
| tree (binomial) | 9,4 | 76,9 | 11 937 | 23 370 | 48 247 |
| naive (driver trung tâm) | 7,3 | 107,2 | 14 379 | 39 643 | 111 409 |
| reduce_bcast | 1,7 | 67,7 | 9 060 | 16 302 | 188 014 |

- (a) **Đúng.** Khớp α-β từ ping-pong cho α = 2,66 µs và băng thông 1/β = 3,4 GB/s
  (shared memory).
- (b) **Đúng một phần.** Message nhỏ: rd (5,8 µs) và tree (9,4 µs) nhanh hơn ring (16,2 µs), đúng
  như số vòng log k so với 2(k−1). **Nhưng** hàm thư viện (`mpi`: 1,0 µs, `reduce_bcast`: 1,7 µs)
  nhanh hơn mọi bản tự viết 5–16×. MPICH dùng đường shared memory chuyên biệt, còn bản tự viết trả
  chi phí đầy đủ của từng `MPI_Sendrecv`.
- (c) **Đúng.** Ở 16 MiB, ring và mpi nhanh nhất; naive chậm nhất và tăng gần tuyến tính theo k
  (14,4 → 39,6 → 111,4 ms, mỗi lần ≈ 2,8×), đúng với chi phí 2(k−1)(α + nβ) của driver trung tâm.
  Trên một node, thứ tự là ring ≈ mpi < rd < tree. Điều này **trái với** phân tích của bài, vốn ưu
  tiên "BitTorrent aggregate" (cây, O(p log k)): cây gửi cả vector ở mỗi tầng, còn ring chỉ gửi
  2(k−1)/k · n.
- (d) **Đúng, nhưng mạnh hơn dự đoán.** k = 8 (8 tiến trình trên 4 nhân thật) làm hỏng nặng một số
  thuật toán: reduce_bcast 16 MiB mất 188 ms (k = 4: 16 ms), rd 103 ms. Chỉ ring giữ được hình dạng
  (1,85× so với k = 4). Hai tiến trình chung một nhân vật lý tranh băng thông bộ nhớ, và MPICH chờ
  bằng cách quay vòng (busy-poll) nên chiếm luôn nhân đang tính.
- (e) **Đúng một phần.** Với message ≥ 1 MiB, mô hình α-β sai trung vị 14% (ring) và 22% (mpi),
  nhưng 50–165% với naive, rd, tree và reduce_bcast. Mô hình bỏ qua tranh chấp bộ nhớ khi nhiều rank
  cộng cùng lúc, và bỏ qua hiện tượng xếp hàng ở rank gốc.

## E4 — strong scaling sync SGD (`dot1_e4_strong_scaling`)

| k | thời gian 1 epoch [s] (TB ± std, 3 lần) | speedup | hiệu suất | chờ đồng bộ [s] | allreduce [s] | tỉ lệ comm | test acc |
|---|---|---|---|---|---|---|---|
| 1 | 19,20 ± 0,10 | 1,00 | 1,00 | 0 | 0 | 0% | 97,97% |
| 2 | 9,93 ± 0,03 | **1,93** | 0,97 | 0,03 | 0,03 | 0,6% | 97,86% |
| 4 | 5,84 ± 0,49 | **3,29** | 0,82 | 0,21 | 0,05 | 4,4% | 97,89% |
| 8 | 5,52 ± 0,67 | **3,48** | 0,44 | 0,31 | 0,54 | 14,8% | 98,04% |

- **Đúng cả ba khoảng dự đoán**: 1,93 (đoán 1,8–1,95), 3,29 (đoán 2,8–3,4), 3,48 (≤ S(4) + 0,5).
  Độ chính xác không phụ thuộc k, như mong đợi vì đây là cùng một thuật toán.
- Tỉ lệ giao tiếp ở k = 4 là 4,4% (< 10%, đúng). Nhưng ở k = 4 thời gian **chờ đồng bộ (0,21 s)
  gấp 4 lần thời gian allreduce (0,05 s)**, dù chênh tải đo theo thời gian tính toán chỉ 1,003.
  Thời gian chờ đến từ dao động từng bước (scheduling của hệ điều hành, turbo), không phải từ chia
  việc không đều.
- Mô hình α-β dự đoán speedup 3,95 (k = 4) và 7,76 (k = 8), cao hơn đo 17% và 55%. Phần thiếu
  **không nằm ở giao tiếp** mà ở chính phần tính toán. Forward + backward ở k = 4 mất 5,52 s so
  với 19,14 / 4 = 4,79 s lý tưởng (+15%): 4 nhân cùng chạy thì xung turbo giảm và băng thông bộ nhớ
  bị chia. Ở k = 8, hai luồng chung một nhân nên phần tính toán gần như không nhanh thêm.
  Kết luận: với mô hình nhỏ, giới hạn scale trên một máy là **tài nguyên tính toán dùng chung**,
  không phải mạng.

## E5 — so sánh chiến lược (`dot1_e5_strategies`)

3 epoch, 4 tiến trình (Downpour: 1 server + 3 worker), trung bình 2 seed (± độ lệch chuẩn).

| chiến lược | dữ liệu | test acc | thời gian [s] | tỉ lệ comm | byte gửi / rank |
|---|---|---|---|---|---|
| tuần tự (1 tiến trình, batch 256) | toàn bộ | 98,64 ± 0,18% | 57,8 | 0% | 0 |
| sync SGD (batch tổng 256) | toàn bộ | 98,58 ± 0,11% | 17,1 | 3,8% | 260 MB |
| PSGD, trung bình mỗi epoch | IID | 98,12 ± 0,11% | 14,7 | 0,2% | 1,1 MB |
| PSGD, trung bình mỗi epoch | không IID | **77,68 ± 3,95%** | 15,6 | 0,3% | 1,1 MB |
| PSGD + ShuffleSGD | không IID | **97,87 ± 0,07%** | 17,1 | 0,9% | 18,8 MB |
| PSGD, trung bình mỗi 10 bước | IID / không IID | 98,15% / 96,98% | 15,4 / 15,4 | 1,1% / 1,2% | 26,7 MB |
| ADMM (ρ = 0,05, 20 bước/vòng) | IID / không IID | 98,06% / 88,48 ± 4,31% | 16,0 / 16,8 | 0,7% / 0,9% | 13,3 MB |
| Downpour (n_push = n_fetch = 2) | IID / không IID | 96,32% / 93,94% | 25,3 / 25,7 | 18,1% / 19,3% | 115,5 MB |

- (a) **Đúng.** Sync SGD ≈ tuần tự (98,58% so với 98,64%, chênh 0,06 điểm), nhanh hơn 3,38×.
- (b) **Đúng.** Với shard sắp theo nhãn, PSGD chỉ lấy trung bình mỗi epoch rơi từ 98,1% xuống
  77,7% (−20 điểm). Mỗi worker học một nhóm nhãn, và trung bình các mô hình đã đi quá xa nhau
  không còn tốt. ShuffleSGD (all-to-all 1/k dữ liệu sau mỗi epoch) kéo lại 97,9%, đúng gợi ý của
  bài §5.1.1, với giá 18,8 MB dữ liệu chuyển đi mỗi rank.
  **Phát hiện thêm**: lấy trung bình thường xuyên hơn (mỗi 10 bước) cũng gỡ được vấn đề (97,0%)
  **mà không phải di chuyển dữ liệu**.
- (c) **Sai.** ADMM không hội tụ chậm hơn PSGD trên dữ liệu IID (98,06% so với 98,12%) và cũng
  **không** giao tiếp ít nhất: 13,3 MB, gấp 12× PSGD-mỗi-epoch, vì mỗi 20 bước có một vòng đồng
  thuận. Trên dữ liệu không IID, biến đối ngẫu u giúp ADMM tốt hơn PSGD-mỗi-epoch (88,5% so với
  77,7%), nhưng vẫn kém PSGD-mỗi-10-bước (97,0%), là cách có số lần giao tiếp tương đương.
- (d) **Đúng trên IID.** Downpour thấp nhất (96,3%) và chậm nhất trong các chiến lược song song
  (25,3 s, 18% thời gian là giao tiếp): một server phục vụ 3 worker là nút thắt, và chỉ 3 trên 4
  tiến trình tính gradient. Staleness trung bình 10,95, lớn nhất 14 (đoán 5–15). Trên dữ liệu
  không IID, Downpour (93,9%) lại hơn PSGD-mỗi-epoch, vì worker lấy tham số mới mỗi 2 bước.

## Đã chứng minh và chưa chứng minh

- **Đã chứng minh** (trên máy này): bản song song đúng bằng bản tuần tự; conv tuyến tính theo batch
  như bài; ring/thư viện MPI hơn cây trên một node; strong scaling 3,3× trên 4 nhân và bị giới hạn
  bởi phần tính toán dùng chung chứ không phải giao tiếp; PSGD cần shuffle hoặc trung bình thường
  xuyên khi dữ liệu không IID.
- **Chưa chứng minh**: mọi điều về GPU (bài §3: GPU nhanh hơn khoảng 40×); OpenMP (Apple Clang
  không hỗ trợ nên các pragma chưa được biên dịch); mạng giữa các máy (mọi phép đo đều trong một
  node); hiệu ứng E-core/P-core của i5-12600K. Tất cả nằm trong đợt 2 (`code/configs/`).
- Số lần lặp còn ít: E4 có 3 lần, E5 có 2 seed, E1–E3 mỗi cấu hình 1 lần chạy (5–20 lần đo bên
  trong). Độ lệch chuẩn lớn ở E5 với dữ liệu không IID (± 4 điểm) cho thấy cần thêm seed trước khi
  so sánh các chênh lệch nhỏ hơn vài điểm.
