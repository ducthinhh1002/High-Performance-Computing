# Báo cáo: Huấn luyện CNN song song và phân tán — cài đặt từ đầu bằng C / MPI / OpenMP / CUDA

**Tác giả**: Tuan Nguyen (thư mục `tuannn/`, Nhóm 10)
**Bài báo tham chiếu**: V. Hegde, S. Usmani, *Parallel and Distributed Deep Learning*, Stanford
University, 2016 — [hgpu.org/?p=16046](https://hgpu.org/?p=16046)
**Ngày**: 22–23/11/2026 · **Phiên bản mã nguồn**: pdl-bench 1.0.0 (`tuannn/code/`)

---

## Tóm tắt

Báo cáo trình bày một hệ thống huấn luyện mạng nơ-ron tích chập (CNN) được cài đặt hoàn toàn bằng C,
không dùng framework, gồm hai phiên bản. **Phiên bản tuần tự** là một chương trình C một luồng.
**Phiên bản song song** dùng MPI cho song song dữ liệu, OpenMP cho đa luồng trên CPU và CUDA cho GPU.
Phiên bản song song có bốn phương pháp tối ưu phân tán được bài báo phân tích: Synchronous SGD,
Parallel SGD (kèm ShuffleSGD), ADMM và Downpour SGD, cùng sáu thuật toán allreduce tự viết. Đợt thực
nghiệm thứ nhất chạy trên CPU 4 nhân (i5-8257U, không GPU) và cho thấy:

1. Hai phiên bản cho kết quả **giống hệt** khi cùng cấu hình. Sync SGD trên k tiến trình lệch với
   tuần tự dưới 6·10⁻⁸.
2. Thời gian của lớp conv trong bài **tuyến tính theo batch** (5,78 ms/ảnh, R² = 0,9987), đúng dạng
   O(n) mà bài mô tả cho CPU. im2col + GEMM nhanh hơn conv trực tiếp 12–15×.
3. Sync SGD đạt speedup **3,29× trên 4 nhân** với độ chính xác không đổi (≈ 98%). Giới hạn scale nằm
   ở phần tính toán dùng chung tài nguyên, không phải giao tiếp (allreduce chỉ chiếm 1% thời gian).
4. Trên một node, ring allreduce và `MPI_Allreduce` nhanh hơn allreduce dạng cây mà bài đề xuất.
5. Khi dữ liệu không IID, Parallel SGD chỉ lấy trung bình mỗi epoch rơi từ 98,1% xuống **77,7%**.
   ShuffleSGD đưa về 97,9%, và lấy trung bình mỗi 10 bước cũng đưa về 97,0%.

Đợt 2 chạy trên i5-12600K + RTX 2060 (Windows 11 + WSL2), lần đầu mã CUDA và OpenMP được chạy thật:

6. **CUDA tính đúng như CPU** (105 kiểm tra, 0 lỗi). cuBLAS đạt **6,23 TFLOP/s**, bằng 96,6% đỉnh
   lý thuyết FP32 của RTX 2060. OpenMP 12 luồng chỉ nhanh hơn 1 luồng **5,3×** vì chỉ có 6 nhân vật lý.
7. Lớp conv của bài ở batch 256: GPU nhanh hơn CPU 1 luồng **55×** và nhanh hơn 12 luồng **13×**, cùng bậc
   với ≈ 40× của bài. Con số này không tính chép dữ liệu; khi tính cả chép host↔device chỉ còn 11× và 2,7×.
8. Sync SGD bão hoà ở **3,1–3,5×** (k = 6–10) với LeNet, thấp hơn nhiều so với dự đoán (5–7×). Phần
   tính toán lệch khỏi lý tưởng tới +148% ở k = 10, còn allreduce chỉ 0,04–0,20 s: giới hạn không phải
   giao tiếp. Trong WSL2 không thể phân biệt P-core và E-core, nên không kiểm chứng được giả thuyết E-core.
9. Chạy nhiều tiến trình dùng chung **một** GPU làm huấn luyện **chậm đi** (0,54× với 2 tiến trình,
   0,40× với 4), trong khi một tiến trình trên GPU nhanh hơn 1 luồng CPU **60×** (CNN trên CIFAR-10).

Hai lỗi của chính phép đo (OpenMP vô tình chạy 1 luồng sau `--verify`; GPU đo lúc xung nhịp còn thấp)
đã được phát hiện và sửa trước khi có bộ số này; xem §7.

---

## 1. Giới thiệu

Huấn luyện mạng sâu tốn nhiều giờ đến nhiều ngày trên một CPU. Bài của Hegde và Usmani đặt hai câu
hỏi: (i) tăng tốc được bao nhiêu nhờ GPU và đa nhân (§3), và (ii) nên cập nhật trọng số thế nào khi
dữ liệu chia trên nhiều máy (§5). Bài đo §3 bằng Caffe và chỉ phân tích lý thuyết §5 (bậc O(·) của
thời gian tính, chi phí và thời gian truyền thông).

Phần việc này **cài đặt lại toàn bộ từ đầu** để có thể:

- đo trực tiếp từng pha của một bước huấn luyện (nạp dữ liệu, forward, backward, chờ đồng bộ,
  giao tiếp, cập nhật), điều mà framework thường che khuất;
- thay đổi thuật toán allreduce và so với mô hình chi phí α-β;
- cài đủ bốn phương pháp ở §5 trên **cùng một** mã nguồn, để so sánh công bằng;
- chứng minh bằng kiểm thử rằng bản song song tính đúng như bản tuần tự.

Phần này bổ sung cho `vinhvn/`, vốn tái lập cùng bài bằng PyTorch trên 4× H100.

## 2. Câu hỏi nghiên cứu

| # | Câu hỏi | Thí nghiệm |
|---|---|---|
| Q1 | Bản song song có cho đúng kết quả như bản tuần tự không? | E0 |
| Q2 | Khẳng định §3 (thời gian conv tuyến tính theo batch, cách cài conv) có đúng trên CPU hiện đại không? | E1, E2 |
| Q3 | Thuật toán allreduce nào tốt nhất theo kích thước message và số tiến trình? Mô hình chi phí có khớp không? | E3 |
| Q4 | Sync SGD scale đến đâu? Giao tiếp chiếm bao nhiêu? | E4 |
| Q5 | Bốn phương pháp ở §5 đánh đổi gì về thời gian, giao tiếp và độ chính xác, với dữ liệu IID và không IID? | E5 |
| Q6 | GPU nhanh hơn CPU bao nhiêu (bài: ≈ 40×)? OpenMP tăng tốc bao nhiêu? | Đợt 2: E1, E2 |
| Q7 | Kết hợp MPI với OpenMP và nhiều tiến trình dùng chung một GPU có lợi không? | Đợt 2: E6, E7 |

## 3. Thiết kế và cài đặt

### 3.1 Lõi CNN dùng chung (C11)

| Thành phần | Cài đặt |
|---|---|
| Lớp | convolution, fully-connected, ReLU, max-pooling; hàm mất mát softmax cross-entropy (công thức (2) của bài) |
| Convolution | im2col + GEMM theo đúng cách Caffe mà bài phân tích ở §3.1: mỗi ảnh được "trải" thành ma trận K × P (K = C·k·k, P = OH·OW), rồi nhân với ma trận trọng số OC × K. Backward: dW = Σ dY·colᵀ, dcol = Wᵀ·dY, rồi col2im |
| Tối ưu | SGD có momentum và weight decay; khởi tạo He |
| Tham số | **một mảng phẳng** cho toàn bộ tham số và một mảng cho gradient, nên đồng bộ gradient chỉ là một lệnh allreduce trên một vùng nhớ liên tục |
| Mô hình | LeNet (MNIST, 61 706 tham số), cấu hình của bài (224×224, 96 bộ lọc 11×11/4, 40 lớp), CNN có hệ số độ rộng để quét kích thước mô hình |
| Dữ liệu | MNIST, CIFAR-10 (định dạng `.pdlds`), dữ liệu tổng hợp sinh theo (seed, chỉ số mẫu) |
| Ngẫu nhiên | xoshiro256** với seed tách theo (seed, rank, epoch), nên một thí nghiệm tái lập được từng bit |

### 3.2 Phiên bản tuần tự — `code/sequential/seq_train.c`

Một chương trình C một luồng (khoảng 300 dòng) chỉ dùng lõi chung và kernel CPU. Biên dịch bằng
`gcc` + `Makefile`, **không cần MPI, OpenMP hay CUDA**. Thứ tự mẫu, khởi tạo và luật cập nhật giống
hệt nhánh tuần tự của bản song song, nên đây là đường cơ sở cho mọi phép đo speedup và để kiểm tra
tính đúng.

### 3.3 Phiên bản song song — `code/`

**a) Song song dữ liệu bằng MPI** (`src/train/`). Mỗi tiến trình giữ một bản sao mô hình và một phần
dữ liệu. Bốn chiến lược của bài:

| Chiến lược | Thuật toán | Giao tiếp |
|---|---|---|
| Sync SGD (§5.1) | Mỗi rank tính gradient trên phần của batch tổng; allreduce gradient rồi chia k, cập nhật như nhau | 1 allreduce p số thực mỗi bước; có các biến thể: gộp một lần, mỗi lớp một lần, chồng lấp với backward |
| Parallel SGD (§5.1.1, Zinkevich) | Mỗi worker chạy SGD riêng trên shard của mình; lấy trung bình mô hình mỗi H bước, mỗi epoch, hoặc chỉ một lần ở cuối. **ShuffleSGD**: sau mỗi epoch, all-to-all chuyển 1/k shard tới từng worker | 1 allreduce mỗi lần lấy trung bình; all-to-all dữ liệu O(D) |
| ADMM (§5.1.2) | Consensus ADMM (Boyd 2011): x_i giải xấp xỉ bằng L bước SGD với hạng phạt ρ/2‖x − z + u_i‖²; z = trung bình(x_i + u_i); u_i += x_i − z | 1 allreduce mỗi vòng (L bước) |
| Downpour SGD (§5.2) | Rank 0 là parameter server; worker đẩy gradient tích lũy mỗi n_push bước và lấy tham số mỗi n_fetch bước, đều không chặn (Isend/Irecv); đo staleness | điểm-điểm worker ↔ server |

*Ghi chú*: Algorithm 3 trong bài (ADMM.SGD) thực chất là parameter server đồng bộ, không có biến đối
ngẫu. Ở đây cài consensus ADMM đúng nghĩa. Mẫu giao tiếp "server đồng bộ" của bài đo được bằng sync
SGD với allreduce `naive`.

**b) Sáu thuật toán allreduce** (`src/comm/comm.c`), tự viết trên giao tiếp điểm-điểm để so với mô
hình chi phí:

| Thuật toán | Thời gian theo mô hình Hockney (α độ trễ, β giây/byte, n byte) |
|---|---|
| ring (reduce-scatter + allgather) | 2(k−1)α + 2(k−1)/k · nβ |
| recursive doubling | log₂k · (α + nβ) |
| tree: binomial reduce + broadcast ("BitTorrent aggregate" của bài) | 2·log₂k · (α + nβ) |
| naive: driver trung tâm (dạng parameter server) | 2(k−1)(α + nβ) |
| `MPI_Reduce` + `MPI_Bcast`, `MPI_Allreduce` | hàm thư viện |

Mọi thuật toán cho kết quả **trùng từng bit** trên mọi rank. Nhờ đó có thể kiểm tra các bản sao mô
hình bằng hash trong khi huấn luyện.

**c) Đa luồng OpenMP** (`src/backend/cpu_kernels.c`): GEMM chia khối theo cache, chia hàng của C cho
các luồng; im2col và các phép theo phần tử song song theo ảnh trong batch. Đây là hai cách dùng
nhiều nhân mà bài nêu ở §2.

**d) GPU bằng CUDA** (`src/backend/cuda_kernels.cu`, sm_75): GEMM naive, GEMM chia tile bằng shared
memory, cuBLAS; im2col/col2im, pooling, softmax. Backward của col2im và max-pool dùng dạng "gom"
(không dùng atomics) nên kết quả tất định. MPI chạy trên bản sao tham số ở host (không đòi hỏi
CUDA-aware MPI). Pha chép device ↔ host được đo riêng.

### 3.4 Đo đạc và kiểm tra tính đúng

- Mỗi bước được chia thành 10 pha có đồng hồ riêng trên mỗi rank: `data, h2d, forward, backward,
  transfer, sync_wait, comm, update, eval, shuffle`. `sync_wait` là thời gian chờ ở barrier ngay
  trước allreduce, tức là chi phí của chênh tải.
- Thời gian huấn luyện không tính lúc đánh giá. Một phép toán tập thể được tính theo rank chậm nhất.
- `pdl_check` kiểm tra: gradient bằng sai phân hữu hạn (bỏ qua các điểm rơi vào chỗ gãy của
  ReLU/max-pool), các biến thể GEMM, kernel CUDA so với CPU, tính tất định, allreduce đúng tổng,
  **sync SGD trên k tiến trình bằng tuần tự**, và replica trùng từng bit.
- Mỗi file kết quả JSON ghi đủ cấu hình và môi trường (CPU, trình biên dịch, thư viện MPI, commit).

## 4. Phương pháp thực nghiệm

| | Đợt 1 (22/11) | Đợt 2 (23/11) |
|---|---|---|
| Phần cứng | Intel Core i5-8257U, 4 nhân / 8 luồng, 16 GB; không GPU | Intel Core i5-12600K (6P + 4E, 16 luồng), 15,8 GB (WSL2 được cấp 7,6 GB); RTX 2060 6 GB |
| Phần mềm | macOS 13.7, Apple Clang 14 (không OpenMP), MPICH 5.0.2, `-O3 -march=native` | Windows 11 + WSL2 (Ubuntu 24.04), GCC 13.3, CUDA 12.0, OpenMPI 4.1.6, `-O3 -march=native` |
| Điều kiện | — | power plan High performance, tải nền Windows 0–7%, ghi trong `manifest.json` |
| Dữ liệu | MNIST 60k/10k; E2 dùng hình học ảnh 224×224 của bài | thêm CIFAR-10 (E7) |
| Lặp | E1–E3: 5–20 lần đo cho mỗi điểm; E4: 3 lần chạy; E5: 2 seed | E1–E3: 3 lần chạy × 5–20 lần đo; E4, E6, E7: 3 lần; E5: 2 seed |
| Quy trình | Dự đoán được ghi vào `experiments/PLAN.md` **trước khi chạy**. Cấu hình YAML ở `experiments/configs/`. Số đo gốc ở `experiments/results/` | như đợt 1, nhưng chỉ 2 trong 12 dự đoán được commit trước khi chạy (xem `PLAN.md`) |

## 5. Kết quả

### 5.1 Tính đúng (Q1)

| Kiểm tra | Kết quả |
|---|---|
| Gradient giải tích so với sai phân hữu hạn (4 mạng, 277 tham số) | sai số tương đối ≤ 1,1·10⁻⁴ |
| Sync SGD k = 2 / k = 4 so với tuần tự cùng batch tổng | lệch tham số 5,7·10⁻⁸ / 2,9·10⁻⁸ |
| Replica sau 6 thuật toán allreduce | trùng từng bit |
| `seq_train` so với `pdl_train --strategy sequential` | loss và accuracy giống hệt ở mọi epoch |
| CTest (12 bài, gồm chạy thử cả 5 chiến lược) | 12/12 đạt |

Bản song song **tính đúng như bản tuần tự**. Sai lệch duy nhất là thứ tự cộng số thực.

### 5.2 Tính toán: GEMM và lớp conv của bài (Q2)

![Conv theo batch](figures/f1_conv_batch.png)

*Hình 1. Lớp conv đầu của bài (96 bộ lọc 11×11, stride 4, ảnh 224×224) trên một nhân CPU. Trái:
thời gian tuyến tính theo batch. Phải: conv trực tiếp chậm hơn im2col + GEMM bao nhiêu lần.*

- Thời gian **tuyến tính theo batch**: 5,78 ms/ảnh, R² = 0,9987, đúng dạng O(n) bài mô tả. Bài
  báo cáo độ dốc 6,45 cho đường CPU (Xeon E5-2670, Caffe) nhưng phần văn bản không ghi đơn vị. Nếu
  đơn vị là ms/ảnh thì hai kết quả cùng bậc, nhưng không nên so sánh trực tiếp hai con số.
- im2col + GEMM nhanh hơn conv trực tiếp **12,4–15,5×**. Lý do chính là GEMM dùng lại dữ liệu trong
  cache, không phải số phép tính (hai cách có cùng số FLOP).

![GEMM](figures/f2_gemm.png)

*Hình 2. GEMM N×N×N một luồng: cài đặt chia khối theo cache so với ba vòng lặp lồng nhau.*

- GEMM chia khối giữ 14,5–17,2 GFLOP/s ở mọi N; bản naive tụt từ 2,1 xuống 0,65 GFLOP/s, tức
  chậm hơn **tới 25×** khi N = 1024. Dự đoán ban đầu (3–10×) đã đánh giá thấp tác động của cache.
- Lớp FC (nhân ma trận–vector) tăng đúng O(n²), trừ một bước nhảy khi ma trận trọng số (16 MB)
  vượt bộ nhớ L3 (6 MB).

### 5.3 Giao tiếp: sáu thuật toán allreduce (Q3)

![Allreduce](figures/f3_allreduce.png)

*Hình 3. Trái: thời gian allreduce theo kích thước message với k = 4 (log–log). Phải: allreduce
16 MiB theo số tiến trình.*

- Đo bằng ping-pong: α = 2,66 µs, băng thông 3,4 GB/s (MPI qua shared memory).
- **Message nhỏ** (≤ 4 KiB): hàm thư viện (`MPI_Allreduce` 1,0 µs) nhanh hơn mọi bản tự viết 5–16×.
  Trong các bản tự viết, recursive doubling và tree (log k vòng) thắng ring (2(k−1) vòng), đúng
  như mô hình.
- **Message lớn** (16 MiB): ring ≈ `MPI_Allreduce` < recursive doubling < tree < naive. Naive tăng
  gần tuyến tính theo k (14 → 40 → 111 ms), xác nhận driver trung tâm là nút thắt (bài §5.2).
  Allreduce dạng cây, mà bài dùng để phân tích, **không** phải lựa chọn tốt khi gradient lớn: cây gửi
  cả vector ở mỗi tầng, còn ring chỉ gửi 2(k−1)/k của vector.
- Mô hình α-β dự đoán allreduce lớn sai trung vị 14% với ring và 22% với `MPI_Allreduce`, nhưng
  50–165% với các thuật toán dồn việc về một rank. Khi k = 8 vượt số nhân thật, vài thuật toán chậm
  đi 5–10× do hai tiến trình tranh một nhân.

### 5.4 Strong scaling của Sync SGD (Q4)

![Strong scaling](figures/f4_strong_scaling.png)

*Hình 4. Trái: speedup đo được (trung bình 3 lần, thanh lỗi là độ lệch chuẩn) so với lý tưởng và
với mô hình α-β. Phải: tỉ lệ thời gian của tính toán, chờ đồng bộ và allreduce.*

| k | thời gian 1 epoch | speedup | hiệu suất | chờ + giao tiếp | test acc |
|---|---|---|---|---|---|
| 1 | 19,20 s | 1,00 | 100% | 0% | 97,97% |
| 2 | 9,93 s | 1,93 | 97% | 0,6% | 97,86% |
| 4 | 5,84 s | 3,29 | 82% | 4,4% | 97,89% |
| 8 (hyper-thread) | 5,52 s | 3,48 | 44% | 14,8% | 98,04% |

- Speedup 3,29× trên 4 nhân với cùng độ chính xác.
- Mô hình α-β (tính toán chia đều cho k cộng với allreduce 247 KB) dự đoán 3,95×. Phần thiếu
  **không phải do giao tiếp**: allreduce chỉ mất 0,05 s mỗi epoch. Thời gian forward + backward ở
  k = 4 tăng 15% so với lý tưởng, vì 4 nhân cùng chạy làm giảm turbo và phải chia băng thông bộ nhớ.
  Thời gian chờ đồng bộ (0,21 s) lớn gấp 4 lần allreduce, do dao động từng bước giữa các rank.
- Với k = 8, hai tiến trình chung một nhân nên gần như không nhanh thêm (3,48×).

### 5.5 Bốn phương pháp tối ưu phân tán (Q5)

![Chiến lược](figures/f5_strategies.png)

*Hình 5. Trái: độ chính xác test sau 3 epoch với dữ liệu IID (xanh) và không IID, tức shard sắp theo
nhãn (cam). Phải: thời gian huấn luyện và tỉ lệ giao tiếp.*

| Phương pháp (4 tiến trình) | IID | Không IID | Thời gian | Speedup | Byte gửi / rank |
|---|---|---|---|---|---|
| Tuần tự (1 tiến trình) | 98,64% | — | 57,8 s | 1,00× | 0 |
| Sync SGD | 98,58% | — | 17,1 s | 3,38× | 260 MB |
| PSGD, trung bình mỗi epoch | 98,12% | **77,68%** | 14,7 s | 3,94× | 1,1 MB |
| PSGD + ShuffleSGD | — | **97,87%** | 17,1 s | — | 18,8 MB |
| PSGD, trung bình mỗi 10 bước | 98,15% | 96,98% | 15,4 s | 3,75× | 26,7 MB |
| ADMM (20 bước/vòng) | 98,06% | 88,48% | 16,0 s | 3,61× | 13,3 MB |
| Downpour (3 worker + server) | 96,32% | 93,94% | 25,3 s | 2,29× | 115,5 MB |

Nhận xét:

1. **Sync SGD** cho đúng độ chính xác của tuần tự (khác 0,06 điểm) và nhanh hơn 3,4×. Đây là lựa
   chọn an toàn nhất, nhưng tốn băng thông nhất (260 MB mỗi rank).
2. **Parallel SGD** giao tiếp ít nhất (1,1 MB) và nhanh nhất (3,94×) **khi dữ liệu IID**. Giả thiết
   "mỗi máy có một mẫu đại diện" của bài là then chốt: khi shard sắp theo nhãn, độ chính xác rơi
   20 điểm. Hai cách khắc phục đều hiệu quả. ShuffleSGD, như bài đề xuất, cho 97,9% nhưng phải
   chuyển dữ liệu. Lấy trung bình thường xuyên hơn cho 97,0% mà không phải di chuyển dữ liệu.
3. **ADMM** ngang PSGD khi IID. Khi không IID, biến đối ngẫu giúp nó hơn PSGD-mỗi-epoch (88,5% so
   với 77,7%), nhưng kém PSGD-mỗi-10-bước (97,0%), phương pháp chỉ tốn gấp đôi lượng giao tiếp (26,7 so với 13,3 MB).
4. **Downpour** chậm nhất trong các phương pháp song song: một server phục vụ 3 worker, giao tiếp
   chiếm 18%. Độ chính xác thấp nhất khi IID vì gradient bị trễ trung bình 11 phiên bản (staleness).
   Bài cảnh báo đúng điều này ở §5.2.1: thông lượng bị chặn bởi server. Muốn thấy điểm bão hòa
   k* = T/τ cần nhiều worker hơn (đợt 2).

### 5.6 Đợt 2: GPU và OpenMP (Q6)

Từ đây là kết quả đợt 2 trên i5-12600K + RTX 2060. Lần đầu mã CUDA chạy thật: `ctest` 13/13,
`pdl_check` 35/35 (CPU và CUDA) và 67/67 với MPI; sai số của GEMM CUDA so với CPU ≤ 2,8·10⁻⁷, của
pipeline CNN bằng 0. Chi tiết đối chiếu dự đoán ở `experiments/results/SUMMARY_dot2.md`.

![GEMM trên CPU và GPU](figures/g1_gemm_gpu.png)

*Hình 6. GEMM N×N×N. cuBLAS đạt 6,23 TFLOP/s ở N = 4096 (96,6% đỉnh lý thuyết FP32 ≈ 6,45 TFLOP/s).*

| N | CPU blocked (1 luồng) | CPU OpenMP (12 luồng) | CUDA tiled | cuBLAS |
|---|---|---|---|---|
| 64 | 34,3 | 9,5 | 8,0 | 10,5 |
| 1024 | 36,0 | 190 | 1 145 | 5 292 |
| 4096 | 25,9 | 135 | 1 198 | 6 231 |

*GFLOP/s, trung bình 3 lần chạy.*

- **OpenMP 12 luồng nhanh hơn 1 luồng 5,3×** ở N = 1024: 12 luồng chỉ chạy trên 6 nhân vật lý (mỗi
  nhân hai luồng HT) nên không thể gần 12×. Ở N = 64 OpenMP chậm hơn 1 luồng (9,5 so với 34,3).
- **Kernel viết tay xa cuBLAS**: tiled đạt 1,2 TFLOP/s (19% đỉnh), cuBLAS gấp 5,2× tiled. Thứ tự
  naive < tiled < cuBLAS giống trên CPU.
- **GPU thua CPU với phép tính nhỏ**: ở N = 64 cuBLAS chỉ 10,5 GFLOP/s, thấp hơn CPU 1 luồng.

![Conv trên CPU và GPU](figures/g2_conv_gpu.png)

*Hình 7. Lớp conv đầu của bài. Trái: thời gian forward theo batch. Phải: GPU nhanh hơn CPU bao nhiêu
lần. Đường chấm là con số ≈ 40× của bài.*

- Thời gian **tuyến tính theo batch ở mọi cách cài** (R² ≥ 0,9988): 2,16 ms/ảnh (CPU 1 luồng),
  0,52 (OpenMP), 0,040 (GPU), 0,192 (GPU có chép dữ liệu).
- **Batch 256: GPU nhanh hơn CPU 1 luồng 54,5× và nhanh hơn OpenMP 12 luồng 13,2×**, cùng bậc với
  ≈ 40× của bài (K520 so với 1 nhân Xeon). Đây là thời gian khi dữ liệu **đã ở trên GPU**. Tính cả chép
  host↔device thì chỉ còn 11,2× và 2,7×.
- **Dự đoán "20–60× so với OpenMP" là sai** (đo 13,2×): OpenMP chỉ tăng tốc conv 4,1× ở batch 256,
  chứ không phải 5–8× như GEMM, vì với batch > 1 code song song theo từng ảnh. Do đó batch 2 chỉ dùng
  2 luồng (1,1× so với 1 luồng). Đây là hạn chế của cài đặt hiện tại.
- **im2col + GEMM chỉ nhanh hơn conv trực tiếp 1,4–2,7×** trên máy này (đợt 1: 12–15×). Chưa xác định
  nguyên nhân (có thể do cache lớn hơn và GCC vector hoá tốt hơn Apple Clang), nên không nên suy rộng kết
  luận "12–15×" của đợt 1 sang CPU khác.
- **Lớp FC (ma trận–vector)**: khi trọng số đã ở trên GPU, cuBLAS ở n = 4096 nhanh hơn CPU 1 luồng 13,8×
  (0,24 so với 3,30 ms). Nhưng **tính cả chép dữ liệu thì chậm hơn CPU 2,3×** (7,68 ms) và thua ở mọi n
  ≤ 4096. Ma trận–vector bị giới hạn bởi băng thông, nên GPU chỉ có lợi khi trọng số ở lại trên GPU.

### 5.7 Đợt 2: giao tiếp và strong scaling (Q3, Q4)

![Allreduce, đợt 2](figures/g3_allreduce_dot2.png)

*Hình 8. Allreduce theo số tiến trình, k = 2…10. Trái: message 4 KiB. Phải: 16 MiB.*

- Độ trễ `MPI_Allreduce` 4 B: 0,3 µs (k = 2) và 1,0 µs (k = 4). Ping-pong đạt đỉnh 21,9 GB/s ở 4 MiB và
  8,4 GB/s ở 64 MiB (vượt cache); khớp α-β cho α = 0,43 µs và 14,3 GB/s.
- **Message lớn**: ring nhanh nhất ở mọi k ≥ 4. Khác đợt 1, ring hơn hẳn `MPI_Allreduce` mặc định của OpenMPI khi
  k ≥ 6 (33,2 so với 46,9 ms ở k = 10). **Recursive doubling sụp ở k ≥ 8** (71,8 ms ở k = 8 và 85,2 ms ở
  k = 10, gấp 2,6× ring); nguyên nhân chưa tách được.
- Mô hình α-β sai nhiều hơn đợt 1: trung vị 43% (ring 25%, `MPI_Allreduce` 38%, naive 104%), so với 14% và 22%.

![Strong scaling, đợt 2](figures/g4_scaling_dot2.png)

*Hình 9. Strong scaling sync SGD (LeNet/MNIST, 3 epoch, batch tổng 256). Trái: speedup, ghim và không
ghim tiến trình cho cùng kết quả. Phải: thời gian theo pha; vạch đen là thời gian tính toán lý tưởng T₁/k.*

| k | 1 epoch [s] | speedup | hiệu suất | tính toán so với lý tưởng | chờ đồng bộ [s] |
|---|---|---|---|---|---|
| 1 | 7,19 | 1,00 | 100% | — | 0 |
| 2 | 4,01 | 1,79 | 90% | +10% | 0,13 |
| 4 | 2,47 | 2,91 | 73% | +33% | 0,16 |
| 6 | 2,30 | 3,12 | 52% | +67% | 0,77 |
| 8 | 2,28 | 3,16 | 40% | +105% | 1,17 |
| 10 | 2,09 | 3,45 | 34% | +148% | 0,71 |

- **Speedup bão hoà ở 3,1–3,5× từ k = 6**, thấp hơn nhiều so với dự đoán (5× ở k = 6 và 6–7× ở k = 8–10).
  Giống đợt 1, giới hạn nằm ở **phần tính toán**, không phải giao tiếp: forward + backward lệch khỏi
  lý tưởng tới +148% ở k = 10, còn allreduce chỉ 0,04–0,20 s.
- **Giả thuyết "E-core chậm hơn P-core" không kiểm chứng được.** Trong WSL2, tốc độ GEMM 1 luồng trên
  mười CPU đã thử (0, 1, 2, 4, 10–15) đều 29,4–31,3 GFLOP/s: Windows xếp vCPU lên nhân vật lý, VM không
  thấy sự khác biệt. Chạy **không ghim** cho cùng kết quả (3,46× so với 3,45× ở k = 10). Các nguyên nhân
  khả dĩ còn lại (xung nhịp toàn nhân thấp hơn, tranh chấp bộ nhớ và L3, lập lịch của Windows) chưa được
  tách riêng bằng phép đo nào.

### 5.8 Đợt 2: MPI × OpenMP và nhiều tiến trình dùng chung GPU (Q7)

![Lai MPI x OpenMP và GPU dùng chung](figures/g5_hybrid_gpu_mpi.png)

*Hình 10. Trái: 12 luồng phần cứng chia thành np tiến trình × (12/np) luồng OpenMP (2 epoch). Phải:
CNN trên CIFAR-10 (20 000 ảnh, 1 epoch), CPU 1 luồng/rank so với GPU, thang log.*

- **Lai MPI × OpenMP**: tốt nhất là 4×3 và 6×2 (1,95–1,96 s/epoch), nhanh hơn chỉ-OpenMP (1×12,
  2,66 s) 1,37×. Chỉ-OpenMP tệ nhất vì OpenMP chỉ tăng tốc LeNet 2,7× so với 1 luồng (các lớp quá nhỏ
  cho 12 luồng); chỉ-MPI (12×1) mất 10,7% thời gian cho giao tiếp và chờ.
- **GPU nhanh hơn 60× một luồng CPU** (1,69 so với 101,1 s) và 16× so với 8 tiến trình CPU (27,4 s)
  với CNN trên CIFAR-10. Độ chính xác không đổi (47,5–48,0% sau 1 epoch).
- **Nhiều tiến trình dùng chung một GPU làm chậm đi**: 3,15 s với 2 tiến trình và 4,29 s với 4
  (0,54× và 0,40×). Mỗi rank có context CUDA riêng, các kernel chia thời gian GPU, và gradient phải qua bộ
  nhớ host trước khi allreduce (chép 0,41 và 0,77 s). Với một GPU, nên dùng **một** tiến trình.
- Chưa chạy LeNet/MNIST trên GPU, nên dự đoán "GPU chỉ nhanh hơn 2–5× với mô hình rất nhỏ" chưa được kiểm chứng.

### 5.9 Đợt 2: bốn phương pháp tối ưu phân tán, 10 epoch (Q5)

![Chiến lược, đợt 2](figures/g6_strategies_dot2.png)

*Hình 11. Độ chính xác test sau 10 epoch (trung bình 2 seed, thanh lỗi là độ lệch chuẩn), 4 và 8 tiến
trình, dữ liệu IID (xanh) và shard theo nhãn (cam).*

| Phương pháp | k | IID | không IID | speedup | byte gửi / rank |
|---|---|---|---|---|---|
| Tuần tự | 1 | 98,9% | — | 1,00× | 0 |
| Sync SGD | 4 / 8 | 98,9% / 98,8% | — | 3,14× / 3,10× | 1,73 GB / 1,01 GB |
| PSGD, TB mỗi epoch | 4 / 8 | 99,0% / 99,0% | **94,6%** / **88,4%** | 3,58× / 3,86× | 3,7 MB / 4,3 MB |
| PSGD + ShuffleSGD | 4 / 8 | 99,1% / 99,0% | 99,0% / 99,0% | 3,58× / 3,91× | 83 MB / 51 MB |
| PSGD, TB mỗi 10 bước | 4 / 8 | 98,9% / 98,9% | 98,6% / 98,3% | 3,37× / 3,49× | 174 MB / 104 MB |
| ADMM | 4 / 8 | 99,1% / 99,0% | 96,9% / **87,4%** | 3,33× / 3,42× | 89 MB / 52 MB |
| Downpour (push mỗi 1 bước) | 4 / 8 | 98,9% / 98,8% | 97,3% / 98,4% | 2,48× / 2,54× | 1,71 GB / 0,78 GB |

- Kết luận đợt 1 giữ nguyên: Sync SGD bằng tuần tự về độ chính xác; PSGD nhanh nhất nhưng cần
  ShuffleSGD hoặc lấy trung bình thường xuyên khi dữ liệu không IID. Khi tăng từ 4 lên 8 tiến trình,
  PSGD-mỗi-epoch rơi sâu hơn (94,6% → 88,4%) vì mỗi shard chứa ít nhãn hơn.
- **ADMM xấu đi khi k tăng trên dữ liệu không IID** (96,9% → 87,4%), giống PSGD-mỗi-epoch.
- **Downpour không còn thấp nhất về độ chính xác** (98,8–98,9% trên IID) như dự đoán đợt 1, vì sau 10
  epoch mọi phương pháp đã hội tụ gần nhau. Nó vẫn chậm nhất (2,5–3,0×) và tốn 34% thời gian cho giao tiếp ở k = 8.
- Không so sánh trực tiếp với con số 77,7% của đợt 1: đợt này huấn luyện 10 epoch với siêu tham số khác.

## 6. Thảo luận: đối chiếu với bài báo

| Khẳng định của bài | Kết quả ở đây |
|---|---|
| §3: thời gian conv trên CPU tuyến tính theo batch (O(n)) | **Khẳng định** (R² = 0,9987; độ dốc 5,78 ms/ảnh) |
| §3.1: im2col + GEMM là cách hiệu quả, GEMM chiếm phần lớn thời gian | **Khẳng định** (nhanh hơn conv trực tiếp 12–15×) |
| §3: GPU nhanh hơn CPU ≈ 40× | **Khẳng định về bậc độ lớn**: 54,5× so với 1 luồng (batch 256, dữ liệu đã ở GPU); chỉ 11× nếu tính chép dữ liệu, và 13× so với OpenMP 12 luồng |
| §3: GPU càng nhanh hơn khi batch càng lớn | **Đúng một phần**: tỉ lệ GPU/1 luồng tăng từ 27× (batch 1) lên khoảng 50× ở batch 4–8, rồi đi ngang ở 52–56× đến batch 256 |
| §5: allreduce kiểu cây ("BitTorrent aggregate"), O(p log k) | Đúng về dạng, nhưng trên một node **ring nhanh hơn cây** với gradient lớn (đợt 2: k = 4…10) |
| §5.1.1: Parallel SGD cần dữ liệu trộn đều; ShuffleSGD khắc phục | **Khẳng định** (77,7% → 97,9%); bổ sung: trung bình thường xuyên cũng khắc phục được |
| §5.1.2: ADMM tránh được việc trộn dữ liệu | **Một phần**: tốt hơn PSGD-mỗi-epoch khi không IID, nhưng chưa bằng trộn dữ liệu |
| §5.2.1: Downpour bị giới hạn bởi server | **Khẳng định**: chậm nhất ở k = 4 và 8 (2,5–3,0×); giao tiếp 15% ở k = 4 và 34% ở k = 8. Chưa đo điểm bão hòa k* = T/τ |

## 7. Giới hạn

- **Đợt 2 chạy trên WSL2, không phải Linux gốc.** Windows xếp lịch cho vCPU nên: (i) không ghim được
  tiến trình vào P-core hay E-core (tốc độ 1 luồng trên mười CPU đã thử đều như nhau, và chạy không ghim cho
  cùng kết quả), do đó **hiệu ứng E-core/P-core của i5-12600K chưa được kiểm chứng**; (ii) power plan và tải
  nền của Windows ảnh hưởng số đo (đã đặt High performance và ghi vào `manifest.json`); (iii) WSL2 chỉ
  được cấp 7,6 GB RAM.
- **Hai lỗi của chính phép đo đã được sửa trước khi có số đợt 2, và chúng cho thấy rủi ro của việc tin
  số đo đầu tiên.** (1) `--verify` tạo context tham chiếu 1 luồng và gọi `omp_set_num_threads(1)` toàn cục,
  khiến GEMM OpenMP đo được 31 GFLOP/s (bằng 1 luồng) thay vì 190. (2) Warm-up cố định quá ngắn nên GPU đo
  lúc xung nhịp còn thấp: cuBLAS 0,38–0,65 TFLOP/s thay vì 3,1–5,8. Cả hai được phát hiện nhờ so sánh với
  một lần chạy riêng lẻ và với đỉnh lý thuyết, chứ không phải nhờ test. Ghi tại `code/docs/ai_effort_log.csv`.
- **Hạn chế của cài đặt chưa sửa**: OpenMP cho conv song song theo ảnh nên batch 2–4 chỉ dùng 2–4 luồng;
  `cpu-omp` FC ở n = 512 chậm bất thường (0,43 ms, chưa điều tra).
- `manifest.json` của đợt 2 ghi `git.dirty = true`: kết quả được tạo từ working tree chưa commit.
- Mọi phép đo đều trên **một máy**. MPI đi qua shared memory, nên "băng thông mạng" trong mô hình là
  băng thông bộ nhớ. Không suy ra được hành vi giữa nhiều máy.
- Mô hình nhỏ (LeNet, 247 KB gradient) làm giao tiếp rẻ. Với mô hình lớn, tỉ lệ giao tiếp sẽ khác
  (cấu hình `model_scaling.yaml` dành cho đợt 2).
- Số lần lặp còn ít (3 lần ở E4, 2 seed ở E5). Độ lệch chuẩn ở E5 với dữ liệu không IID là ± 4
  điểm, nên các chênh lệch nhỏ hơn vài điểm chưa có ý nghĩa thống kê.
- Siêu tham số (learning rate, ρ, n_push) chọn một lần, chưa tinh chỉnh cho từng phương pháp, nên so
  sánh độ chính xác là so sánh "cấu hình mặc định hợp lý", không phải tốt nhất của mỗi phương pháp.

## 8. Hướng tiếp theo

1. **Chạy trên Linux gốc** (không qua WSL2) để ghim được P-core/E-core và kiểm chứng giả thuyết E-core
   cho strong scaling; so sánh với kết quả WSL2 ở đây.
2. Tách nguyên nhân khiến phần tính toán lệch khỏi lý tưởng khi k ≥ 6 (xung nhịp toàn nhân, băng thông
   bộ nhớ, L3), ví dụ bằng bộ đếm phần cứng hoặc bằng cách chạy k bản sao độc lập không giao tiếp.
3. Cho OpenMP song song theo cả batch lẫn GEMM (lồng nhau) để batch 2–4 dùng đủ luồng; chạy LeNet/MNIST
   trên GPU để kiểm tra dự đoán "GPU chỉ 2–5×".
4. Tăng số seed cho E5 và thêm Downpour với nhiều worker để tìm điểm bão hòa k* = T/τ; thử mô hình lớn
   hơn (`model_scaling.yaml`) để xem tỉ lệ giao tiếp thay đổi thế nào.
5. Điều tra vì sao conv trực tiếp chỉ chậm hơn im2col 1,4–2,7× trên CPU này, và vì sao recursive doubling
   sụp ở k ≥ 8.

## 9. Tái lập

Xem [`../README.md`](../README.md), mục "Cách chạy lại". Tóm tắt:

```bash
cd tuannn/code/sequential && make && ./seq_train --epochs 3          # bản tuần tự
cd tuannn/code && scripts/build.sh && ctest --test-dir build          # bản song song + kiểm tra
for e in e0_correctness e1_gemm e2_paper_cpu e3_comm e4_strong_scaling e5_strategies; do
  pdlbench all ../experiments/configs/dot1_$e.yaml --results-dir ../experiments/results; done
# Đợt 2 (máy có GPU; trên WSL2 cần export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH)
for e in e0_correctness e1_gemm e2_paper_cpu_gpu e3_comm e4_strong_scaling e4b_strong_scaling_unbound \
         e5_strategies e6_hybrid e7_gpu_mpi; do
  pdlbench all ../experiments/configs/dot2_$e.yaml --results-dir ../experiments/results; done
python ../report/make_figures.py                                       # vẽ lại hình của báo cáo
```

## Tài liệu tham khảo

1. V. Hegde, S. Usmani. *Parallel and Distributed Deep Learning*. Stanford University, 2016.
2. M. Zinkevich, M. Weimer, A. Smola, L. Li. *Parallelized Stochastic Gradient Descent*. NIPS 2010.
3. S. Boyd, N. Parikh, E. Chu, B. Peleato, J. Eckstein. *Distributed Optimization and Statistical
   Learning via the Alternating Direction Method of Multipliers*. Foundations and Trends in ML, 2011.
4. J. Dean et al. *Large Scale Distributed Deep Networks*. NIPS 2012.
5. Y. Jia et al. *Caffe: Convolutional Architecture for Fast Feature Embedding*. 2014.
6. R. Thakur, R. Rabenseifner, W. Gropp. *Optimization of Collective Communication Operations in
   MPICH*. IJHPCA, 2005.
