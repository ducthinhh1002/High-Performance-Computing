# Kế hoạch thực nghiệm

Bám bài "Parallel and Distributed Deep Learning" (Hegde, Usmani, Stanford 2016 —
[hgpu.org/?p=16046](https://hgpu.org/?p=16046)): §3 (CPU so với GPU, im2col + GEMM) và §5
(Sync SGD, Parallel SGD, ADMM, Downpour SGD). Khác với `vinhvn/` (PyTorch trên H100), phần này
**tự cài đặt toàn bộ bằng C/MPI/OpenMP/CUDA** (`code/`), nên đo được cả những chỗ framework che
mất: thuật toán allreduce, chi phí từng pha, tính tương đương tuần tự ↔ song song.

Dự đoán dưới đây được ghi **trước** khi chạy. Kết quả đối chiếu nằm trong `results/SUMMARY_*.md`,
kể cả những chỗ dự đoán sai.

## Đợt 1 — máy phát triển (CPU + MPI)

Phần cứng: Intel Core i5-8257U (4 nhân / 8 luồng, 1,4 GHz cơ bản, turbo 3,9 GHz), 16 GB RAM,
macOS 13, Clang, MPICH 5.0.2. **Không có GPU, không có OpenMP** (Apple Clang), nên đợt này
chỉ đo CPU tuần tự và MPI. Dữ liệu: MNIST (60k/10k), LeNet (61 706 tham số).

| Mã | Thí nghiệm | Câu hỏi | Ước tính |
|---|---|---|---|
| E0 | Tính đúng | Bản song song có cho đúng kết quả như bản tuần tự không? | 1 phút |
| E1 | GEMM CPU | Tối ưu cache (blocked) nhanh hơn naive bao nhiêu? | 3 phút |
| E2 | §3 trên CPU | Thời gian conv có tuyến tính theo batch như bài nói không? im2col+GEMM so với conv trực tiếp? | 5 phút |
| E3 | Giao tiếp | 6 thuật toán allreduce xếp hạng thế nào theo kích thước message và số tiến trình? Mô hình α-β có khớp không? | 5 phút |
| E4 | Strong scaling | Sync SGD tăng tốc bao nhiêu khi tăng tiến trình (1–8) với batch tổng cố định? | 10 phút |
| E5 | So sánh 5 chiến lược | Sync / Parallel SGD / ADMM / Downpour đánh đổi gì về độ chính xác và thời gian? Shuffle có cần khi dữ liệu không IID? | 15 phút |

### E0 — tính đúng (`pdl_check`, `ctest`)
Kiểm tra gradient bằng sai phân hữu hạn; sync SGD trên k tiến trình so với tuần tự cùng batch tổng;
replica trùng từng bit; `seq_train` (bản tuần tự thuần C) so với `pdl_train --strategy sequential`.
- Dự đoán: tất cả qua. Sai số gradient ≲ 1e-3. Sync SGD khác tuần tự ≲ 1e-4 (chỉ do thứ tự cộng số
  thực). `seq_train` và `pdl_train` cho loss **giống hệt** vì cùng thứ tự mẫu và cùng khởi tạo.

### E1 — GEMM (`pdl_bench_compute --op gemm`)
N = 64…1024, cpu-naive so với cpu-blocked, 5 lần lặp.
- Dự đoán: blocked nhanh hơn naive 3–10× khi N ≥ 256 (naive đọc B theo cột, trượt cache);
  blocked đạt 5–15 GFLOP/s trên một nhân (AVX2, không dùng thư viện BLAS).

### E2 — §3 của bài trên CPU
Lớp conv đầu của bài (ảnh 224×224 xám, 96 bộ lọc 11×11, stride 4 → 54×54), batch 1…64;
im2col+GEMM so với conv trực tiếp. FC kích thước n = 256…4096 với batch 1 (nhân ma trận–vector).
- Bài: thời gian CPU tuyến tính theo batch (độ dốc 6,45 ms/ảnh trên Xeon E5-2670), FC tăng O(n²).
- Dự đoán: (a) tuyến tính theo batch, R² > 0,99, độ dốc cỡ 5–15 ms/ảnh; (b) im2col+GEMM nhanh hơn
  conv trực tiếp ≥ 5×; (c) FC: thời gian tăng ≈ 4× khi n tăng 2× (O(n²)).

### E3 — giao tiếp (`pdl_bench_comm`)
k = 2, 4, 8 tiến trình; message 4 B…16 MiB; 6 thuật toán; 20 lần lặp, lấy max theo rank.
- Bài: allreduce kiểu "BitTorrent aggregate" (cây) tốn O(p log k) thời gian.
- Dự đoán: (a) α (độ trễ một chiều) cỡ 1–5 µs vì MPI đi qua shared memory; (b) message nhỏ: rd/tree
  nhanh hơn ring (ít vòng hơn); (c) message ≥ 1 MiB: ring hoặc mpi nhanh nhất, naive chậm nhất và
  tăng gần tuyến tính theo k; (d) k = 8 dùng hyper-thread nên chậm hẳn so với k = 4; (e) mô hình
  α-β dự đoán allreduce lớn sai < 50% (nhỏ thì sai nhiều hơn vì bỏ qua chi phí đồng bộ).

### E4 — strong scaling (sync SGD)
LeNet trên MNIST, batch tổng 256, 1 epoch, k = 1, 2, 4, 8, lặp 3 lần.
- Dự đoán: S(2) ≈ 1,8–1,95; S(4) ≈ 2,8–3,4; S(8) không hơn S(4) quá 0,5 (chỉ có 4 nhân vật lý);
  tỉ lệ giao tiếp < 10% ở k = 4 vì mô hình nhỏ (247 KB gradient). Nếu S(4) < 2,5 thì nút thắt là
  băng thông bộ nhớ hoặc chênh tải, không phải giao tiếp. Kiểm tra bằng `sync_wait` và `load_imbalance`.

### E5 — so sánh chiến lược
MNIST, 4 tiến trình (Downpour: 1 server + 3 worker), 3 epoch, 2 cách chia dữ liệu: `iid` và `sorted`
(mỗi worker gần như chỉ có vài nhãn, đúng tình huống của bài §5.1.1).
- Dự đoán: (a) sync ≈ tuần tự (độ chính xác cuối chênh < 0,5 điểm) vì về toán là cùng thuật toán;
  (b) với `sorted`, Parallel SGD không shuffle kém hẳn so với `iid` (> 5 điểm), bật ShuffleSGD kéo lại
  về gần `iid` (< 1 điểm); (c) ADMM hội tụ chậm hơn Parallel SGD theo epoch nhưng giao tiếp ít nhất;
  (d) Downpour cho độ chính xác thấp nhất sau cùng số epoch, staleness trung bình cỡ 5–15.

## Đợt 2 — máy mục tiêu (i5-12600K + RTX 2060) — đã chạy 23/11/2026

Phần cứng: Intel Core i5-12600K (6 P-core + 4 E-core, 16 luồng), RTX 2060 6 GB, Windows 11 + WSL2
(Ubuntu 24.04), GCC 13, CUDA 12.0, OpenMPI 4.1.6. Kết quả và đối chiếu ở
[`results/SUMMARY_dot2.md`](results/SUMMARY_dot2.md).

| Mã | Cấu hình (`configs/dot2_*.yaml`) | Nội dung |
|---|---|---|
| E0 | `e0_correctness` | `pdl_check` CPU + CUDA, 1/2/4 tiến trình |
| E1 | `e1_gemm` | GEMM CPU (naive/blocked/OpenMP) và GPU (naive/tiled/cuBLAS), N = 64…4096 |
| E2 | `e2_paper_cpu_gpu` | §3 của bài đầy đủ có GPU: conv theo batch, FC theo n, có và không tính chép dữ liệu |
| E3 | `e3_comm` | 6 thuật toán allreduce, k = 2…10 |
| E4 | `e4_strong_scaling`, `e4b_strong_scaling_unbound` | sync SGD k = 1…10; E4b là đối chứng không ghim tiến trình |
| E5 | `e5_strategies` | 5 chiến lược × IID/không IID × k = 4, 8, 10 epoch |
| E6 | `e6_hybrid` | MPI × OpenMP với 12 luồng phần cứng |
| E7 | `e7_gpu_mpi` | 1/2/4 tiến trình dùng chung một GPU so với CPU |

**Cách các dự đoán này được ghi lại (nói rõ để không bị hiểu nhầm là đã commit trước khi chạy):**
P1–P2 nằm trong file này từ trước khi chạy (commit riêng). P3–P12 được viết trong ghi chú kế hoạch
trước đợt 2, **nhưng chỉ được đưa vào file này sau khi đã chạy xong**, nên không có commit nào chứng
minh chúng có trước số đo. Cột "đo được" điền sau khi chạy.

| # | Dự đoán | Đo được | Kết luận |
|---|---|---|---|
| P1 | GPU/CPU cho conv đạt 20–60× ở batch 256 (cuBLAS so với CPU 12 luồng OpenMP) | 13,2× (54,5× so với 1 luồng) | **sai** |
| P2 | `sync_wait` tăng rõ khi k > 6 vì E-core chậm hơn P-core | tăng từ k = 6 (0,16 → 0,77 s), giảm lại ở k = 10; không phân biệt được E-core trong WSL2 | **sai về chỗ; nguyên nhân không kiểm chứng được** |
| P3 | Tính đúng: CUDA khớp CPU, độ chính xác giữ nguyên | 105/105, 35/35 CUDA, 67/67 MPI; 98,5% | đúng |
| P4 | 1 epoch LeNet tuần tự 8–11 s (Mac: 19 s) | 7,19 s | nhanh hơn dự đoán |
| P5 | GEMM 1 luồng 25–35 GFLOP/s | 34–40 (N ≤ 1024), 25,9 (N = 4096) | đúng một phần |
| P6 | OpenMP 12 luồng nhanh hơn 1 luồng 5–8× | GEMM 5,3×; conv 4,1×; huấn luyện LeNet 2,7× | đúng với GEMM, sai với conv và huấn luyện |
| P7 | Conv: GPU nhanh hơn 1 luồng 50–150×, bài đo ≈ 40× | 54,5× (không tính chép dữ liệu); 11,2× nếu tính | đúng ở mép dưới |
| P8 | LeNet/MNIST: GPU chỉ nhanh hơn 2–5× (mô hình quá nhỏ) | không đo (E7 dùng CNN lớn hơn: 60×) | **không kiểm chứng** |
| P9 | Sync SGD: hiệu suất 85–90% ở k = 2–6, speedup 5× ở k = 6 | 90% / 73% / 52% ở k = 2 / 4 / 6; 3,12× ở k = 6 | **sai** |
| P10 | k = 8–10 phải dùng E-core, speedup chững ở 6–7× | 3,16× / 3,45× | **sai** |
| P11 | Allreduce: độ trễ ≈ 1 µs, băng thông 8–15 GB/s, ring và MPI tốt nhất cho message lớn | 0,3–1,0 µs; 14,3 GB/s (đỉnh 21,9); ring tốt nhất từ k = 4 | đúng |
| P12 | Nhiều tiến trình dùng chung một GPU: gần như không nhanh hơn 1 tiến trình | **chậm hơn** (0,54× ở np = 2, 0,40× ở np = 4) | đúng, thực tế còn tệ hơn |
