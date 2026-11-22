# Dự đoán trước khi đo băng thông (kiểm chứng giai đoạn 5)

Viết ngày 22/11/2026, **trước** mọi lần chạy `code/bench_mem.py` (kể cả chạy thử trên laptop). Các phép đo này tách khỏi thực nghiệm của bài báo;
mục đích là kiểm chứng các giả thuyết ghi trong [ANALYSIS.md](ANALYSIS.md) mục 7. Kết quả và nhận xét sẽ ghi ở cột cuối **sau khi chạy**, giữ nguyên cột dự đoán.

Các phép đo (mỗi điểm 3 lần khởi động, 10 lần tính giờ, 3 lần chạy trên Kaggle): băng thông bằng `sum`, `dot`, `copy`, `triad` (độc lập với matvec);
`(n×n)·(n×B)`; tách `im2col` và `GEMM` trong conv kiểu Caffe.

| Ký hiệu | Dự đoán (trước khi chạy) | Nếu sai thì nghĩa là | Kết quả |
|---|---|---|---|
| D1 | Kaggle, CPU 1 luồng, n ≥ 4096: băng thông đọc tốt nhất (max của `sum`, `dot`) khoảng 12–16 GB/s, và matvec (12,3–13,5 GB/s) đạt ≥ 75% con số đó | Nếu matvec < 60%: matvec trên CPU **không** bị giới hạn bởi băng thông bộ nhớ | **Đúng.** 14,3 / 13,5 / 13,6 GB/s (n = 4096 / 8192 / 16384); matvec 14,2 / 13,4 / 12,5 (100% / 100% / 92%) |
| D2 | Kaggle, GPU, n ≥ 8192: `sum`/`dot`/`triad` đạt 240–300 GB/s và matvec (262–268 GB/s) lệch không quá 15% so với tốt nhất trong số đó | Nếu `sum`/`dot` cao hơn hẳn (> 300 GB/s hoặc matvec < 70%): matvec GPU không chạm trần bộ nhớ | **Gần như đúng.** `sum` 269 / 272, `dot` 260 / 253, matvec 262 / 268 (97–99% của tốt nhất). Sai nhẹ: `triad` ở n = 8192 là 237,6 (dưới ngưỡng 240), 246,0 ở n = 16384 |
| D3 | Kaggle, CPU: băng thông đọc tốt nhất của 2 luồng gấp 1,5–2 lần 1 luồng ở n ≥ 4096 (giống matvec 2 luồng = 1,91×) | Nếu ≈ 1×: một luồng đã lấp đầy băng thông bộ nhớ | **Đúng.** 1,90 / 1,92 / 1,92 lần (n = 4096 / 8192 / 16384) |
| D4 | Kaggle, `(n×n)·(n×64)`: tỷ lệ CPU-1 luồng / GPU khoảng 22–28× ở n = 8192 (gần tỷ lệ GEMM 24×), **không** đạt 40–57× của bài báo | Nếu đạt ≥ 40×: giả thuyết "FC của bài báo là nhân nhiều vector" giải thích được cả tỷ lệ | **Sai.** Đo 35,5× (35,0–36,0) ở B = 64, cao hơn dự đoán; chưa đạt 40 ở B = 64. Ở B = 128: 45× (34,6–51,1; lần 2 bị hạ xung xuống 742 MHz). Giả thuyết phù hợp nhưng không được xác nhận (xem ANALYSIS mục 3) |
| D5 | Conv kiểu Caffe trên GPU (batch 64 và 256): `im2col` chiếm ≥ 40% thời gian mỗi ảnh (GEMM ước tính khoảng 20 µs trong khoảng 65 µs) | Nếu `im2col` < 20%: nguyên nhân conv chỉ đạt 14% trần không nằm ở im2col | **Đúng** (43% ở batch 256, 46% tính theo chu kỳ). Ước tính GEMM 20 µs **sai**: đo được 39 µs/ảnh (26% trần FP32). Batch 64 bị hạ xung nên không dùng |
| D6 | Laptop (nếu chạy), CPU 1 luồng, n ≥ 4096: băng thông đọc tốt nhất 28–40 GB/s, matvec (28 GB/s) ≥ 70% con số đó | Nếu matvec < 60%: như D1 | Chưa chạy |

**Ghi chú minh bạch:** sau khi viết bảng này, code đã được chạy thử trên laptop để kiểm tra (chỉ xem kết quả ở n ≤ 1024; lưới đầy đủ chạy một lần nhưng không xem số ở n ≥ 4096). Các cột dự đoán không sửa sau đó.

Không dự đoán: băng thông của bài báo (không có số liệu), và hành vi 8 luồng của matvec trên laptop.

**Nguồn kết quả:** Kaggle, 22/11/2026 (UTC 09:53), 3 lần chạy, 10 lần tính giờ mỗi điểm; dữ liệu `results/kaggle/bw{1,2,3}_20261122_095300`, bảng số `results/analysis/mem_tables.md`. Dự đoán được viết và chạy code trước khi thấy kết quả này, nhưng **chưa commit vào git trước khi chạy**, nên git không chứng minh thứ tự thời gian.
