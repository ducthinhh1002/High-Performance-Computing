# Đợt 6–8: kiểm chứng bổ sung (22/11/2026)

Cùng phần cứng và phần mềm với Đợt 2–5 (4× H100, PyTorch 2.5.1, FP32 TF32 tắt). Dự đoán được ghi trong `PLAN.md` **trước** khi chạy.
Log: `wave6/wave6.log`, `wave7/wave7.log`, `wave8/wave8.log`; CSV trong `wave*/data/`.
Số lần lặp: E7 và E6 một lần; E5b 3 seed; E3 và E4 5 lần lặp (báo median và khoảng min–max giữa các lần).

## E7: kernel nào làm VGG-16 BF16 chậm (Đợt 6a)

Self CUDA time mỗi bước, batch 256:

| Biến thể | Kernel tốn nhất | Thời gian |
|---|---|---|
| base, FP32 | `implicit_convolve_sgemm` | 9,0 ms |
| base, BF16 | `atomic_adaptive_average_gradinput<BFloat16>` | **29,4 ms** |
| no_avgpool, BF16 | `nchwToNhwcKernel` | 0,78 ms |
| channels_last, BF16 | `multi_tensor_apply_kernel` (optimizer) | 1,17 ms |

- **Đã chứng minh trong phạm vi thí nghiệm:** nguyên nhân điểm bất thường là kernel backward của AdaptiveAvgPool2d (nhánh atomic add trên BF16).
  Kernel này tốn 29,4 ms, gấp hơn 3 lần kernel tốn nhất ở FP32, và **không còn trong top kernel** khi bỏ avgpool hoặc dùng `channels_last`.
- `channels_last` sửa được vì nó chuyển avgpool sang kernel khác, không phải vì layout làm conv nhanh hơn. Dự đoán của tôi ("kernel atomic biến mất") **đúng**.
- H2 (phép chuyển vị layout NCHW↔NHWC làm chậm) **bị bác bỏ làm cơ chế**: kernel chuyển vị chỉ tốn 0,78 ms.
- Chưa làm: kiểm tra phiên bản PyTorch khác, hay GPU khác, có còn kernel này không.

## E6: Downpour, phép thử điểm bão hoà (Đợt 6b)

ResNet-18, 1600 cập nhật tổng. `extra` là độ trễ thêm vào mỗi lần server áp cập nhật (cộng thẳng vào τ). T là thời gian worker tính một gradient.

| extra | W | cập nhật/giây | T (ms) | τ (ms) | 1/τ | staleness | acc |
|---|---|---|---|---|---|---|---|
| 30 ms | 1 | 11,2 | 49 | 33,3 | 30 | 0 | 0,656 |
| | 2 | 23,2 | 44 | 32,9 | 30 | 1,0 | 0,646 |
| | 4 | 29,5 | 47 | 33,6 | 30 | 3,0 | 0,639 |
| | 8 | 29,4 | 47 | 33,7 | 30 | 7,0 | 0,533 |
| | 16 | 29,2 | 47 | 33,7 | 30 | 14,9 | 0,345 |
| 10 ms | 1 | 15,5 | 43 | 14,0 | 72 | 0 | 0,653 |
| | 2 | 30,0 | 47 | 12,8 | 78 | 1,0 | 0,639 |
| | 4 | 57,6 | 48 | 13,4 | 74 | 3,0 | 0,631 |
| | 8 | 71,8 | 49 | 13,5 | 74 | 7,0 | 0,538 |
| | 16 | 72,8 | 48 | 13,2 | 76 | 14,9 | 0,334 |

- **Bão hoà đúng ở 1/τ:** với trễ 30 ms, tốc độ bão hoà 29,4 cập nhật/giây = 98% của 1/τ (≈ 29,8); với 10 ms, 71–73 = 96–98% của 1/τ (≈ 74).
  Dự đoán của tôi (trong ±15% của 1/τ) **đúng**. Lưu ý: vì server tuần tự và ngủ cố định, mức trần này gần như đúng theo cấu tạo; không phải bằng chứng độc lập.
- **Vị trí knee là phép thử có nghĩa.** Công thức của bài: bão hoà khi W ≥ P = T/τ, tức P ≈ 1,4 (extra 30) và ≈ 3,6 (extra 10).
  Số đo: ở extra 30, W=2 mới đạt 79% mức bão hoà (23,2/29,4), dù P ≈ 1,4 dự đoán đã bão hoà; ở extra 10, W=4 mới đạt 77–80% (57,6/72), dù P ≈ 3,6 dự đoán đã bão hoà.
  Số đo khớp tốt hơn với (T+τ)/τ = P + 1 (2,4 và 4,6): W=2 và W=4 nằm dưới knee, W=4 (extra 30) và W=8 (extra 10) đã bão hoà.
  Khác biệt này do thiết kế của tôi: worker chờ server áp xong mới tính tiếp, nên chu kỳ mỗi worker là T + τ. Vì vậy đây **không phải bác bỏ bài**, mà là công thức cho một biến thể có worker chờ. Lưới W thô (1, 2, 4, 8, 12, 16) nên không xác định được knee chính xác hơn "giữa 2 và 4" và "giữa 4 và 8".
- **Thêm worker sau knee không tăng tốc mà làm giảm độ chính xác:** extra 10, W=4 → 16: 0,631 → 0,334 với cùng số cập nhật và cùng tốc độ.
- **Không làm chậm server (extra 0):** ResNet-18 đạt đỉnh ở W=8 (96,4) rồi giảm (W=12: 89,8; W=16: 82,3). T tăng 47 → 117 ms và τ tăng 3,6 → 9,9 ms vì chung GPU và tranh CPU,
  nên T/τ không còn đầu vào cố định; thí nghiệm này không so sạch với công thức được. VGG-16 bão hoà quanh 19,8 cập nhật/giây từ W=12 (bằng 77–95% của 1/τ).
- **VGG-16 phân kỳ khi staleness lớn:** W=12 đạt acc 0,100 (bằng đoán ngẫu nhiên), W=16 đạt 0,162, ở learning rate 0,02. Nguyên nhân phân kỳ chưa được kiểm chứng (chưa thử learning rate nhỏ hơn).

## E5b: BatchNorm so với GroupNorm (Đợt 7)

Độ chính xác test sau 20 epoch, mean ± độ lệch chuẩn của 3 seed:

| Thuật toán | BN shuffled | BN sorted | GN shuffled | GN sorted |
|---|---|---|---|---|
| sync | 0,848 ± 0,001 | 0,531 ± 0,021 | 0,790 ± 0,014 | **0,792 ± 0,025** |
| avg_epoch | 0,844 ± 0,001 | 0,591 ± 0,010 | 0,664 ± 0,040 | 0,417 ± 0,060 |
| avg50 | 0,843 ± 0,004 | 0,614 ± 0,002 | 0,672 ± 0,041 | 0,432 ± 0,057 |
| oneshot | 0,160 ± 0,039 | 0,100 ± 0,000 | 0,176 ± 0,061 | 0,100 ± 0,000 |

Chênh lệch shuffled − sorted: sync BN **+0,317**, sync GN **−0,001**; avg_epoch BN +0,253, GN +0,246; avg50 BN +0,229, GN +0,240.

- **Đã chứng minh trong phạm vi thí nghiệm:** sync tệ trên dữ liệu sorted là do BatchNorm. Thay bằng GroupNorm, khoảng cách biến mất hoàn toàn (−0,001 so với +0,317).
  Dự đoán của tôi ("cải thiện nhiều nhưng không bằng shuffled") **sai về độ lớn**: GroupNorm đóng khoảng cách hoàn toàn.
- **Các thuật toán lấy trung bình vẫn chịu phạt ~0,24 trên sorted ở cả hai loại chuẩn hoá**, nên phần phạt này **không do BatchNorm**.
  Giả thuyết còn lại là trôi cục bộ của mỗi máy khi chỉ thấy 2–3 lớp (client drift); chưa kiểm chứng.
- Hạn chế: GroupNorm cho độ chính xác thấp hơn BatchNorm và phương sai giữa các seed lớn hơn (độ lệch chuẩn tới 0,06). Learning rate 0,05 chưa được chỉnh cho GroupNorm.
  Với GroupNorm, avg_epoch kém sync nhiều (0,66 so với 0,79) ngay cả trên shuffled, khác với BatchNorm (0,844 so với 0,848); chưa giải thích.
  Vì vậy chỉ so chênh lệch sorted − shuffled trong từng loại chuẩn hoá, không so độ chính xác tuyệt đối giữa BN và GN.
- Thêm seed 2 cho BatchNorm xác nhận kết quả hai seed trước (độ lệch chuẩn ≤ 0,021).

## E3 và E4 lặp 5 lần (Đợt 8)

**E3, all_reduce (median [min–max] qua 5 lần):**

| Kích thước | k=2 | k=3 | k=4 |
|---|---|---|---|
| 4 KiB | 21,6 µs [14,7–42,5] | 21,5 µs [16,7–43,1] | 31,4 µs [23,3–47,6] |
| 1 GiB | 3.229 µs [3.218–3.240] | 4.105 µs [4.089–4.144] | 4.551 µs [4.524–4.558] |

- t(1 GiB) ở k=4 gấp k=2 là **1,41×**, rất ổn định (khoảng min–max hẹp hơn 1%). Kết luận ring all-reduce vững.
- Điểm k=3 ở 4 KiB (43 µs ở Đợt 2) là **nhiễu**: median 21,5 µs, 43 µs chỉ là giá trị lớn nhất trong 5 lần. Dự đoán của tôi **đúng**. Phép đo cỡ nhỏ (4 KiB–1 MiB) dao động tới 2–3 lần giữa các lần.

**E4, hiệu suất weak scaling ở k=4 (median [min–max] qua 5 lần):**

| Mô hình | manual | ddp | t_comm (manual) | all_reduce thuần (E3, k=4) |
|---|---|---|---|---|
| ResNet-18 | 96,0% [95,8–96,6] | 92,5% [92,3–93,0] | 0,98 ms | 0,22 ms |
| AlexNet-xám | 89,5% [89,3–89,9] | 93,5% [93,0–93,8] | 1,78 ms | 1,04 ms |
| VGG-16 | 81,2% [79,2–81,8] | 88,6% [86,9–88,9] | 3,90 ms | 2,32 ms |

- **DDP kém all-reduce thủ công ở ResNet-18 lặp lại** (khoảng của hai chế độ không chồng nhau), còn ở AlexNet và VGG-16 DDP tốt hơn. Dự đoán của tôi **đúng**. Nguyên nhân (chi phí bucket của DDP khi giao tiếp chỉ ~0,2 ms) chưa kiểm chứng.
- **`t_comm` của chế độ manual luôn cao hơn all_reduce thuần ở cùng kích thước** (cao hơn 0,76 ms, 0,74 ms và 1,58 ms), lặp lại ổn định.
  Cần đọc đúng định nghĩa: `t_comm` trong E4 gồm cả gộp/tách vector gradient, chia và chép lại về từng tensor, không phải thuần giao tiếp. Ước lượng "VGG-16 lệch 60%" ở bản trước vì thế là lệch do định nghĩa đo, không phải mô hình E3 sai.
  Chưa tách từng phần để chứng minh; việc này cần đo riêng thao tác gộp/tách.
- Giữa Đợt 3 và Đợt 8, hiệu suất VGG-16 manual đổi từ 78% sang 81,2% và ddp từ 85% sang 88,6%, tức dao động theo thời gian lớn hơn khoảng min–max trong cùng một đợt (±1–2 điểm).
  Do đó chênh lệch dưới ~3 điểm giữa các lần chạy khác thời điểm không nên đọc là khác biệt thật.
- Lỗi NCCL lúc thoát: 0 lần trong 15 lần chạy E3 của Đợt 8 (và trong mọi bản đầy đủ). Lỗi từng thấy một lần, ở `wave2-quick`, trước khi sửa `cleanup()`.
  Tần suất ban đầu quá thấp (1 lần trong khoảng 9 lần chạy) nên **không kết luận được** bản sửa có tác dụng.

## Việc vẫn mở
1. Cơ chế khiến averaging chịu phạt ~0,24 trên dữ liệu sorted không do BatchNorm (client drift?).
2. Phân kỳ của VGG-16 ở W ≥ 12 trong E6: chưa thử learning rate nhỏ hơn.
3. Tách thời gian gộp/tách vector gradient khỏi giao tiếp thuần trong E4 manual.
4. Lưới W của E6 thô: chưa xác định knee chính xác (cần W = 3, 5, 6, 7).
5. Tất cả vẫn là một node; chi phí qua mạng giữa các máy chưa đo được.
