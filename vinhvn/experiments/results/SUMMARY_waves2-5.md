# Đợt 2–5: kết quả (21/11/2026)

Phần cứng: 1 node, 4× H100 80GB HBM3, PyTorch 2.5.1, CUDA 12.4, FP32 với TF32 tắt trừ khi ghi rõ.
Log gốc: `wave2/wave2.log` … `wave5/wave5.log`; CSV/JSON tách ra trong `wave*/data/` (`tools/extract_csv.py`).
Số lần lặp: E3 và E4 mỗi cấu hình một lần chạy (E4 lấy median 40 bước); E5 hai seed; E6 một lần chạy mỗi cấu hình.
Chưa có khoảng tin cậy giữa các lần chạy lặp lại, nên chênh lệch dưới vài phần trăm không đủ kết luận.

## E3: chi phí giao tiếp trong node (Đợt 2)

| k | all_reduce 4 KiB | all_reduce 1 GiB | busbw ở 1 GiB |
|---|---|---|---|
| 2 | 21,9 µs | 3,2 ms | 332 GB/s |
| 3 | 43,2 µs | 4,1 ms | 348 GB/s |
| 4 | 24,5 µs | 4,5 ms | 356 GB/s |

Khớp t = α + c·n/B cho all_reduce: α ≈ 20–24 µs, B ≈ 310–380 GB/s (`fit_*.json`; sai số khớp tối đa có trong file).

- (a) t(1 GiB) tăng 1,4× khi k từ 2 lên 4, trong ngưỡng dự đoán ≤ 1,5×: **đúng**. Phù hợp ring all-reduce, không phù hợp O(p·log k) của bài.
- (b) busbw gần mốc nccl-tests (~360 GB/s): **đúng**.
- (c) α cỡ chục µs: **đúng**.
- Điểm k=3 ở 4 KiB (43 µs) cao gần gấp đôi k=2 và k=4; chưa giải thích, một lần đo.
- Phạm vi: băng thông trong node. Không suy ra được chi phí qua mạng giữa các máy.
- Lỗi NCCL (`ncclSystemError`) lúc dọn dẹp tiến trình chỉ xuất hiện trong log của `wave2-quick`, sau khi kết quả đã ghi xong.
  Bản đầy đủ của Đợt 2 và các đợt sau không có lỗi này. (Bản đầu của tổng hợp này ghi nhầm là lỗi nằm ở bản đầy đủ; đã sửa.)

## E7: VGG-16 BF16 chậm hơn FP32 (Đợt 2)

Ảnh/giây của bước huấn luyện, batch 512:

| Biến thể | FP32 | BF16 |
|---|---|---|
| base (như Đợt 1) | 10.530 | **7.383** |
| no_avgpool | 12.799 | 64.084 |
| channels_last | 10.085 | 62.784 |
| pure_bf16 (không autocast) | | 7.577 |
| ResNet-18 (đối chứng) | 6.921 | 23.181 |

- Điểm bất thường **lặp lại**: ở `base`, BF16 chậm hơn FP32 khoảng 1,4× (batch 512), 1,4× (batch 128).
  Con số "khoảng 2 lần" của Đợt 1 không tái hiện đúng ở đây; chưa so cùng batch với Đợt 1.
- Kernel đứng đầu ở BF16 là `atomic_adaptive_average_gradinput<BFloat16>`; backward 63,9 ms so với 30,2 ms ở FP32.
- H1 (avgpool backward): bỏ AdaptiveAvgPool2d khiến BF16 nhanh hơn FP32 5×. **Có bằng chứng ủng hộ.**
- H2 (layout): `channels_last` cũng khôi phục tốc độ. Hai cách sửa đều hiệu quả nên **chưa tách được** hai giả thuyết.
  Cách giải thích khả dĩ là `channels_last` đổi sang kernel avgpool khác, nhưng chưa profile kernel của biến thể này.
- Đối chứng ResNet-18 không chậm ở BF16: phương pháp đo không sai kiểu hệ thống.
- `pure_bf16` chậm như autocast, loại trừ chi phí cast.

## E4: scaling đồng bộ (Đợt 3)

Hiệu suất weak scaling ở k=4 (ảnh/giây so với k × tốc độ k=1 không giao tiếp):

| Mô hình | p | manual (all-reduce sau backward) | ddp |
|---|---|---|---|
| ResNet-18 | 11M | 96% | 92% |
| AlexNet-xám | 58M | 86% | 89% |
| VGG-16 | 134M | 78% | 85% |

Strong scaling (batch tổng 512) tăng tốc ở k=4: ResNet-18 3,08×, VGG-16 2,65×.

- VGG-16 (p lớn) mất nhiều nhất: **đúng dự đoán**.
- manual kém ddp: **đúng với AlexNet và VGG-16, sai với ResNet-18** (manual 96% so với ddp 92%).
  Nguyên nhân chưa kiểm chứng; giả thuyết là chi phí bucket của DDP vượt lợi ích chồng lấp khi giao tiếp chỉ ~1 ms.
- Đối chiếu với E3: t_comm đo ở manual cao hơn all_reduce thuần ở cùng kích thước. Chi tiết và số đo lặp lại ở `SUMMARY_waves6-8.md`
  (VGG-16: 3,9 ms so với 2,32 ms đo trực tiếp trong E3). Đoạn ước lượng thô ở bản đầu tổng hợp này đã được thay bằng số đo trực tiếp.
- Dữ liệu tổng hợp trên GPU: không đo chi phí nạp dữ liệu.

## E5: hội tụ (Đợt 4)

Độ chính xác test sau 20 epoch, ResNet-18/CIFAR-10, k=4, 2 seed:

| Thuật toán | shuffled | sorted theo nhãn | giao tiếp |
|---|---|---|---|
| sync | 0,848 / 0,848 | 0,513 / 0,553 | 86.711 MB |
| avg_epoch | 0,842 / 0,844 | 0,588 / 0,602 | 895 MB |
| avg50 | 0,848 / 0,841 | 0,612 / 0,616 | 1.700 MB |
| oneshot | 0,131 / 0,204 | 0,100 / 0,100 | 0 MB |

- ParallelSGD (avg_epoch) ngang sync trên dữ liệu shuffled mà giao tiếp ít hơn khoảng 97×: **ủng hộ** đề xuất của bài.
- oneshot hỏng ở cả hai bố cục: **đúng dự đoán** (kể cả trên shuffled).
- Shuffle cần cho các thuật toán lấy trung bình: **đúng**, độ chính xác giảm 23–25 điểm khi dữ liệu sắp theo nhãn.
- **Dự đoán sai:** tôi dự đoán sync trên dữ liệu sorted vẫn ổn vì gradient gộp mỗi bước. Thực tế sync là tệ nhất ở bố cục sorted
  (0,51–0,55), thấp hơn avg50. Giả thuyết: BatchNorm tính thống kê trên batch chỉ gồm 2–3 lớp khiến train và test lệch.
  **Chưa kiểm chứng**; muốn tách cần chạy lại với GroupNorm hoặc với batch trộn lớp.
- Thời gian thực gần như bằng nhau (51–53 giây) vì giao tiếp trong node rất rẻ; lợi thế của lấy trung bình ít ở đây là giảm byte, không giảm thời gian.
- Số seed ít (2): chênh lệch giữa avg_epoch, avg50 và sync trên shuffled nằm trong nhiễu seed.

## E6: Downpour SGD (Đợt 5)

1600 cập nhật tổng, batch 128 mỗi worker, server trên CPU. W > 4 dùng chung GPU.

| Mô hình | W | cập nhật/giây | T (ms) | τ (ms) | T/τ | staleness TB | acc |
|---|---|---|---|---|---|---|---|
| ResNet-18 | 1 | 17,9 | 47 | 3,6 | 13,2 | 0 | 0,654 |
| | 2 | 36,0 | 46 | 3,6 | 13,0 | 1,0 | 0,657 |
| | 4 | 69,6 | 48 | 3,3 | 14,6 | 3,0 | 0,639 |
| | 8 | 96,4 | 62 | 4,7 | 13,4 | 6,9 | 0,590 |
| VGG-16 | 1 | 2,9 | 260 | 24,3 | 10,7 | 0 | 0,550 |
| | 2 | 5,0 | 285 | 24,7 | 11,5 | 0,9 | 0,488 |
| | 4 | 10,0 | 299 | 27,1 | 11,0 | 3,0 | 0,407 |
| | 8 | 17,8 | 327 | 30,4 | 10,7 | 6,9 | 0,213 |

- Staleness trung bình ≈ W − 1: **đúng dự đoán**.
- Độ chính xác giảm khi W tăng ở cùng số cập nhật tổng: **đúng**, rõ nhất ở VGG-16 (0,55 → 0,21).
- ResNet-18 gần tuyến tính đến W=4 (3,9×) rồi chững ở W=8 (5,4×): **đúng hướng**, nhưng ở W=8 T tăng 47 → 62 ms vì chung GPU, nên không thuần.
- **Dự đoán sai:** tôi dự đoán VGG-16 bão hoà ngay từ W=1–2. Thực tế VGG-16 scale gần tuyến tính (W=8 đạt 6,1×).
  Lý do là T (260 ms, gồm cả chép gradient 537 MB về CPU) lớn hơn nhiều so với τ (24 ms), nên P = T/τ ≈ 11, không phải < 2.
- **Công thức P = T/τ của bài chưa được thử ở điểm bão hoà**: W lớn nhất là 8, nhỏ hơn P ≈ 11–13 ở cả hai mô hình.
  Số đo chỉ cho thấy chưa bão hoà trong vùng đã thử, không xác nhận hay bác bỏ công thức.
- τ chỉ gồm thời gian áp cập nhật, không gồm xếp hàng; ngoài ra T gồm cả chép D2H, nên T/τ là ước lượng thô.

## Việc còn mở (bốn việc dưới đây đã được xử lý ở `SUMMARY_waves6-8.md`)
1. Tách H1 và H2 của E7 bằng profile kernel của biến thể `channels_last`.
2. Kiểm chứng giả thuyết BatchNorm cho kết quả sync trên dữ liệu sorted (E5).
3. Chạy E6 với W = 12–16 (hoặc làm chậm server có kiểm soát) để thử thật điểm bão hoà P = T/τ.
4. Lặp lại E3 và E4 nhiều lần để có khoảng tin cậy; xem lại điểm k=3 ở 4 KiB.
