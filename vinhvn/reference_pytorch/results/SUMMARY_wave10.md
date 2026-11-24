# Đợt 10: hai pod ghép thành job 2 node (22/11/2026)

Phần cứng: 2 pod × 2 GPU H100, ghép thành job torchrun 2 node (rendezvous tĩnh qua Service headless). PyTorch 2.5.1, NCCL 2.21.5, FP32, TF32 tắt.
Log: `wave10/wave10-final-pod0.log` (kết quả) và `-pod1.log`; log chẩn đoán trong `wave10/diag1-*`, `diag2-*`; bản quick đạt `wave10/quick-ok/`.
Mọi phép đo là **một lần chạy**.

## Phạm vi, cần đọc trước
**Hai pod nằm trên cùng một node vật lý** (`fke-gpu-container-host-…-m7cxc`, ghi trong `MN:` của cả hai log), vì cụm chỉ có một node. Vì vậy đường giữa hai pod đi qua mạng container (CNI) **của cùng một máy**:
không có card mạng, switch hay độ trễ vật lý. NCCL báo `GPU Direct RDMA Disabled for HCA 0 'eth0'` và dùng `NET/Socket` trên `eth0` của pod. Kết quả đo **chi phí khi đi qua mạng container/TCP**,
không phải của một mạng thật giữa hai máy. Cùng mã chạy được trên cụm đa node thật (`SPREAD=1 tools/submit_wave.sh 10`) nhưng chưa có dịp kiểm tra.

Hai chế độ: `native` (NCCL tự chọn: NVLink trong pod, Socket giữa pod) và `socket` (ép mọi cặp đi Socket, kể cả trong pod). Kích thước world: 4 rank (2 pod × 2 GPU) và 2 rank (mỗi pod 1 GPU, mọi bước nhảy đều qua pod khác).

## Việc đã chạy xong và việc không chạy được

| Cấu hình | E3 | E4 | Pipeline | Tensor parallel |
|---|---|---|---|---|
| native, 4 rank | **dừng ở 64 MiB** (treo) | xong | xong | xong |
| native, 2 rank | xong | xong | xong | xong |
| socket, 4 rank | xong | xong | xong | xong |
| socket, 2 rank | xong | xong | xong | xong |

Job kết thúc với `rc=1` vì đúng một bước hỏng: E3 `native` 4 rank chạy tốt đến 44,7 MB rồi treo ở phép all-reduce 64 MiB (sau 1.615 collective), bị timeout NCCL 180 giây bắt.
Các kích thước ≥ 64 MiB của cấu hình này **không có số đo**. 15 trong 16 bước còn lại có đủ kết quả.

## Sự cố và cách xử lý trước khi chạy được (cần biết để đọc đúng số liệu)
Lần chạy đầu của Đợt 10 treo nhiều chỗ. Tôi chẩn đoán bằng một job ma trận (`diag2-matrix-*.log`), mỗi biến thể đổi một yếu tố:

| Biến thể (native, qua 2 pod, trừ khi ghi khác) | Kết quả |
|---|---|
| Ép toàn bộ qua Socket, 4 rank | DDP chạy tốt |
| DDP gộp một bucket (1000 MB) | DDP chạy tốt |
| DDP bỏ broadcast buffer BatchNorm | vẫn treo |
| 1 GPU mỗi pod | treo, **do lỗi của tôi** (xem dưới) |
| Pipeline, chờ 5 giây trước khi huỷ nhóm | vẫn treo lúc thoát |

1. **Lỗi `barrier` của tôi (đã sửa).** Khi mỗi pod chỉ chạy 1 rank, NCCL tự đoán GPU của `barrier` theo `rank % số GPU`, sai với rank toàn cục 1 (GPU cục bộ 0); hai rank tạo communicator khác khoá nên chờ nhau mãi.
   Sửa: `barrier(device_ids=[GPU hiện tại])` trong `common/dist.py`. Các đợt 2–9 không bị ảnh hưởng vì chạy trong một pod, nơi rank trùng chỉ số GPU.
2. **DDP ở `native` qua hai pod treo khi có nhiều bucket.** Gộp một bucket thì chạy. Cơ chế **chưa xác định** (nghi ngờ liên quan đến nhiều all-reduce chồng lấp trên một vòng NCCL trộn NVLink với Socket, nhưng chưa có bằng chứng).
   Cách xử lý: ở chế độ `native` dùng DDP một bucket (`DDP_BUCKET_MB=1000`). **Hệ quả: DDP `native` qua hai pod không còn chồng lấp giao tiếp với tính toán, nên số DDP của cấu hình này không so trực tiếp được với DDP mặc định ở các đợt trước.**
3. **Pipeline làm xong việc nhưng tiến trình không thoát.** Cách xử lý: thoát thẳng (`os._exit(0)`) sau khi đã đồng bộ và ghi kết quả (`FAST_EXIT=1`). Sau đó mọi bước pipeline và tensor parallel kết thúc bình thường. Cơ chế treo **chưa xác định**.
4. `py-spy` không dùng được trong container (không có quyền theo dõi tiến trình), nên không có stack Python của các tiến trình treo.

## E3: băng thông all-reduce qua hai pod (busbw, GB/s)

| Cấu hình | 1 MiB | 16 MiB | 44,7 MB | 64 MiB | 1 GiB | α (đo 4 KiB) |
|---|---|---|---|---|---|---|
| native, 4 rank | 1,55 | 3,64 | 3,62 | treo | — | 124 µs |
| native, 2 rank | 2,05 | 3,22 | 3,46 | 4,48 | **4,42** | 50 µs |
| socket, 4 rank | 1,30 | 2,61 | 2,61 | 2,60 | **2,61** | 152 µs |
| socket, 2 rank | 2,26 | 3,38 | 3,41 | 3,43 | **4,24** | 56 µs |
| *Đợt 9: Socket trong pod, k=2* | | | 4,54 | | *5,57* | *34 µs* |
| *Đợt 9: Socket trong pod, k=4* | | | 3,06 | | *3,07* | *100 µs* |
| *NVLink (Đợt 8), k=2 / k=4* | | | | | *332 / 356* | *≈22 µs* |

- **Dự đoán #8 đúng:** băng thông giữa hai pod ≥ 0,5× của Socket trong một pod: 4,42/5,57 = 0,79 (native 2 rank), 4,24/5,57 = 0,76 (socket 2 rank), 2,61/3,07 = 0,85 (socket 4 rank). Mạng container thêm khoảng 15–25% chi phí băng thông so với TCP loopback. Độ trễ đo ở 4 KiB cao hơn nhiều hơn: 50–56 µs (2 rank) so với 34 µs, và 152 µs (4 rank) so với 100 µs, tức cao hơn khoảng 47–65% (cột α ở trên là độ trễ đo trực tiếp, không phải α khớp).
- Ở 4 rank, `native` (còn dùng NVLink trong pod) nhanh hơn `socket` (3,62 so với 2,61 GB/s ở 44,7 MB), nhưng vẫn kém NVLink thuần khoảng 100×.
- Cả hai pod chung một máy: **không suy ra** được băng thông của mạng vật lý.

## E4: data parallel qua hai pod (hiệu suất weak scaling so với 1 GPU, Đợt 9)

| Mô hình | k | native, manual | native, ddp (1 bucket) | socket, manual | socket, ddp | *Đợt 9: Socket trong pod, manual / ddp* |
|---|---|---|---|---|---|---|
| ResNet-18 | 2 | 58% | 60% | 58% | 77% | 64% / 87% |
| | 4 | 51% | 51% | 40% | 48% | 44% / 54% |
| AlexNet-xám | 4 | 16% | 16% | 9% | 10% | 10% / 12% |
| VGG-16 | 4 | 8% | 9% | 4% | 5% | 5% / 5% |

- Qua hai pod, hiệu suất của `socket` thấp hơn Socket trong một pod 1–10 điểm (ResNet-18: 77% so với 87% ở k=2, 48% so với 54% ở k=4; AlexNet và VGG-16 chỉ lệch khoảng 1 điểm vì đã gần như bị giao tiếp chi phối).
- Ở k=4, `native` (51% cho ResNet-18) tốt hơn `socket` (40–48%) vì các cặp trong cùng pod vẫn dùng NVLink.
- `ddp ≈ manual` ở `native` vì DDP chỉ có một bucket, không còn chồng lấp (xem mục sự cố 2); ở `socket` DDP mặc định có chồng lấp nên tốt hơn manual (48% so với 40%).
- `t_comm` của manual cao hơn all_reduce thuần cùng kích thước 18–29% (ResNet-18, 44,7 MB: 16,7 ms so với 12,9 ms; 33,0 ms so với 25,7 ms; 21,8 so với 18,5; 16,3 so với 13,1), cùng kiểu chênh do gộp/tách gradient đã thấy ở Đợt 8.

## E9a: pipeline qua hai pod (bước ms, tăng tốc so với 1 GPU)

| Mô hình | S | M | native (2 pod) | socket (2 pod) | *Đợt 9 NVLink* | *Đợt 9 Socket trong pod* |
|---|---|---|---|---|---|---|
| MLP | 2 | 8 | 375 (1,36×) | 377 (1,35×) | 314 | 355 |
| MLP | 4 | 16 | 242 (2,10×) | 285 (1,78×) | 191 | 270 |
| VGG-16 | 2 | 4 | 43,6 (1,09×) | 43,4 (1,10×) | 36,9 | 40,6 |
| VGG-16 | 4 | 4 | 32,9 (1,45×) | 61,0 (0,78×) | 27,3 | 62,0 |

- S=2: mọi bước nhảy đều qua pod khác, nên `native` và `socket` như nhau (375 so với 377 ms) và chậm hơn Socket trong một pod 6% (MLP) đến 7% (VGG-16).
- S=4 `native`: chỉ một ranh giới stage (stage 1→2) nằm giữa hai pod, còn các ranh giới khác dùng NVLink. Chậm hơn NVLink thuần 27% (MLP, M=16: 242 so với 191 ms) và 20% (VGG-16, M=4), nhưng nhanh hơn nhiều so với toàn Socket (242 so với 285 ms; 32,9 so với 61,0 ms).
- Với VGG-16 ở S=4, `native` qua hai pod (1,45×) thắng 1 GPU, còn toàn Socket (0,78×) thua 1 GPU.

## E9b: tensor parallel qua hai pod (4,3 tỉ tham số, 1 GPU: 1.999 ms)

| T | native (2 pod) | socket (2 pod) | *Đợt 9 NVLink* | *Đợt 9 Socket trong pod* | giao tiếp |
|---|---|---|---|---|---|
| 2 | 1.510 ms (1,32×, 66%) | 1.514 ms (1,32×, 66%) | 1.020 | 1.381 | 2,15 GB/bước |
| 4 | 1.020 ms (1,96×, 49%) | 1.656 ms (1,21×, 30%) | 518 | 1.492 | 3,22 GB/bước |

- Mô hình "bước = tính toán + lượng giao tiếp / băng thông" (tính toán lấy từ NVLink, băng thông từ busbw 1 GiB của E3 cùng cấu hình) khớp với số đo ở hai trường hợp:
  T=2: 1.020 + 2,15/4,42×1000 = 1.506 ms (đo 1.510); T=4 `socket`: 518 + 3,22/2,61×1000 = 1.752 ms (đo 1.656, lệch 6%).
- **Mô hình không áp dụng được cho T=4 `native`:** dùng busbw đo ở 44,7 MB (3,62 GB/s) cho 1.407 ms, đo thực tế 1.020 ms. Không kiểm tra được ở kích thước thật (134 MB) vì E3 `native` 4 rank treo ở 64 MiB.
- Tensor parallel nhạy với mạng nhất: T=4 `native` qua hai pod đạt 49% so với 96% khi chỉ dùng NVLink.

## Đối chiếu dự đoán đã ghi trước
Chỉ dự đoán #8 trong `PLAN.md` thuộc Đợt 10: **đúng** (tỉ lệ 0,76–0,85, ngưỡng ≥ 0,5). Mọi nhận xét còn lại ở trên là **quan sát sau khi có số đo**, không có dự đoán ghi trước.

## Giới hạn
1. Hai pod cùng một node: đây là mô phỏng chi phí mạng container, không phải mạng giữa hai máy.
2. Một lần chạy cho mỗi cấu hình; Đợt 8 cho thấy dao động giữa các thời điểm có thể vài điểm phần trăm.
3. E3 `native` 4 rank thiếu dữ liệu từ 64 MiB trở lên. Số `native` 4 rank chỉ đúng đến 44,7 MB.
4. DDP `native` dùng một bucket (không chồng lấp); chỉ so sánh `native` với `socket` bằng DDP mặc định ở cột `socket`.
5. Cơ chế treo ở `native` qua hai pod (DDP nhiều bucket, E3 ≥ 64 MiB, thoát tiến trình) **chưa xác định**; ba cách xử lý (`1 bucket`, `FAST_EXIT`, ép Socket) là cách tránh, không phải sửa nguyên nhân.
6. Chưa đo trên cụm đa node thật.
