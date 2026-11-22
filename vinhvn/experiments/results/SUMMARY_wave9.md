# Đợt 9: model parallelism và mô phỏng mạng trong một pod (22/11/2026)

Phần cứng: 1 pod, 4× H100 trên một node vật lý. PyTorch 2.5.1, CUDA 12.4, FP32 với TF32 tắt. Log: `wave9/wave9.log`, CSV: `wave9/data/`.
Mọi phép đo ở đợt này là **một lần chạy** (không lặp), nên chênh lệch nhỏ (vài phần trăm) không đủ kết luận.

**Phạm vi, cần đọc trước.** Cụm chỉ có một node vật lý. Chế độ `socket` ép NCCL không dùng NVLink/shared memory (`NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_NET=Socket`),
nên mọi trao đổi giữa GPU đi qua **TCP loopback trên cùng một máy** (log NCCL xác nhận `NET/Socket` trên `eth0` của pod, `GPU Direct RDMA Disabled`; ở `native` các cặp đi `P2P/CUMEM`).
Vì vậy số liệu `socket` là **chi phí khi đi qua stack TCP thay NVLink**, không phải băng thông hay độ trễ của một mạng vật lý giữa hai máy. Mọi nhận xét về "mạng giữa các máy" ở đây là suy luận từ mô phỏng này.

## Tính đúng
Chuẩn gradient toàn cục của pipeline trùng nhau (đến 8 chữ số) với S = 1, 2, 4 và M = 1, 2; loss và chuẩn gradient của tensor parallel trùng nhau với T = 1, 2, 4. Kiểm tra trên cả CPU và GPU.

## E3 qua Socket: giao tiếp đắt hơn NVLink rất nhiều

all_reduce, một lần đo mỗi cỡ (so với NVLink từ Đợt 8, median 5 lần):

| k | busbw 1 GiB, Socket | busbw 1 GiB, NVLink | chậm hơn | t(1 GiB), Socket | t(1 GiB), NVLink |
|---|---|---|---|---|---|
| 2 | 5,57 GB/s | 332 GB/s | ≈60× | 193 ms | 3,2 ms |
| 4 | 3,07 GB/s | 356 GB/s | ≈116× | 524 ms | 4,5 ms |

- Trên Socket, t(1 GiB) ở k=4 gấp k=2 là **2,7×** (NVLink: 1,41×). Hợp ring chỉ cho 1,5×, nên Socket không theo mô hình "B không đổi theo k":
  băng thông hiệu dụng giảm khi k tăng (có thể do các luồng TCP tranh CPU và loopback của cùng máy; chưa kiểm chứng).
- Độ trễ α khớp được: 43,5 µs (k=2) và 118 µs (k=4), so với ≈ 22 µs ở NVLink (gấp 2× và 5×).
- Mô hình t = α + c·n/B **khớp kém trên Socket** (sai số khớp tối đa 33–38%, so với ≤ 7% ở NVLink): băng thông ở 1 MiB (3,2 GB/s) khác ở 1 GiB (5,6 GB/s), nên một cặp (α, B) không đủ.
  Vì vậy ngoại suy sang các mạng khác bằng cặp (α, B) này kém tin cậy. Ngoại suy đó **chưa làm**.

## E4 qua Socket: data parallel sụp đổ khi giao tiếp đắt

Hiệu suất weak scaling (ảnh/giây so với k × tốc độ k=1), Socket; `t_comm` là của chế độ manual:

| Mô hình | k | manual | ddp | t_comm (manual) | all_reduce thuần cùng kích thước (E3) |
|---|---|---|---|---|---|
| ResNet-18 | 2 | 64% | 87% | 13,3 ms | 9,8 ms |
| | 4 | 44% | 54% | 29,4 ms | 21,9 ms |
| AlexNet-xám | 4 | 10% | 12% | 127,6 ms | 119,1 ms |
| VGG-16 | 4 | 5% | 5% | 283,3 ms | 266,1 ms |

- Ở NVLink cùng cấu hình hiệu suất là 78–96%; qua Socket VGG-16 còn 5% (k=4), tức 4 GPU chậm hơn 1 GPU nhiều lần (1.706 so với 8.057 ảnh/giây).
- Khi giao tiếp chiếm gần hết bước (AlexNet, VGG-16), DDP và manual như nhau (chồng lấp không còn che được gì). Với ResNet-18, DDP tốt hơn rõ (87% so với 64% ở k=2).
- `t_comm` đo được cao hơn all_reduce thuần 6–35%, trong ngưỡng dự đoán ±40%. Phần chênh lớn hơn ở mô hình nhỏ (chi phí gộp/tách gradient chiếm tỉ lệ lớn hơn).

## E9a: pipeline parallelism (batch toàn cục: VGG-16 512, MLP 4.096)

Tăng tốc so với S=1 (MLP S=1: 508 ms/bước, VGG-16 S=1: 47,7 ms/bước). Hiệu suất = T(S=1)/(S·T). "Lý thuyết" = 1 − (S−1)/(M+S−1).

| Mô hình | S | M | tăng tốc NVLink | hiệu suất NVLink | lý thuyết | tăng tốc Socket | chậm đi do Socket |
|---|---|---|---|---|---|---|---|
| MLP | 2 | 1 | 1,00× | 50% | 50% | 0,91× | 1,10× |
| | 2 | 8 | 1,62× | 81% | 89% | 1,43× | 1,13× |
| | 4 | 1 | 1,00× | 25% | 25% | 0,74× | 1,36× |
| | 4 | 8 | 2,64× | 66% | 73% | 1,77× | 1,49× |
| | 4 | 16 | **2,66×** | 66% | 84% | 1,88× | 1,41× |
| VGG-16 | 2 | 4 | 1,29× | 65% | 80% | 1,18× | 1,10× |
| | 4 | 4 | **1,75×** | 44% | 57% | 0,77× | 2,27× |
| | 4 | 16 | 1,24× | 31% | 84% | 0,83× | 1,50× |

- **Model parallelism thuần (M=1) không tăng tốc:** 1,00× ở cả S=2 và S=4. Ủng hộ điều bài nói (§2): model parallelism không dùng để tăng tốc huấn luyện.
- Pipeline có tăng tốc (MLP S=4 tới 2,66×) nhưng **kém data parallel**: VGG-16 với batch 512, pipeline S=4 tốt nhất đạt 1,75×, trong khi data parallel 4 GPU (E4, cùng batch) đạt 2,65×.
- Tăng M không phải lúc nào cũng tốt: tăng tốc đạt đỉnh ở M=8–16 (MLP) và M=4 (VGG-16) rồi giảm, vì micro-batch nhỏ làm mỗi phép tính kém hiệu quả. Vì thế hiệu suất đo thấp hơn lý thuyết khi M lớn (cách biệt 18–19 điểm ở MLP S=4/M=16 và S=2/M=16; 53–58 điểm ở VGG-16 M=16).
- **Qua Socket, pipeline S=4 chậm hơn cả một GPU với VGG-16** (0,48–0,88×). Với S=2 ảnh hưởng nhỏ (chậm đi ≤ 16%).
- Giải thích bằng băng thông: pipeline S=4 truyền 805 MB/bước (MLP) và 170 MB/bước (VGG-16). Chia cho busbw k=4 trên Socket (3,07 GB/s) cho 262 ms (MLP, M=16: đo 270 ms) và 55 ms (VGG-16, M=4: đo 62 ms).
  Hai con số gần nhau, nhưng đây là phép ước lượng thô: dùng băng thông all_reduce thay cho send/recv điểm-điểm, và giả sử truyền không chồng lấp. Chỉ nên đọc là "nhất quán với", không phải bằng chứng.
- Bộ nhớ mỗi GPU (MLP): 9,07 GB (S=1), 5,8 GB (S=2), 3,4 GB (S=4), không đúng 1/S vì micro-batch lưu thêm activation.

## E9b: tensor parallelism (4,3 tỉ tham số, 8 khối H=8192, 4.096 token)

| Mạng | T | bước | tăng tốc | hiệu suất | lượng giao tiếp (all-reduce) | chậm đi so với NVLink | bộ nhớ mỗi GPU |
|---|---|---|---|---|---|---|---|
| NVLink | 1 | 1.999 ms | 1,00× | — | 0 | — | 35,3 GB |
| NVLink | 2 | 1.020 ms | 1,96× | 98% | 2,15 GB/bước | — | 17,8 GB |
| NVLink | 4 | 518 ms | 3,86× | 96% | 3,22 GB/bước | — | 9,1 GB |
| Socket | 2 | 1.381 ms | 1,45× | 72% | 2,15 GB/bước | 1,35× | 17,8 GB |
| Socket | 4 | 1.492 ms | 1,34× | 33% | 3,22 GB/bước | **2,88×** | 9,1 GB |

- Qua NVLink, tensor parallel gần tuyến tính (96% ở T=4) với mô hình này, vì lượng giao tiếp 3,2 GB chỉ mất vài ms ở 356 GB/s.
- Qua Socket, mô hình cộng thời gian tính toán với lượng giao tiếp chia cho băng thông khớp trong 5%: T=4: 518 + 3,22/3,07 GB/s = 1.567 ms (đo 1.492 ms); T=2: 1.020 + 2,15/5,57 = 1.406 ms (đo 1.381 ms).
  (Dùng busbw all_reduce 1 GiB làm băng thông; thông điệp thật 134 MB nên có thể sai khác; việc khớp tốt có thể là ngẫu nhiên một phần.)
- Tensor parallel nhạy với mạng hơn pipeline: cùng lượng dữ liệu, giao tiếp xảy ra mỗi lớp mỗi bước, không chồng lấp được với tính toán trong cấu hình này.

## E9c: mô hình không vừa một GPU (12,1 tỉ tham số, FP32)

| S | trạng thái | bước (M=4) | bộ nhớ đỉnh mỗi GPU (max/min) |
|---|---|---|---|
| 1 | **OOM** (ở bước backward) | — | 83,4 GB (chạm giới hạn) |
| 2 | chạy được | 427 ms | 50,9 / 48,7 GB |
| 4 | chạy được | 302 ms | 27,1 / 24,9 GB |

Đây là trường hợp duy nhất mà bài nói model parallelism có lý do tồn tại, và số đo xác nhận: 1 GPU không chứa nổi, 2 hoặc 4 GPU chứa được. Bộ nhớ mỗi GPU ≈ tổng/S (48,4 và 24,2 GB lý thuyết).

## Đối chiếu dự đoán đã ghi trước (`PLAN.md`)

| # | Dự đoán | Kết quả |
|---|---|---|
| 1 | log NCCL: `native` đi P2P, `socket` đi `NET/Socket` | **Đúng** |
| 2 | busbw Socket thấp hơn NVLink ≥ 10×; α cao hơn ≥ 5× | busbw **đúng** (60–116×). α: k=2 chỉ 2× (**sai**), k=4 đúng 5,0× (sát ngưỡng) |
| 3 | ResNet-18 k=4 manual 50–80%; VGG-16 < 40%; mô hình dự đoán `t_comm` trong ±40% | ResNet-18: 44% (**thấp hơn dự đoán**); VGG-16: 5% (**đúng**, thấp hơn nhiều); `t_comm` trong ±40% **đúng** (dùng đo trực tiếp của E3, không dùng cặp (α, B) vì khớp kém) |
| 4 | S=4/M=1 không nhanh hơn 10%; MLP S=4/M=16 tăng tốc 1,5–3×; hiệu suất trong 15 điểm của lý thuyết; pipeline kém DP | M=1: 1,00× **đúng**; 2,66× **đúng**; trong 15 điểm: **sai** ở M lớn (cách 18–19 điểm); kém DP: **đúng** |
| 5 | pipeline qua Socket chậm đi < 25% | **đúng cho S=2** (≤ 16%), **sai cho S=4** (36–51% với MLP, 50–127% với VGG-16 ở M nhỏ) |
| 6 | TP native T=4 ≥ 70%; Socket chậm hơn 1,8–3× | native 96% **đúng**; Socket T=4 2,88× **đúng** (sát ngưỡng trên), T=2 chỉ 1,35× (ngoài khoảng) |
| 7 | S=1 OOM; S=2, 4 chạy được; bộ nhớ ≈ tổng/S ±20% | **Đúng** (lệch 5% và 12%) |
| 8 | mạng giữa hai pod ≥ 0,5× Socket trong pod | thuộc Đợt 10 |

## Giới hạn
1. **Một node**: số liệu `socket` không phản ánh mạng vật lý; không có độ trễ truyền thật, không có switch.
2. **Một lần chạy** mỗi cấu hình; Đợt 8 cho thấy dao động giữa các lần có thể lên vài điểm phần trăm.
3. Hiệu suất pipeline thấp ở M lớn có thể do micro-batch nhỏ chứ không do bong bóng; chưa tách riêng.
4. Ước lượng "pipeline chạm trần băng thông Socket" và "TP = tính toán + giao tiếp" là các phép khớp thô, chưa kiểm chứng bằng đo trực tiếp băng thông điểm-điểm.
5. VGG-16 chia cho pipeline là 40 lớp chia theo MAC; mất cân bằng giữa các stage chưa được đo riêng.
