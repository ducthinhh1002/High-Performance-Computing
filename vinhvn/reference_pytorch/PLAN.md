# Kế hoạch thực nghiệm Đợt 2–5

Bám bài "Parallel and Distributed Deep Learning" (Hegde, Usmani): Đợt 1 đã kiểm chứng §3 (CPU so với GPU).
Các đợt sau kiểm chứng §5 (data parallelism) và đóng điểm bất thường còn mở của Đợt 1.
Phần cứng: 1 node, tối đa 4× H100 (quota namespace), nên "phân tán" ở đây là đa GPU trong một node.
Không đo được mạng giữa các node; kết luận về O(L·k) và băng thông mạng chỉ áp dụng cho kết nối trong node.

| Đợt | Thí nghiệm | Câu hỏi | GPU | Ước tính |
|---|---|---|---|---|
| 2 | E3 giao tiếp, E7 VGG BF16 | Mô hình chi phí của bài có khớp số đo không? Vì sao VGG-16 BF16 chậm? | 4 | ~25 phút |
| 3 | E4 scaling đồng bộ | Data parallel đồng bộ scale được bao nhiêu, giao tiếp chiếm bao nhiêu? | 4 | ~40 phút |
| 4 | E5 hội tụ | ParallelSGD/ADMM có hội tụ như SGD đồng bộ không? Shuffle có cần không? | 4 | ~30 phút |
| 5 | E6 Downpour | Bất đồng bộ bão hoà ở W = T/τ như bài nói không? Staleness ảnh hưởng gì? | 4 (+CPU) | ~25 phút |

Chạy từng đợt, tuần tự: `tools/submit_wave.sh <N> [quick]`. Chạy `quick` trước để kiểm tra Job lên được.
Mã nguồn đã chạy thử trên CPU/gloo (E3, E4, E5, E6); E7 cần GPU nên chưa chạy thử ở đâu cả.

## Đợt 2

**E3, chi phí giao tiếp** (`e3_comm/allreduce_bench.py`). Đo all_reduce/broadcast/reduce, k = 2, 3, 4, kích thước 4 KiB–1 GiB
và kích thước gradient thật của 3 mô hình; khớp t = α + c·n/B.
- Bài: thời gian gộp O(p·log k), chi phí truyền O(k·p) (BitTorrent aggregate).
- NCCL ring: dữ liệu mỗi rank truyền 2(k-1)/k·n, gần như không đổi theo k.
- Dự đoán kiểm chứng được: (a) t(1 GiB) tăng ≤ 1,5× khi k từ 2 lên 4 (ring), trái với mô hình log k của bài;
  (b) busbw tiến gần mốc nccl-tests (khoảng 360 GB/s ở 1 GiB, k=4) khi n ≥ 128 MiB; (c) α cỡ chục µs.
  Nếu t tăng theo k như O(k), mô hình bài đúng hơn; nếu phẳng, ring đúng.
- Lưu ý đo: Đợt này đo trong một node nên "băng thông B" là NVLink/PCIe, không phải mạng. Bảng topo
  của pilot cho thấy GPU nối qua PXB/NODE/SYS (không chắc có NVLink); đối chiếu với `nvidia_smi_topo` trong log.

**E7, VGG-16 BF16** (`e7_bf16/vgg_bf16.py`). Mỗi biến thể đổi đúng một yếu tố (xem docstring).
- H1: AdaptiveAvgPool2d backward chậm ở BF16. Nếu đúng: `no_avgpool` hết chậm (BF16 ≤ FP32).
- H2: layout NCHW. Nếu đúng: `channels_last` hết chậm.
- Cả hai có thể sai: khi đó top kernel (`e7_kernels.csv`) là bằng chứng để tìm nguyên nhân thật.
- Đối chứng: ResNet-18 (không có avgpool 1×1→7×7) phải không chậm lại ở BF16, nếu chậm thì lỗi nằm ở phương pháp đo.

## Đợt 3

**E4, scaling đồng bộ** (`e4_dp/scaling.py`). k = 1–4; ResNet-18 (p=11M), AlexNet-xám, VGG-16 (p=134M);
weak (128/GPU) và strong (512 tổng). Ba chế độ: local (không giao tiếp), manual (all-reduce sau backward, đúng bài), ddp (chồng lấp).
- Dự đoán: hiệu suất weak = step(local)/step(mode). VGG-16 (p lớn, tính toán/tham số thấp) mất nhiều hơn ResNet-18;
  manual kém ddp, chênh lệch ≈ phần giao tiếp được che. strong scaling bão hoà sớm khi batch mỗi GPU nhỏ.
- Đối chiếu với E3: t_comm đo ở `manual` phải khớp t(n = 4p byte) từ E3 trong vài chục phần trăm.
- Giới hạn: dữ liệu tổng hợp trên GPU, nên không đo chi phí nạp dữ liệu.

## Đợt 4

**E5, hội tụ** (`e5_sgd/convergence.py`). ResNet-18 trên CIFAR-10, k=4, 20 epoch, 2 seed.
Thuật toán: sync (= ADMM.SGD của bài), avg_epoch (ParallelSGD, một vòng = một epoch), avg50, oneshot.
Bố cục: shuffled so với sorted theo nhãn.
- Bài: chỉ ParallelSGD (lấy trung bình) cần shuffle vì giả định dữ liệu đại diện; lấy trung bình ít thì rẻ giao tiếp.
- Dự đoán: oneshot trên sorted gần như hỏng (mỗi máy chỉ thấy 2–3 lớp, trung bình tham số của mạng phi tuyến);
  trên shuffled oneshot kém sync rõ rệt (bài tự nhận strongly convex không đúng cho mạng sâu); avg_epoch gần sync nhất
  trong các phương án lấy trung bình; sync trên sorted vẫn ổn vì gradient được gộp mỗi bước.
- Bất lợi đã biết: chu kỳ lấy trung bình dài làm BatchNorm lệch (thống kê cục bộ), không chỉ tham số.
  Kết quả sorted+BN không tách được ảnh hưởng của tham số khỏi BN.
- Số liệu giao tiếp (`comm_MB`) cho thấy đánh đổi: sync ≈ steps·p, avg_epoch ≈ epochs·p.

## Đợt 5

**E6, Downpour** (`e6_async/downpour.py`). Parameter server trên CPU, W worker = 1, 2, 4, 8 (W > 4 dùng chung GPU), ResNet-18 và VGG-16, 1600 cập nhật tổng.
- Bài: tối đa 1/τ cập nhật/giây, thêm worker vô ích khi W ≥ P = T/τ.
- Dự đoán: ResNet-18 (p nhỏ) tăng gần tuyến tính đến W≈4 rồi chững; VGG-16 (τ ∝ p = 134M) bão hoà ngay từ W=1–2.
  Staleness trung bình ≈ W−1. Độ chính xác sau cùng số cập nhật giảm khi W tăng.
- Hạn chế chính: server là một process Python trên CPU chia sẻ với 4 GPU worker, W > 4 làm T phình ra do chung GPU,
  nên P_pred = T/τ lúc W=8 không thuần. τ chỉ gồm thời gian áp cập nhật (không gồm xếp hàng và sao chép gradient về CPU, tính vào T).
  Kết luận chỉ ở mức "cùng xu hướng với bài", không phải xác nhận định lượng.

## Đợt 6–8: kiểm chứng bổ sung (ghi trước khi chạy, 22/11/2026)

Sau Đợt 2–5 còn 4 điểm mở (xem `results/SUMMARY_waves2-5.md`). Dự đoán dưới đây ghi **trước** khi có số đo.

**Đợt 6a, E7 profile `channels_last`** (tách H1 và H2). Thêm top kernel của `channels_last` ở FP32 và BF16 (batch 256).
- Nếu `atomic_adaptive_average_gradinput` biến mất khỏi top kernel ở `channels_last` BF16: `channels_last` sửa được vì đổi sang kernel avgpool khác, cùng nguyên nhân H1.
- Nếu kernel đó vẫn đứng đầu nhưng bước vẫn nhanh: H1 không phải toàn bộ nguyên nhân.
- Dự đoán của tôi: kernel atomic biến mất (khoảng 70% tin).

**Đợt 6b, E6 mở rộng.** (i) ResNet-18 và VGG-16 với W = 12, 16. (ii) ResNet-18 với server bị làm chậm 10 ms và 30 ms mỗi cập nhật, W = 1…16.
- Bài: bão hoà khi W ≥ T/τ, tốc độ tối đa 1/τ.
- Thiết kế của tôi (worker chờ server áp xong mới tiếp tục) cho knee ở W ≈ (T+τ)/τ, tức hơn bài khoảng 1 worker; sai khác này nhỏ so với nhiễu.
- Dự đoán: với trễ 30 ms (τ ≈ 34 ms, T ≈ 47 ms) knee ở W ≈ 2–3 và tốc độ bão hoà trong ±15% của 1/τ (≈ 29 cập nhật/giây). Với trễ 10 ms knee ở W ≈ 4–5. Không thêm trễ: ResNet-18 gần bão hoà quanh W = 12–16 (1/τ ≈ 270/giây, nhưng chung GPU nên thực tế thấp hơn).
- Lưu ý: vì server tuần tự và ngủ cố định, "tốc độ tối đa ≤ 1/τ" gần như đúng theo cấu tạo. Phép thử có nghĩa ở **vị trí knee** và ở việc W lớn có làm T phình ra (chung GPU) hay không.

**Đợt 7, E5b: BatchNorm so với GroupNorm.** ResNet-18 GroupNorm(8), 3 seed, cùng 4 thuật toán × 2 bố cục; thêm seed 2 cho BatchNorm (đủ 3 seed).
- H_BN: sync tệ trên dữ liệu sorted là do thống kê BatchNorm tính trên batch 2–3 lớp.
- Nếu đúng: với GroupNorm, sync trên sorted tiến gần sync trên shuffled (chênh dưới 10 điểm). Nếu sai: vẫn tệ hơn rõ rệt, nguyên nhân khác (ví dụ gradient từ các shard lệch lớp, LR).
- Dự đoán: H_BN đúng một phần (sync/sorted cải thiện nhiều nhưng không bằng shuffled).
- Hạn chế: GroupNorm đổi cả khả năng mô hình, nên độ chính xác tuyệt đối không so được với BatchNorm; chỉ so chênh lệch sorted − shuffled trong từng loại norm.

**Đợt 8, lặp E3 và E4 5 lần.** Báo median và khoảng min–max giữa các lần lặp.
- Câu hỏi: điểm k=3 ở 4 KiB (43 µs) có lặp lại không; DDP kém all-reduce thủ công ở ResNet-18 (92% so với 96%) có lặp lại không; t_comm của VGG-16 có luôn cao hơn ước lượng E3 khoảng 60% không.
- Dự đoán: DDP < manual ở ResNet-18 lặp lại (khoảng 75% tin); điểm k=3 là nhiễu (khoảng 60% tin).
- Cũng kiểm tra lỗi NCCL lúc thoát tiến trình có còn không sau khi thêm `barrier` + `synchronize` vào `cleanup()` (nguyên nhân chưa kiểm chứng).

## Đợt 9–10: mạng giữa các máy và model parallelism (thiết kế ngày 22/11/2026, chưa chạy)

**Giới hạn phần cứng, cần đọc trước.** Cụm chỉ có **một node vật lý** (8× H100, namespace giới hạn 4 GPU). Không đo được mạng thật giữa hai máy
(card mạng, switch, độ trễ vật lý). Nhật ký `nvidia-smi topo` cho thấy node có các NIC, nhưng pod không được cấp thiết bị RDMA/InfiniBand
(NCCL chỉ thấy transport Socket). Vì vậy đợt này **mô phỏng đường mạng bằng stack TCP**:
- Đợt 9, `socket`: trong một pod, ép NCCL không dùng NVLink/shared memory (`NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_NET=Socket`), mọi trao đổi giữa GPU đi qua TCP loopback.
- Đợt 10: hai pod (2 GPU mỗi pod) ghép thành job torchrun 2 node; trao đổi giữa pod đi qua mạng container (CNI) của cùng một máy.
Cả hai cho biết **chi phí của việc đi qua stack mạng** thay NVLink, không phải băng thông/độ trễ của một mạng vật lý. Số liệu cho mạng vật lý (10/25/100/400 Gb/s)
chỉ có thể là **ngoại suy bằng mô hình** t = α + 2(k−1)/k·n/B với (α, B) đo được, ghi rõ là ngoại suy. Meta mỗi lần chạy ghi `NODE_NAME` và các biến NCCL để biết hai pod
có thật sự ở hai node khác nhau không. Cùng mã chạy được trên cụm đa node thật: `SPREAD=1 tools/submit_wave.sh 10` buộc hai pod ở hai node (cần ≥ 2 node và quota).

**E9a, pipeline parallelism** (`e9_mp/pipeline.py`, GPipe). Rank = stage, S = 1, 2, 4; M = 1…16 micro-batch; VGG-16 (chia cân bằng MAC) và MLP 16×8192² (1,07 tỉ tham số).
**E9b, tensor parallelism** (`e9_mp/tensor_parallel.py`, khối MLP Megatron), T = 1, 2, 4; 8 khối H=8192, 4096 token.
**E9c, "không vừa một GPU":** MLP 45×16384² (12,1 tỉ tham số, FP32, trọng số+gradient ≈ 97 GB), S = 1, 2, 4.
Tính đúng đã kiểm tra trên CPU: chuẩn gradient toàn cục trùng nhau đến 8 chữ số với mọi S và M (pipeline), loss và chuẩn gradient trùng nhau với T=1, 2, 4 (tensor parallel).
Wave 9 chạy lại bước kiểm tra này trên GPU trước khi đo.

Dự đoán (ghi trước khi có số đo; độ tin cậy thấp vì chưa có số đo transport Socket):
1. Bằng chứng transport: ở `native`, NCCL log có P2P/NVLink; ở `socket`, mọi cặp đi `NET/Socket`. Nếu không đúng như vậy, kết quả `socket` không dùng được.
2. E3 qua Socket: busbw ở 1 GiB thấp hơn NVLink (≈356 GB/s) ít nhất 10×; α cao hơn α native (≈20 µs) ít nhất 5×.
3. E4 qua Socket, weak scaling k=4 (all-reduce sau backward): ResNet-18 hiệu suất 50–80%, VGG-16 dưới 40%. Mô hình (α, B) khớp từ E3 dự đoán `t_comm` đo được trong ±40% (E4 `t_comm` gồm cả gộp/tách vector gradient nên sẽ cao hơn một chút).
4. Pipeline: S=4, M=1 (model parallelism thuần tuý) không nhanh hơn S=1 quá 10% (bài: model parallelism không để tăng tốc). S=4, M=16 trên MLP đạt tăng tốc 1,5–3× và hiệu suất trong 15 điểm của 1 − (S−1)/(M+S−1); VGG-16 thấp hơn do stage lệch tải.
   Pipeline nhanh hơn data parallel cùng số GPU: **không** (E4 strong scaling VGG-16 đạt 2,65× ở 4 GPU).
5. Pipeline qua Socket chậm đi dưới 25% so với NVLink (chỉ truyền activation ở biên stage, vài trăm MB mỗi bước).
6. Tensor parallel: native T=4 hiệu suất ≥ 70% (lượng giao tiếp ≈ 3,2 GB/bước, ≈ 9 ms ở 356 GB/s, so với ~0,5 s tính toán). Qua Socket chậm hơn native 1,8–3× (cùng lượng 3,2 GB ở vài GB/s xấp xỉ bằng thời gian tính toán).
7. Không vừa một GPU: S=1 báo OOM; S=2 và S=4 chạy được; bộ nhớ đỉnh mỗi GPU ≈ tổng/S (±20%).
8. Đợt 10: busbw giữa hai pod ≥ 0,5× busbw của `socket` trong một pod (mạng container cùng máy không thêm nhiều chi phí so với TCP loopback). Đây là ô tôi đoán kém nhất: chưa biết loại CNI của cụm.

Cách chạy: `tools/submit_wave.sh 9 quick` rồi `9`; sau đó `10 quick` rồi `10`. Ước tính đợt 9 khoảng 45–60 phút, đợt 10 khoảng 50–70 phút (chưa đo). Cả hai giữ 4 GPU của namespace, chạy tuần tự.

## Việc chưa làm / phụ thuộc
- Cụm không kết nối được từ máy này tại thời điểm viết (TLS handshake bị ngắt), nên chưa chạy Job nào của Đợt 2–5.
- Chưa có script tổng hợp kết quả (SUMMARY.md như Đợt 1); viết sau khi có số liệu để tránh ép kết luận trước.
- Model parallelism (bài §2) nằm ngoài phạm vi: các mô hình đều vừa một GPU 80 GB, nên không có lý do thực tế để chia mô hình.
