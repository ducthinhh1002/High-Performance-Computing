# Thực nghiệm: Parallel and Distributed Deep Learning

Tái lập và kiểm chứng các khẳng định trong bài "Parallel and Distributed Deep Learning" (Hegde, Usmani, Stanford):
§3 (CPU so với GPU) và §5 (data parallelism, ParallelSGD, ADMM, Downpour SGD), cùng model parallelism (§2).
Chạy trên một node 4× H100 (Kubernetes), PyTorch 2.5.1.

Bài báo gốc không nằm trong thư mục này.

## Đọc kết quả
Mỗi bản tổng hợp đối chiếu kết quả với dự đoán đã ghi **trước** khi chạy trong `experiments/PLAN.md`, gồm cả những chỗ dự đoán sai.

| File | Nội dung |
|---|---|
| `experiments/results/wave1/SUMMARY.md` | Đợt 1: convolution/GEMM, GPU so với CPU |
| `experiments/results/SUMMARY_waves2-5.md` | Đợt 2–5: giao tiếp, scaling đồng bộ, hội tụ, Downpour, VGG-16 BF16 |
| `experiments/results/SUMMARY_waves6-8.md` | Đợt 6–8: kiểm chứng bổ sung, lặp E3 và E4 năm lần |
| `experiments/results/SUMMARY_wave9.md` | Đợt 9: model parallelism và mô phỏng mạng trong một pod |

## Cấu trúc
- `experiments/e1_profile` … `e9_mp`: mã từng thí nghiệm (E1 profile, E2 CPU/GPU, E3 giao tiếp, E4 scaling, E5 hội tụ, E6 Downpour, E7 VGG BF16, E9 model parallelism).
- `experiments/common`: mô hình, dữ liệu, đo thời gian, khởi tạo torch.distributed.
- `experiments/k8s`, `experiments/tools`: manifest Job và script chạy (`tools/submit_wave.sh <N> [quick]`).
- `experiments/results`: log và CSV gốc của từng đợt.

## Giới hạn cần biết
- Cụm chỉ có **một node vật lý**. "Mạng giữa các máy" chỉ được mô phỏng bằng cách ép NCCL đi qua TCP; kết quả đó đo chi phí khi bỏ NVLink, không phải mạng vật lý.
- Hầu hết phép đo là một lần chạy; chỉ E3 và E4 lặp 5 lần, E5 có 3 seed.
- Đợt 10 (hai pod) chưa có trong thư mục này.
- Kubeconfig và thông tin cụm không được đưa vào repo (`.gitignore` chặn `*kubeconfig*`).
