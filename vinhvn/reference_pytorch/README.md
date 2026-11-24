# Đối chứng: thực nghiệm bằng PyTorch/NCCL (KHÔNG thuộc phần nộp)

Thư mục này giữ lại phần thực nghiệm làm ban đầu trên PyTorch 2.5.1 và NCCL (4× H100): giao tiếp NCCL, data parallel,
hội tụ ParallelSGD, Downpour SGD, pipeline và tensor parallelism, mô phỏng mạng. Đề bài yêu cầu **tự cài đặt, không dùng
thư viện có sẵn**, bằng MPI hoặc CUDA, nên phần này **không phải bài nộp**. Nó được giữ lại vì hai lý do:

1. Làm **mốc đối chiếu** cho bản tự cài đặt ở `../src/` (cùng mô hình khối MLP, cùng câu hỏi về model parallelism):
   số liệu của PyTorch cho biết bản tự viết còn cách một thư viện đã tối ưu bao xa.
2. Chứa các kết quả khác (đo trên cụm Kubernetes, mô phỏng mạng qua hai pod) mà phần tự cài đặt không lặp lại.

Phần nộp nằm ở `../README.md`, `../src/`, `../scripts/`, `../Makefile`. Các bản tổng hợp kết quả của thư mục này:
`results/SUMMARY_waves2-5.md`, `results/SUMMARY_waves6-8.md`, `results/SUMMARY_wave9.md`, `results/SUMMARY_wave10.md`,
kế hoạch và dự đoán trong `PLAN.md`. Các đường dẫn trong các tài liệu này là tương đối so với thư mục này.
