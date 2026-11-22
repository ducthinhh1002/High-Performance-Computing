# Huấn luyện CNN song song và phân tán tự cài đặt bằng C / MPI / OpenMP / CUDA

Tái lập và mở rộng bài "Parallel and Distributed Deep Learning" (Hegde, Usmani, Stanford 2016 —
[hgpu.org/?p=16046](https://hgpu.org/?p=16046)). Không dùng framework: mạng CNN, các kernel
CPU/GPU, các thuật toán allreduce và cả bốn phương pháp tối ưu phân tán của bài đều được viết lại
từ đầu, để đo được những gì framework che mất. Phần này bổ sung cho `vinhvn/` (PyTorch trên H100).

Bài báo gốc không nằm trong thư mục này (bản quyền của tác giả).

## Mục tiêu

1. Cài đặt một bản **tuần tự** (C thuần) và một bản **song song** (MPI + OpenMP + CUDA) của cùng
   một quá trình huấn luyện CNN, rồi chứng minh hai bản cho cùng kết quả.
2. Đo lại khẳng định §3 của bài: thời gian conv tuyến tính theo batch, GPU nhanh hơn CPU nhiều lần.
3. So sánh 5 chiến lược của §5 (tuần tự, Sync SGD, Parallel SGD, ADMM, Downpour SGD) về tốc độ,
   chi phí giao tiếp và độ chính xác, với dữ liệu IID và không IID.
4. So mô hình chi phí α-β (và phân tích O(·) của bài) với số đo thật của 6 thuật toán allreduce.

## Sản phẩm

| Sản phẩm | Vị trí |
|---|---|
| Báo cáo | [`report/BAO_CAO.md`](report/BAO_CAO.md) |
| Slide | [`slides/tuannn-pdl-bench.pptx`](slides/tuannn-pdl-bench.pptx) |
| Mã nguồn **tuần tự** (C thuần) | [`code/sequential/`](code/sequential/): `seq_train.c` + `Makefile`, dùng lõi chung `code/src/{core,nn,data}` và kernel CPU `code/src/backend/cpu_kernels.c` |
| Mã nguồn **song song** | [`code/`](code/): MPI (`src/comm`, `src/train`), OpenMP (`src/backend/cpu_kernels.c`), CUDA (`src/backend/cuda_kernels.cu`), chương trình `apps/` |
| Kế hoạch và dự đoán (ghi trước khi chạy) | [`experiments/PLAN.md`](experiments/PLAN.md) |
| Kết quả: log/JSON gốc và bản tổng hợp | [`experiments/results/`](experiments/results/) |

## Cấu trúc thư mục

```
tuannn/
├── README.md                 # tài liệu này
├── report/                   # báo cáo (BAO_CAO.md) + hình
├── slides/                   # slide trình bày (.pptx)
├── experiments/
│   ├── PLAN.md               # dự đoán trước khi chạy, theo từng thí nghiệm E0–E5
│   ├── configs/              # cấu hình YAML của từng thí nghiệm (đợt 1)
│   └── results/              # mỗi thí nghiệm: raw/*.json, logs/, analysis/*.csv, figures/, report.md
│                             # và SUMMARY_dot1.md đối chiếu dự đoán ↔ kết quả
└── code/                     # dự án pdl-bench (CMake)
    ├── sequential/           # BẢN TUẦN TỰ: seq_train.c + Makefile (gcc, không MPI/OpenMP/CUDA)
    ├── include/pdl/          # API (backend, model, dataset, comm, train, …)
    ├── src/core/             # log, đồng hồ đo pha, RNG, JSON, CLI, ghi môi trường
    ├── src/nn/ src/data/     # mạng CNN (im2col + GEMM), dữ liệu MNIST/CIFAR/synthetic
    ├── src/backend/          # kernel CPU (naive/blocked/OpenMP) và CUDA (naive/tiled/cuBLAS)
    ├── src/comm/             # allreduce: ring, recursive doubling, tree, naive, MPI
    ├── src/train/            # BẢN SONG SONG: sync SGD, Parallel SGD, ADMM, Downpour SGD
    ├── apps/                 # pdl_train, pdl_bench_compute, pdl_bench_comm, pdl_check
    ├── tests/                # CTest: kiểm tra tính đúng + chạy thử mọi chiến lược
    ├── configs/              # bộ thí nghiệm đầy đủ cho máy mục tiêu (đợt 2)
    ├── pytools/pdlbench/     # Python: tải dữ liệu, chạy thí nghiệm, tổng hợp, vẽ, mô hình α-β, báo cáo
    ├── scripts/              # setup_ubuntu.sh, build.sh, quickstart.sh, run_all.sh
    └── docs/                 # kiến trúc, thuật toán, phương pháp đo, CLI, schema kết quả
```

## Cách chạy lại

### Môi trường

| | Đã dùng (đợt 1) | Máy mục tiêu (đợt 2) |
|---|---|---|
| CPU | Intel Core i5-8257U (4 nhân / 8 luồng), 16 GB | Intel Core i5-12600K (6P + 4E, 16 luồng) |
| GPU | không có | NVIDIA RTX 2060 6 GB (sm_75) |
| Hệ điều hành | macOS 13 | Ubuntu 22.04/24.04 hoặc Windows 11 + WSL2 |
| Trình biên dịch | Apple Clang 14 (không có OpenMP) | GCC ≥ 9, CUDA Toolkit ≥ 11 |
| MPI | MPICH 5.0.2 | OpenMPI ≥ 4 (hoặc MPICH) |
| Python | 3.9, numpy, pandas, matplotlib, pyyaml | như bên trái |

Cài đặt trên Ubuntu/WSL2 (cài gói hệ thống, tạo `.venv`, gợi ý cài CUDA):

```bash
cd tuannn/code
scripts/setup_ubuntu.sh
. .venv/bin/activate                      # cung cấp lệnh `pdlbench`
pdlbench data --dataset mnist cifar10     # tải và chuyển MNIST/CIFAR-10 sang code/data/*.pdlds
```

### 1. Bản tuần tự (C thuần)

Chỉ cần trình biên dịch C. Không cần MPI, OpenMP hay CUDA.

```bash
cd tuannn/code/sequential
make                                      # tạo ./seq_train (gcc -O3, một luồng)
./seq_train --help                        # liệt kê tham số
./seq_train --epochs 3 --output seq.json  # LeNet trên MNIST (cần ../data/mnist-*.pdlds)
make run-synthetic                        # chạy thử không cần tải dữ liệu
```

Kết quả in ra mỗi epoch: loss, độ chính xác train/test, thời gian, mẫu/giây. `--output` ghi JSON
cùng định dạng với bản song song để so sánh trực tiếp.

### 2. Bản song song (MPI + OpenMP + CUDA)

```bash
cd tuannn/code
scripts/build.sh                          # CMake Release; tự bật CUDA nếu có nvcc (sm_75); --cuda off để bỏ GPU
ctest --test-dir build --output-on-failure    # 12 bài kiểm tra tính đúng, khoảng 10 giây
```

Huấn luyện bằng `build/bin/pdl_train`. Số tiến trình do `mpirun -np` quyết định:

```bash
# Sync SGD, 4 tiến trình, batch tổng 256, allreduce dạng ring
mpirun -np 4 build/bin/pdl_train --strategy sync --global-batch 256 --lr 0.05 --allreduce ring --output sync4.json
# Parallel SGD có ShuffleSGD, ADMM, Downpour (rank 0 là parameter server)
mpirun -np 4 build/bin/pdl_train --strategy psgd --avg-period 10 --reshuffle 1
mpirun -np 4 build/bin/pdl_train --strategy admm --admm-local-steps 20
mpirun -np 5 build/bin/pdl_train --strategy downpour --push-every 2 --fetch-every 2
# GPU (CUDA), một tiến trình hoặc nhiều tiến trình dùng chung GPU
build/bin/pdl_train --strategy sequential --device cuda
mpirun -np 2 build/bin/pdl_train --strategy sync --device cuda
# OpenMP: --impl omp --threads 12 (CPU nhiều luồng trong một tiến trình)
build/bin/pdl_train --strategy sequential --impl omp --threads 12
```

Benchmark và kiểm tra:

```bash
build/bin/pdl_bench_compute --op conv --impls cpu-omp,cuda-tiled,cublas --batches 1,16,64,256   # §3 của bài
mpirun -np 4 build/bin/pdl_bench_comm --algos all                                                # 6 thuật toán allreduce
build/bin/pdl_check --device both                                                                # gradient, kernel CUDA ↔ CPU
mpirun -np 4 build/bin/pdl_check --tests allreduce,equivalence,sync
```

Danh sách đầy đủ tham số: [`code/docs/CLI.md`](code/docs/CLI.md).

### 3. Tái lập các thí nghiệm của đợt 1

Chạy từ `tuannn/code` sau khi build. Mỗi lệnh lần lượt chạy, tổng hợp, vẽ biểu đồ và viết báo cáo:

```bash
for e in e0_correctness e1_gemm e2_paper_cpu e3_comm e4_strong_scaling e5_strategies; do
  pdlbench all ../experiments/configs/dot1_$e.yaml --results-dir ../experiments/results
done
pdlbench perfmodel ../experiments/results/dot1_e4_strong_scaling --comm-dir ../experiments/results/dot1_e3_comm
```

Bộ thí nghiệm đầy đủ cho máy có GPU (đợt 2) nằm ở `code/configs/`. Chạy toàn bộ bằng `scripts/run_all.sh`.

## Kết quả

Đợt 1 (22/11/2026) trên i5-8257U (4 nhân / 8 luồng), không GPU, MPICH 5. Chi tiết và đối chiếu
dự đoán ở [`experiments/results/SUMMARY_dot1.md`](experiments/results/SUMMARY_dot1.md), phân tích ở
[`report/BAO_CAO.md`](report/BAO_CAO.md).

| Câu hỏi | Kết quả chính | Điều kiện đo |
|---|---|---|
| Bản song song có đúng bằng bản tuần tự? | **Có.** `seq_train` và `pdl_train` cho loss giống hệt; sync SGD k = 2/4 lệch tuần tự 5,7e-8 / 2,9e-8; gradient check sai số ≤ 1,1e-4 | 12/12 CTest, `pdl_check` 1/2/4 tiến trình |
| Conv tuyến tính theo batch (bài §3)? | **Có**: 5,78 ms/ảnh, R² = 0,9987, đúng dạng O(n) như bài. im2col + GEMM nhanh hơn conv trực tiếp 12–15× | lớp 96×11×11/4 trên 224×224, batch 1–64, 5 lần đo |
| Tối ưu cache cho GEMM | blocked 14,5–17,2 GFLOP/s, nhanh hơn naive tới 25× (N = 1024) | 1 luồng, 5 lần đo |
| Allreduce tốt nhất | message nhỏ: `MPI_Allreduce` (1 µs); message lớn: ring ≈ MPI < recursive doubling < tree < naive. α = 2,66 µs, 3,4 GB/s | k = 2/4/8, 4 B–16 MiB, 20 lần đo |
| Strong scaling sync SGD | speedup **1,93× / 3,29× / 3,48×** ở k = 2 / 4 / 8; độ chính xác ≈ 98% không đổi; giao tiếp + chờ 4,4% ở k = 4 | LeNet/MNIST, batch tổng 256, 3 lần lặp |
| So sánh 4 phương pháp (bài §5) | Sync = tuần tự (98,58% so với 98,64%, nhanh 3,4×). PSGD nhanh nhất (3,9×) nhưng rơi còn **77,7%** khi dữ liệu không IID; ShuffleSGD → 97,9%. ADMM 98,1% / 88,5%. Downpour chậm nhất, staleness ≈ 11 | 4 tiến trình, 3 epoch, 2 seed |

Đã chứng minh: tính đúng của bản song song; khẳng định §3 về CPU; ring/MPI tốt hơn cây trên một node;
giới hạn scale trên một máy là phần tính toán dùng chung tài nguyên, không phải giao tiếp; PSGD cần
trộn dữ liệu hoặc lấy trung bình thường xuyên khi dữ liệu không IID.

## Giới hạn

- **Chưa có số đo GPU và OpenMP.** Máy đợt 1 không có GPU và Apple Clang không hỗ trợ OpenMP. Mã CUDA
  mới qua kiểm tra biên dịch (clang CUDA sm_75), chưa chạy trên GPU. Con số "GPU nhanh hơn ≈ 40×"
  của bài chưa được kiểm chứng. Việc này thuộc đợt 2 trên i5-12600K + RTX 2060 (cấu hình sẵn ở `code/configs/`).
- **Một máy.** MPI đi qua shared memory, nên không suy ra được hành vi giữa nhiều máy qua mạng.
- **Mô hình nhỏ** (LeNet, 247 KB gradient): giao tiếp rẻ. Với mô hình lớn hơn, kết luận về tỉ lệ
  giao tiếp có thể khác.
- **Số lần lặp ít**: E4 lặp 3 lần, E5 2 seed. Độ lệch chuẩn ± 4 điểm ở E5 với dữ liệu không IID.
- **Siêu tham số chưa tinh chỉnh** riêng cho từng phương pháp.
- k = 8 trên máy 4 nhân dùng hyper-thread, nên con số ở k = 8 phản ánh tranh chấp nhân, không phải
  khả năng scale của thuật toán.
