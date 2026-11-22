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
│   ├── PLAN.md               # dự đoán theo từng thí nghiệm (đợt 1: E0–E5, đợt 2: E0–E7)
│   ├── configs/              # cấu hình YAML của từng thí nghiệm (dot1_*, dot2_*)
│   └── results/              # mỗi thí nghiệm: raw/*.json, logs/, analysis/*.csv, figures/, report.md
│                             # và SUMMARY_dot1.md, SUMMARY_dot2.md đối chiếu dự đoán ↔ kết quả
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

| | Đợt 1 | Đợt 2 (đã chạy) |
|---|---|---|
| CPU | Intel Core i5-8257U (4 nhân / 8 luồng), 16 GB | Intel Core i5-12600K (6P + 4E, 16 luồng), 15,8 GB (WSL2 được cấp 7,6 GB) |
| GPU | không có | NVIDIA RTX 2060 6 GB (sm_75) |
| Hệ điều hành | macOS 13 | Windows 11 + WSL2 (Ubuntu 24.04); Ubuntu 22.04/24.04 gốc cũng được hỗ trợ |
| Trình biên dịch | Apple Clang 14 (không có OpenMP) | GCC 13.3, CUDA Toolkit 12.0 |
| MPI | MPICH 5.0.2 | OpenMPI 4.1.6 |
| Python | 3.9, numpy, pandas, matplotlib, pyyaml | như bên trái |

Cài đặt trên Ubuntu/WSL2 (cài gói hệ thống, tạo `.venv`, gợi ý cài CUDA):

```bash
cd tuannn/code
scripts/setup_ubuntu.sh
. .venv/bin/activate                      # cung cấp lệnh `pdlbench`
pdlbench data --dataset mnist cifar10     # tải và chuyển MNIST/CIFAR-10 sang code/data/*.pdlds
```

**Lưu ý trên WSL2:** nếu `nvidia-smi` thấy GPU nhưng `pdl_check` báo "CUDA requested but not available",
gói apt `nvidia-cuda-toolkit` đã cài một `libcuda.so` che thư viện driver của WSL. Sửa bằng
`export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH` (nên thêm vào `~/.bashrc`). `pdlbench` có
bước kiểm tra trước khi chạy và sẽ dừng ngay với hướng dẫn này nếu cấu hình cần GPU mà CUDA không chạy.

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
ctest --test-dir build --output-on-failure    # 13 bài kiểm tra tính đúng, khoảng 10 giây
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

### 3. Tái lập các thí nghiệm

Chạy từ `tuannn/code` sau khi build. Mỗi lệnh lần lượt chạy, tổng hợp, vẽ biểu đồ và viết báo cáo:

```bash
# Đợt 1 (CPU + MPI, máy không GPU)
for e in e0_correctness e1_gemm e2_paper_cpu e3_comm e4_strong_scaling e5_strategies; do
  pdlbench all ../experiments/configs/dot1_$e.yaml --results-dir ../experiments/results
done
pdlbench perfmodel ../experiments/results/dot1_e4_strong_scaling --comm-dir ../experiments/results/dot1_e3_comm

# Đợt 2 (có GPU, khoảng 1 giờ trên i5-12600K + RTX 2060)
for e in e0_correctness e1_gemm e2_paper_cpu_gpu e3_comm e4_strong_scaling e4b_strong_scaling_unbound \
         e5_strategies e6_hybrid e7_gpu_mpi; do
  pdlbench all ../experiments/configs/dot2_$e.yaml --results-dir ../experiments/results
done
python ../report/make_figures.py          # vẽ lại hình báo cáo (f1–f5: đợt 1, g1–g6: đợt 2)
```

Bộ cấu hình đầy đủ nằm ở `code/configs/`; `scripts/run_all.sh` chạy toàn bộ. Khi đo, nên đóng ứng dụng
nặng và (trên Windows) đặt power plan High performance: `manifest.json` ghi lại cả hai điều kiện này.

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

**Đợt 2** (23/11/2026) trên i5-12600K + RTX 2060, Windows 11 + WSL2, OpenMPI 4.1.6, CUDA 12.0. 149 lần
chạy, 0 lỗi. Chi tiết ở [`experiments/results/SUMMARY_dot2.md`](experiments/results/SUMMARY_dot2.md).

| Câu hỏi | Kết quả chính | Điều kiện đo |
|---|---|---|
| CUDA có đúng như CPU? | **Có.** 105 kiểm tra, 0 lỗi; GEMM CUDA lệch ≤ 2,8e-7, pipeline CNN lệch 0 | `ctest` 13/13, `pdl_check` 35 (CUDA) và 67 (MPI) |
| GPU đạt bao nhiêu? | cuBLAS **6,23 TFLOP/s** (96,6% đỉnh FP32); tiled viết tay 1,2 TFLOP/s | GEMM N = 4096 |
| OpenMP 12 luồng | GEMM nhanh hơn 1 luồng **5,3×** (6 nhân vật lý); conv 4,1×; LeNet 2,7× | 3 lần chạy |
| GPU so với CPU (bài §3: ≈ 40×) | Conv batch 256: **54,5×** so với 1 luồng, **13,2×** so với 12 luồng; tính cả chép dữ liệu: 11× và 2,7× | lớp 96×11×11/4 trên 224×224 |
| Strong scaling sync SGD | **1,79× / 2,91× / 3,12× / 3,16× / 3,45×** ở k = 2 / 4 / 6 / 8 / 10, bão hoà vì phần tính toán chứ không phải giao tiếp | LeNet/MNIST, ghim và không ghim cho cùng kết quả |
| Allreduce | message lớn: ring nhanh nhất từ k = 4 (33 ms so với 47 ms của `MPI_Allreduce` ở k = 10); recursive doubling sụp ở k ≥ 8 | 16 MiB, k = 2–10 |
| MPI × OpenMP | tốt nhất 4×3 và 6×2 (1,37× so với chỉ-OpenMP) | 12 luồng phần cứng |
| Nhiều tiến trình dùng chung GPU | **chậm đi**: 0,54× (np = 2), 0,40× (np = 4); một tiến trình GPU nhanh hơn 1 luồng CPU **60×** | CNN trên CIFAR-10 |
| 4 phương pháp, 10 epoch | Sync = tuần tự (98,9%); PSGD không IID: 94,6% (k = 4), 88,4% (k = 8); ShuffleSGD và trung bình mỗi 10 bước đều gỡ được; ADMM xấu đi khi k tăng | 4 và 8 tiến trình, 2 seed |

Dự đoán sai ở đợt 2 (xem `experiments/PLAN.md`): strong scaling thấp hơn nhiều so với dự đoán (3,1–3,5×
thay vì 5–7×) và GPU/CPU 12 luồng chỉ 13× thay vì 20–60×.

## Giới hạn

- **Đợt 2 chạy trên WSL2, không phải Linux gốc.** Windows xếp lịch cho vCPU nên không ghim được P-core/E-core:
  hiệu ứng E-core của i5-12600K **chưa được kiểm chứng**. Power plan và tải nền của Windows ảnh hưởng số
  đo (đã đặt High performance, ghi vào `manifest.json`). WSL2 chỉ được cấp 7,6 GB RAM.
- **Hai lỗi của phép đo đã sửa trước khi có số đợt 2** (OpenMP vô tình chạy 1 luồng sau `--verify`; GPU
  đo lúc xung nhịp thấp), xem `code/docs/ai_effort_log.csv`. Nguyên nhân sai lệch sẽ không thể thấy nếu chỉ
  dựa vào `ctest`: phải so sánh với một lần chạy riêng và với đỉnh lý thuyết của GPU.
- **Hạn chế OpenMP chưa sửa**: conv song song theo ảnh nên batch 2–4 chỉ dùng 2–4 luồng.
- **Một máy.** MPI đi qua shared memory, nên không suy ra được hành vi giữa nhiều máy qua mạng.
- **Mô hình nhỏ** (LeNet, 247 KB gradient): giao tiếp rẻ. Với mô hình lớn hơn, kết luận về tỉ lệ
  giao tiếp có thể khác.
- **Số lần lặp ít**: E4 lặp 3 lần, E5 2 seed. Độ lệch chuẩn ± 4 điểm ở E5 với dữ liệu không IID.
- **Siêu tham số chưa tinh chỉnh** riêng cho từng phương pháp.
- Ở k lớn hiệu suất giảm mạnh (đợt 1: k = 8 trên máy 4 nhân dùng hyper-thread; đợt 2: từ k = 6). Nguyên nhân
  của đợt 2 chưa tách được (xung nhịp toàn nhân, tranh chấp bộ nhớ/L3, lập lịch của Windows), nên con số
  scaling ở đó không nên đọc là khả năng scale của thuật toán.
