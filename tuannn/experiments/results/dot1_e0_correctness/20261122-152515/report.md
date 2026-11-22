# pdl-bench report

Generated 2026-11-22T15:25:17 from 1 run directory. CSV and LaTeX tables are in each run's `analysis/` folder, figures (PNG + PDF) in `figures/`.

## Experiment `dot1_e0_correctness`

Run directory: `tuannn/experiments/results/dot1_e0_correctness/20261122-152515`

pdl_check on CPU (1 process) and MPI tests with 2 and 4 processes.

Runs: ok: 3

### Performance model

- Single-node runs: MPI uses shared memory, so α/β describe intra-node copies and the model ignores memory-bandwidth contention between ranks.

### Correctness

57 checks, **0 failed**.

| name | device | world_size | passed | max_err | tolerance | detail |
|---|---|---|---|---|---|---|
| gradcheck/conv+relu+pool+fc | cpu | 1 | True | 0.0001035 | 0.02 | 56 params probed, 24 skipped at kinks |
| gradcheck/strided-conv+overlap-pool | cpu | 1 | True | 7.608e-05 | 0.02 | 66 params probed, 14 skipped at kinks |
| gradcheck/mlp | cpu | 1 | True | 0.0001145 | 0.02 | 79 params probed, 14 skipped at kinks |
| gradcheck/two-conv | cpu | 1 | True | 2.73e-05 | 0.02 | 76 params probed, 11 skipped at kinks |
| determinism/same-seed-same-params | cpu | 1 | True | 0 | 0 | bitwise parameter hash |
| finite/detects-nan-and-inf | cpu | 1 | True | 2 | 2 | expects exactly 2 |
| kernels/gemm-cpu-naive-NN | cpu | 1 | True | 2.262e-07 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |
| kernels/gemm-cpu-naive-NT | cpu | 1 | True | 2.691e-07 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |
| kernels/gemm-cpu-naive-TN | cpu | 1 | True | 2.835e-07 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |
| kernels/gemm-cpu-naive-TT | cpu | 1 | True | 0 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |
| kernels/gemm-cpu-blocked-NN | cpu | 1 | True | 0 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |
| kernels/gemm-cpu-blocked-NT | cpu | 1 | True | 0 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |
| kernels/gemm-cpu-blocked-TN | cpu | 1 | True | 0 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |
| kernels/gemm-cpu-blocked-TT | cpu | 1 | True | 0 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |
| kernels/gemm-cpu-omp-NN | cpu | 1 | True | 0 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |
| kernels/gemm-cpu-omp-NT | cpu | 1 | True | 0 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |
| kernels/gemm-cpu-omp-TN | cpu | 1 | True | 0 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |
| kernels/gemm-cpu-omp-TT | cpu | 1 | True | 0 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |
| kernels/conv-direct-vs-im2col | cpu | 1 | True | 1.23e-07 | 0.001 |  |
| equivalence/sync-sgd-vs-sequential | cpu | 2 | True | 5.72e-08 | 0.0001 | 2 ranks x batch 8 vs 1 rank x batch 16, 5 steps |
| sync/replicas-identical-mpi | cpu | 2 | True | 0 | 0 | bitwise parameter hash |
| sync/replicas-identical-ring | cpu | 2 | True | 0 | 0 | bitwise parameter hash |
| sync/replicas-identical-rd | cpu | 2 | True | 0 | 0 | bitwise parameter hash |
| sync/replicas-identical-tree | cpu | 2 | True | 0 | 0 | bitwise parameter hash |
| sync/replicas-identical-naive | cpu | 2 | True | 0 | 0 | bitwise parameter hash |
| sync/replicas-identical-reduce_bcast | cpu | 2 | True | 0 | 0 | bitwise parameter hash |
| allreduce/mpi-exact-sum | cpu | 2 | True | 0 | 0 | sizes 1, 7, 1000, 65539 |
| allreduce/mpi-identical-on-ranks | cpu | 2 | True | 0 | 0 |  |
| allreduce/ring-exact-sum | cpu | 2 | True | 0 | 0 | sizes 1, 7, 1000, 65539 |
| allreduce/ring-identical-on-ranks | cpu | 2 | True | 0 | 0 |  |
| allreduce/rd-exact-sum | cpu | 2 | True | 0 | 0 | sizes 1, 7, 1000, 65539 |
| allreduce/rd-identical-on-ranks | cpu | 2 | True | 0 | 0 |  |
| allreduce/tree-exact-sum | cpu | 2 | True | 0 | 0 | sizes 1, 7, 1000, 65539 |
| allreduce/tree-identical-on-ranks | cpu | 2 | True | 0 | 0 |  |
| allreduce/naive-exact-sum | cpu | 2 | True | 0 | 0 | sizes 1, 7, 1000, 65539 |
| allreduce/naive-identical-on-ranks | cpu | 2 | True | 0 | 0 |  |
| allreduce/reduce_bcast-exact-sum | cpu | 2 | True | 0 | 0 | sizes 1, 7, 1000, 65539 |
| allreduce/reduce_bcast-identical-on-ranks | cpu | 2 | True | 0 | 0 |  |
| equivalence/sync-sgd-vs-sequential | cpu | 4 | True | 2.896e-08 | 0.0001 | 4 ranks x batch 8 vs 1 rank x batch 32, 5 steps |
| sync/replicas-identical-mpi | cpu | 4 | True | 0 | 0 | bitwise parameter hash |
| sync/replicas-identical-ring | cpu | 4 | True | 0 | 0 | bitwise parameter hash |
| sync/replicas-identical-rd | cpu | 4 | True | 0 | 0 | bitwise parameter hash |
| sync/replicas-identical-tree | cpu | 4 | True | 0 | 0 | bitwise parameter hash |
| sync/replicas-identical-naive | cpu | 4 | True | 0 | 0 | bitwise parameter hash |
| sync/replicas-identical-reduce_bcast | cpu | 4 | True | 0 | 0 | bitwise parameter hash |
| allreduce/mpi-exact-sum | cpu | 4 | True | 0 | 0 | sizes 1, 7, 1000, 65539 |
| allreduce/mpi-identical-on-ranks | cpu | 4 | True | 0 | 0 |  |
| allreduce/ring-exact-sum | cpu | 4 | True | 0 | 0 | sizes 1, 7, 1000, 65539 |
| allreduce/ring-identical-on-ranks | cpu | 4 | True | 0 | 0 |  |
| allreduce/rd-exact-sum | cpu | 4 | True | 0 | 0 | sizes 1, 7, 1000, 65539 |
| allreduce/rd-identical-on-ranks | cpu | 4 | True | 0 | 0 |  |
| allreduce/tree-exact-sum | cpu | 4 | True | 0 | 0 | sizes 1, 7, 1000, 65539 |
| allreduce/tree-identical-on-ranks | cpu | 4 | True | 0 | 0 |  |
| allreduce/naive-exact-sum | cpu | 4 | True | 0 | 0 | sizes 1, 7, 1000, 65539 |
| allreduce/naive-identical-on-ranks | cpu | 4 | True | 0 | 0 |  |
| allreduce/reduce_bcast-exact-sum | cpu | 4 | True | 0 | 0 | sizes 1, 7, 1000, 65539 |
| allreduce/reduce_bcast-identical-on-ranks | cpu | 4 | True | 0 | 0 |  |

### Environment

| Item | Value |
|---|---|
| Host | dev-macbook |
| OS | Darwin 22.6.0 x86_64 |
| CPU | Intel(R) Core(TM) i5-8257U CPU @ 1.40GHz |
| Physical cores | 4 |
| Logical CPUs | 8 |
| Memory | 16.0 GiB |
| Compiler | Apple clang version 14.0.3 (clang-1403.0.22.14.1) |
| MPI | HYDRA build details: |
| CMake | cmake version 4.4.5 |
| Python | 3.9.6 |
| Git commit | `5b0f0ac8f200` (dirty) |
