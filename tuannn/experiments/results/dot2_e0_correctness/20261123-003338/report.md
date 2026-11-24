# pdl-bench report

Generated 2026-11-23T00:33:45 from 1 run directory. CSV and LaTeX tables are in each run's `analysis/` folder, figures (PNG + PDF) in `figures/`.

## Experiment `correctness`

Run directory: `../experiments/results/dot2_e0_correctness/20261123-003338`

Correctness checks for every component. CUDA variants compare GPU kernels with the CPU reference. MPI tests run with 2 and 4 ranks.

Runs: ok: 5

### Performance model

- Single-node runs: MPI uses shared memory, so α/β describe intra-node copies and the model ignores memory-bandwidth contention between ranks.

### Correctness

105 checks, **0 failed**.

| name | device | world_size | passed | max_err | tolerance | detail |
|---|---|---|---|---|---|---|
| gradcheck/conv+relu+pool+fc | cpu | 1 | True | 0.0001031 | 0.02 | 56 params probed, 24 skipped at kinks |
| gradcheck/strided-conv+overlap-pool | cpu | 1 | True | 5.789e-05 | 0.02 | 66 params probed, 14 skipped at kinks |
| gradcheck/mlp | cpu | 1 | True | 7.499e-05 | 0.02 | 79 params probed, 14 skipped at kinks |
| gradcheck/two-conv | cpu | 1 | True | 3.051e-05 | 0.02 | 76 params probed, 11 skipped at kinks |
| determinism/same-seed-same-params | cpu | 1 | True | 0 | 0 | bitwise parameter hash |
| finite/detects-nan-and-inf | cpu | 1 | True | 2 | 2 | expects exactly 2 |
| kernels/gemm-cpu-naive-NN | cpu | 1 | True | 2.262e-07 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |
| kernels/gemm-cpu-naive-NT | cpu | 1 | True | 1.794e-07 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |
| kernels/gemm-cpu-naive-TN | cpu | 1 | True | 2.835e-07 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |
| kernels/gemm-cpu-naive-TT | cpu | 1 | True | 9.125e-08 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |
| kernels/gemm-cpu-blocked-NN | cpu | 1 | True | 0 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |
| kernels/gemm-cpu-blocked-NT | cpu | 1 | True | 0 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |
| kernels/gemm-cpu-blocked-TN | cpu | 1 | True | 0 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |
| kernels/gemm-cpu-blocked-TT | cpu | 1 | True | 0 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |
| kernels/gemm-cpu-omp-NN | cpu | 1 | True | 0 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |
| kernels/gemm-cpu-omp-NT | cpu | 1 | True | 0 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |
| kernels/gemm-cpu-omp-TN | cpu | 1 | True | 0 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |
| kernels/gemm-cpu-omp-TT | cpu | 1 | True | 0 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |
| kernels/conv-direct-vs-im2col | cpu | 1 | True | 1.23e-07 | 0.001 |  |
| gradcheck/conv+relu+pool+fc | cpu | 1 | True | 0.0001031 | 0.02 | 56 params probed, 24 skipped at kinks |
| gradcheck/strided-conv+overlap-pool | cpu | 1 | True | 5.789e-05 | 0.02 | 66 params probed, 14 skipped at kinks |
| gradcheck/mlp | cpu | 1 | True | 7.499e-05 | 0.02 | 79 params probed, 14 skipped at kinks |
| gradcheck/two-conv | cpu | 1 | True | 3.051e-05 | 0.02 | 76 params probed, 11 skipped at kinks |
| determinism/same-seed-same-params | cpu | 1 | True | 0 | 0 | bitwise parameter hash |
| finite/detects-nan-and-inf | cpu | 1 | True | 2 | 2 | expects exactly 2 |
| gradcheck/conv+relu+pool+fc | cuda | 1 | True | 0.0001712 | 0.02 | 56 params probed, 24 skipped at kinks |
| gradcheck/strided-conv+overlap-pool | cuda | 1 | True | 7.77e-05 | 0.02 | 66 params probed, 14 skipped at kinks |
| gradcheck/mlp | cuda | 1 | True | 7.489e-05 | 0.02 | 79 params probed, 14 skipped at kinks |
| gradcheck/two-conv | cuda | 1 | True | 4.465e-05 | 0.02 | 76 params probed, 11 skipped at kinks |
| determinism/same-seed-same-params | cuda | 1 | True | 0 | 0 | bitwise parameter hash |
| finite/detects-nan-and-inf | cuda | 1 | True | 2 | 2 | expects exactly 2 |
| kernels/gemm-cpu-naive-NN | cpu | 1 | True | 2.262e-07 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |
| kernels/gemm-cpu-naive-NT | cpu | 1 | True | 1.794e-07 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |
| kernels/gemm-cpu-naive-TN | cpu | 1 | True | 2.835e-07 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |
| kernels/gemm-cpu-naive-TT | cpu | 1 | True | 9.125e-08 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |
| kernels/gemm-cpu-blocked-NN | cpu | 1 | True | 0 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |
| kernels/gemm-cpu-blocked-NT | cpu | 1 | True | 0 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |
| kernels/gemm-cpu-blocked-TN | cpu | 1 | True | 0 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |
| kernels/gemm-cpu-blocked-TT | cpu | 1 | True | 0 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |
| kernels/gemm-cpu-omp-NN | cpu | 1 | True | 0 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |
| kernels/gemm-cpu-omp-NT | cpu | 1 | True | 0 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |
| kernels/gemm-cpu-omp-TN | cpu | 1 | True | 0 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |
| kernels/gemm-cpu-omp-TT | cpu | 1 | True | 0 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |
| kernels/conv-direct-vs-im2col | cpu | 1 | True | 1.23e-07 | 0.001 |  |
| kernels/gemm-cuda-naive-NN | cuda | 1 | True | 2.262e-07 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |
| kernels/gemm-cuda-naive-NT | cuda | 1 | True | 2.243e-07 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |
| kernels/gemm-cuda-naive-TN | cuda | 1 | True | 2.835e-07 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |
| kernels/gemm-cuda-naive-TT | cuda | 1 | True | 1.825e-07 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |
| kernels/gemm-cuda-tiled-NN | cuda | 1 | True | 2.262e-07 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |
| kernels/gemm-cuda-tiled-NT | cuda | 1 | True | 2.243e-07 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |
| kernels/gemm-cuda-tiled-TN | cuda | 1 | True | 2.835e-07 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |
| kernels/gemm-cuda-tiled-TT | cuda | 1 | True | 1.825e-07 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |
| kernels/cnn-pipeline-cuda-vs-cpu | cuda | 1 | True | 0 | 0.001 | conv, direct conv, relu, maxpool fwd/bwd, col2im, softmax |
| kernels/softmax-loss-cuda-vs-cpu | cuda | 1 | True | 0 | 0.001 |  |
| kernels/sumsq-cuda-vs-cpu | cuda | 1 | True | 1.537e-13 | 1e-06 |  |
| kernels/sgd-momentum-cuda-vs-cpu | cuda | 1 | True | 1.533e-07 | 0.001 |  |
| kernels/gemm-cublas-NN | cuda | 1 | True | 2.262e-07 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |
| kernels/gemm-cublas-NT | cuda | 1 | True | 2.243e-07 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |
| kernels/gemm-cublas-TN | cuda | 1 | True | 2.835e-07 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |
| kernels/gemm-cublas-TT | cuda | 1 | True | 1.825e-07 | 0.001 | 37x53x29 batch 3, alpha 1.5 beta 0.5 |

_45 more rows in the CSV files_

### Environment

| Item | Value |
|---|---|
| Host | TuanNguyen |
| OS | Linux 5.15.146.1-microsoft-standard-WSL2 x86_64 |
| CPU | 12th Gen Intel(R) Core(TM) i5-12600K |
| Physical cores | 8 |
| Logical CPUs | 16 |
| Memory | 7.6 GiB |
| GPU | GPU 0: NVIDIA GeForce RTX 2060 (UUID: GPU-32e2fcd7-f21c-bdc3-c469-19441f361371) |
| Compiler | gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0 |
| CUDA | Build cuda_12.0.r12.0/compiler.32267302_0 |
| MPI | mpirun (Open MPI) 4.1.6 |
| CMake | cmake version 3.28.3 |
| Python | 3.12.3 |
| Git commit | `4acdeb458c8a` (dirty) |
