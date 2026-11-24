#!/usr/bin/env bash
# Benchmark đầy đủ (cần 4 GPU và đã `make`). Mỗi dòng kết quả có dạng `RESULT k=v ...` (xem scripts/results_to_csv.py).
#   A. So sánh bản tuần tự (C, CPU) với bản CUDA một GPU trên mô hình nhỏ.
#   B. Mô hình vừa một GPU: 1 GPU, pipeline S=2,4 theo số micro-batch M, tensor parallel T=2,4.
#   C. Mô hình KHÔNG vừa một GPU (12,9 tỉ tham số, FP32): 1 GPU hết bộ nhớ, 2 và 4 GPU chạy được.
# Biến: STEPS (mặc định 3), WARMUP (1).
cd "$(dirname "$0")/.."
STEPS=${STEPS:-3}; WARMUP=${WARMUP:-1}
run() { echo "+ $*" >&2; "$@" | grep -E "^RESULT"; }

echo "# A. tuần tự (C, CPU) so với CUDA 1 GPU (H=512 L=4 B=128)"
run bin/mlp_seq --H 512 --L 4 --B 128 --steps 2 --warmup 1
run bin/mlp_cuda --mode pipeline --gpus 1 --micro 1 --H 512 --L 4 --B 128 --steps 20 --warmup 3

echo "# B. mô hình vừa một GPU (H=4096 L=8 B=2048, 1,07 tỉ tham số)"
B="--H 4096 --L 8 --B 2048 --steps $STEPS --warmup $WARMUP"
run bin/mlp_cuda --mode pipeline --gpus 1 --micro 1 $B
for S in 2 4; do for M in 1 2 4 8 16; do run bin/mlp_cuda --mode pipeline --gpus $S --micro $M $B; done; done
for T in 2 4; do run bin/mlp_cuda --mode tensor --gpus $T $B; done

echo "# C. mô hình không vừa một GPU (H=8192 L=24 B=256, 12,9 tỉ tham số; trọng số + gradient FP32 ≈ 103 GB > 80 GB)"
BIG="--H 8192 --L 24 --B 256 --steps 2 --warmup 1"
run bin/mlp_cuda --mode pipeline --gpus 1 --micro 4 $BIG
run bin/mlp_cuda --mode pipeline --gpus 2 --micro 4 $BIG
run bin/mlp_cuda --mode pipeline --gpus 4 --micro 4 $BIG
