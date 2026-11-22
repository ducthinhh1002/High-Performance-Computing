#!/usr/bin/env bash
# Đo băng thông collective trên 4 GPU, làm mốc cho E3 (communication cost).
# Kết quả: $OUT/<run_id>/*.txt trên hostPath, đồng thời in ra log.
set -uo pipefail
BIN=${BIN:-/hostdata/tools/nccl-tests}
RUN=$(date -u +%Y%m%dT%H%M%SZ)
OUT=${OUT:-/hostdata/results/nccl-tests}/$RUN
mkdir -p "$OUT"
# Image runtime để libnccl và libcudart trong các gói pip nvidia/* → thêm mọi thư mục lib của chúng.
NV_LIBS=$(python -c "
import nvidia, os, glob
print(':'.join(sorted(glob.glob(os.path.join(list(nvidia.__path__)[0], '*', 'lib')))))")
export LD_LIBRARY_PATH=$NV_LIBS:${LD_LIBRARY_PATH:-}
ARGS="-b 8 -e 1G -f 2 -w 5 -n 20"

# Kiểm tra trước: thiếu thư viện thì dừng ngay, không giữ 4 GPU vô ích.
if ldd "$BIN/all_reduce_perf" | grep -q "not found"; then
  echo "PREFLIGHT FAIL: missing shared libraries:"; ldd "$BIN/all_reduce_perf" | grep "not found"; exit 1
fi
ldd "$BIN/all_reduce_perf" | grep -E "nccl|cudart"

run() {  # run <tên> <lệnh...>
  local name=$1; shift
  echo; echo "===== $name ====="; echo "+ $*"
  "$@" 2>&1 | tee "$OUT/$name.txt"
}

echo "===== topology ====="
nvidia-smi -L | tee "$OUT/gpus.txt"
nvidia-smi topo -m | tee "$OUT/topo.txt"
nvidia-smi nvlink -s 2>&1 | head -40 | tee "$OUT/nvlink_status.txt"
cat "$BIN/BUILD_INFO.txt"

# Mức "nhanh": NCCL mặc định (NVLink/NVSwitch nếu có). NCCL_DEBUG để log transport được chọn.
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH run all_reduce_g4_debug "$BIN/all_reduce_perf" -b 1M -e 1M -g 4 -n 5
run all_reduce_g4     "$BIN/all_reduce_perf" $ARGS -g 4
run all_reduce_g2     env CUDA_VISIBLE_DEVICES=0,1 "$BIN/all_reduce_perf" $ARGS -g 2
run reduce_g4         "$BIN/reduce_perf"     $ARGS -g 4
run broadcast_g4      "$BIN/broadcast_perf"  $ARGS -g 4
run all_gather_g4     "$BIN/all_gather_perf" $ARGS -g 4

# Tách riêng đóng góp của NVLS (reduce trong NVSwitch): tắt NVLS, giữ P2P qua NVLink.
run all_reduce_g4_nonvls env NCCL_NVLS_ENABLE=0 "$BIN/all_reduce_perf" $ARGS -g 4

# Mức "trung bình": dữ liệu đi qua bộ nhớ host. Với >2 GPU, NCCL_P2P_DISABLE một mình
# không đủ vì NCCL vẫn dùng NVLS (đo 20/11: 366 GB/s) → phải tắt cả NVLS.
P2P_OFF="NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0"
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH run all_reduce_g4_nop2p_nonvls_debug env $P2P_OFF "$BIN/all_reduce_perf" -b 1M -e 1M -g 4 -n 5
run all_reduce_g4_nop2p        env NCCL_P2P_DISABLE=1 "$BIN/all_reduce_perf" $ARGS -g 4   # giữ để đối chiếu
run all_reduce_g4_nop2p_nonvls env $P2P_OFF "$BIN/all_reduce_perf" $ARGS -g 4
run all_reduce_g2_nop2p_nonvls env $P2P_OFF CUDA_VISIBLE_DEVICES=0,1 "$BIN/all_reduce_perf" $ARGS -g 2
run reduce_g4_nop2p_nonvls     env $P2P_OFF "$BIN/reduce_perf"    $ARGS -g 4
run broadcast_g4_nop2p_nonvls  env $P2P_OFF "$BIN/broadcast_perf" $ARGS -g 4

# Tóm tắt: dòng "Avg bus bandwidth" của mỗi lần chạy.
echo; echo "===== SUMMARY (avg busbw, GB/s) ====="
for f in "$OUT"/*.txt; do
  bw=$(grep -h "Avg bus bandwidth" "$f" | awk '{print $NF}')
  [ -n "$bw" ] && printf "%-24s %s\n" "$(basename "$f" .txt)" "$bw"
done | tee "$OUT/summary.txt"
echo "results in $OUT"
