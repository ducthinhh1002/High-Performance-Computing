#!/usr/bin/env bash
# Đợt 10: chạy trong MỖI pod của một Job Indexed (2 pod). Hai pod chạy cùng một chuỗi lệnh, ghép lại thành các job
# torchrun đa node (rendezvous tĩnh qua DNS của Service headless).
# Cụm hiện chỉ có 1 node vật lý, nên hai pod nằm trên cùng một máy: đường pod↔pod đi qua mạng container (CNI),
# không qua card mạng/switch. Meta ghi NODE_NAME để biết hai pod có thật sự ở hai máy khác nhau hay không.
set -uo pipefail
mkdir -p /tmp/exp && tar -xzf /code/code.tgz -C /tmp/exp
cd /tmp/exp && export PYTHONPATH=/tmp/exp
Q=""; [ "${QUICK:-0}" = "1" ] && Q="--quick"
BASE=${RUN_ID:?RUN_ID phải giống nhau ở hai pod}
IDX=${JOB_COMPLETION_INDEX:?}
echo "MN: idx=$IDX node=${NODE_NAME:-?} master=$MASTER_ADDR run=$BASE quick=${QUICK:-0}"; nvidia-smi -L

mn() {  # mn <số GPU mỗi pod> <module> [args...]; cổng tăng dần theo bộ đếm trong file (cùng thứ tự ở hai pod)
  local n=$1; shift
  local c=$(( $(cat /tmp/cnt 2>/dev/null || echo 0) + 1 )); echo $c > /tmp/cnt
  timeout -k 20 "$STEP_TIMEOUT" python -u -m torch.distributed.run --nnodes=2 --nproc_per_node="$n" --node_rank="$IDX" \
    --master_addr="$MASTER_ADDR" --master_port=$((29500 + c)) -m "$@"
}
export FAST_EXIT=1     # thoát thẳng sau khi xong (huỷ communicator giữa pod có thể treo); xem common/dist.py
net_run() {
  local net=$1; shift
  ( export NET_LABEL=$net
    if [ "$net" = socket ]; then export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_NET=Socket
    else export DDP_BUCKET_MB=${NATIVE_BUCKET_MB:-1000}   # native qua 2 pod: nhiều bucket DDP chồng lấp làm treo; dùng 1 bucket (không chồng lấp)
    fi
    "$@" )
}
probe() {  # bằng chứng NCCL chọn transport nào (P2P/SHM/NET) cho từng cặp rank
  export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,P2P
  mn 2 e3_comm.allreduce_bench --quick --ops all_reduce 2>&1 \
    | grep -E "Using network|NET/Socket|via P2P|via SHM|via NET|Connected" | sed -E 's/^[^ ]+ //; s/\[[0-9]+\]//' | sort | uniq -c | sort -rn | head -12
}
# Cấu hình (mặc định = chạy đầy đủ). Chẩn đoán: DIAG=1 NETS=native WORLDS=2 STEPS="e3 e4 pp tp"
NETS=${NETS:-"native socket"}; WORLDS=${WORLDS:-"2 1"}; STEPS=${STEPS:-"e3 e4 pp tp"}
STEP_TIMEOUT=${STEP_TIMEOUT:-900}
has() { [[ " $STEPS " == *" $1 "* ]]; }
echo "CFG: NETS='$NETS' WORLDS='$WORLDS' STEPS='$STEPS' STEP_TIMEOUT=$STEP_TIMEOUT DIST_TIMEOUT_S=${DIST_TIMEOUT_S:-600} DIAG=${DIAG:-0}"
if [ "${DIAG:-0}" = 1 ]; then
  # Chẩn đoán treo: cài py-spy; tiến trình worker sống quá HANG_S giây thì in stack Python của nó (một lần mỗi pid).
  pip install -q py-spy > /tmp/pyspy.install 2>&1 || echo "CẢNH BÁO: không cài được py-spy: $(tail -1 /tmp/pyspy.install)"
  export NCCL_DEBUG=${NCCL_DEBUG:-WARN} GLOO_SOCKET_IFNAME=eth0
  [ "${DEBUG_DETAIL:-0}" = 1 ] && export TORCH_DISTRIBUTED_DEBUG=DETAIL    # phát hiện collective lệch nhau giữa các rank
  ( while sleep 10; do
      for pid in $(pgrep -f "python -u -m e[0-9]"); do
        et=$(ps -o etimes= -p $pid 2>/dev/null | tr -d ' ')
        [ -z "$et" ] && et=$(ps -o etime= -p $pid 2>/dev/null | awk -F'[-:]' '{n=NF; s=$n+$(n-1)*60; if(n>=3)s+=$(n-2)*3600; if(n>=4)s+=$(n-3)*86400; print s}')
        if [ "${et:-0}" -gt "${HANG_S:-60}" ] && ! grep -q " $pid " /tmp/dumped 2>/dev/null; then
          echo "=== PY-SPY pid=$pid sống ${et}s: $(ps -o args= -p $pid | cut -c1-90)"
          py-spy dump --pid $pid 2>&1 | head -45
          echo " $pid " >> /tmp/dumped
        fi
      done
    done ) &
fi
if has e4m; then
  # Ma trận chẩn đoán: mỗi biến thể đổi đúng một yếu tố so với "native, 4 rank qua 2 pod" (đã biết là treo ở DDP).
  variant() {  # variant <nhãn> <net> <GPU mỗi pod> "<module và đối số>" [BIẾN=GIÁ_TRỊ...]
    local label=$1 net=$2 n=$3 mod=$4; shift 4
    echo "##### VARIANT $label net=$net gpu/pod=$n mod='$mod' env='$*'"
    ( [ $# -gt 0 ] && export "$@"; RUN_ID=${BASE}_mn_$label net_run $net mn $n $mod $Q ) || echo "VARIANT $label: rc!=0"
  }
  variant socket-w4      socket 2 "e4_dp.scaling"                      # mọi cặp đi Socket (không trộn P2P với mạng)
  variant native-w2      native 1 "e4_dp.scaling"                      # 1 GPU mỗi pod: mọi bước nhảy đều qua pod khác
  variant nobcast-w4     native 2 "e4_dp.scaling" DDP_BCAST_BUF=0      # DDP không broadcast buffer BatchNorm
  variant bucket1000-w4  native 2 "e4_dp.scaling" DDP_BUCKET_MB=1000   # DDP gộp thành một bucket
  variant pp-sleep       native 2 "e9_mp.pipeline --archs mlp" TEARDOWN_SLEEP_S=5     # chờ 5 s trước khi huỷ nhóm
  variant tp-sleep       native 2 "e9_mp.tensor_parallel" TEARDOWN_SLEEP_S=5
  variant coll-w4        native 2 "e4_dp.scaling" NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL   # ghi từng collective để so giữa các rank
  echo "WAVE10 FINISHED rc=0 idx=$IDX"
  exit 0
fi
rc=0
for net in $NETS; do
  echo "##### NCCL transport (net=$net, world 4)"
  net_run $net probe
  for n in $WORLDS; do                   # n=2: world 4 (2 GPU/pod), n=1: world 2 (mọi bước nhảy đều qua pod khác)
    W=$((2 * n)); R=${BASE}_mn_${net}_w$W
    echo "##### net=$net world=$W"
    has e3 && { RUN_ID=$R net_run $net mn $n e3_comm.allreduce_bench $Q --ops all_reduce || rc=1; }
    has e4 && { RUN_ID=$R net_run $net mn $n e4_dp.scaling $Q || rc=1; }
    has pp && { RUN_ID=$R net_run $net mn $n e9_mp.pipeline $Q --archs vgg16,mlp || rc=1; }
    has tp && { RUN_ID=$R net_run $net mn $n e9_mp.tensor_parallel $Q || rc=1; }
  done
done
echo "WAVE10 FINISHED rc=$rc idx=$IDX"
exit $rc
