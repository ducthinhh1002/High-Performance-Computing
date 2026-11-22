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
  timeout 1500 python -u -m torch.distributed.run --nnodes=2 --nproc_per_node="$n" --node_rank="$IDX" \
    --master_addr="$MASTER_ADDR" --master_port=$((29500 + c)) -m "$@"
}
net_run() {
  local net=$1; shift
  ( export NET_LABEL=$net
    if [ "$net" = socket ]; then export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_NET=Socket; fi
    "$@" )
}
probe() {  # bằng chứng NCCL chọn transport nào (P2P/SHM/NET) cho từng cặp rank
  export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,P2P
  mn 2 e3_comm.allreduce_bench --quick --ops all_reduce 2>&1 \
    | grep -E "Using network|NET/Socket|via P2P|via SHM|via NET|Connected" | sed -E 's/^[^ ]+ //; s/\[[0-9]+\]//' | sort | uniq -c | sort -rn | head -12
}
rc=0
for net in native socket; do
  echo "##### NCCL transport (net=$net, world 4)"
  net_run $net probe
  for n in 2 1; do                       # n=2: world 4 (2 GPU/pod), n=1: world 2 (mọi bước nhảy đều qua pod khác)
    W=$((2 * n)); R=${BASE}_mn_${net}_w$W
    echo "##### net=$net world=$W"
    RUN_ID=$R net_run $net mn $n e3_comm.allreduce_bench $Q --ops all_reduce || rc=1
    RUN_ID=$R net_run $net mn $n e4_dp.scaling $Q || rc=1
    RUN_ID=$R net_run $net mn $n e9_mp.pipeline $Q --archs vgg16,mlp || rc=1
    RUN_ID=$R net_run $net mn $n e9_mp.tensor_parallel $Q || rc=1
  done
done
echo "WAVE10 FINISHED rc=$rc idx=$IDX"
exit $rc
