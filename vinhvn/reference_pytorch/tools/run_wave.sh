#!/usr/bin/env bash
# Chạy một đợt thực nghiệm phân tán. WAVE=2|3|4|5, QUICK=1 để chạy thử nhanh.
#   2: E3 (chi phí giao tiếp, k=2,3,4) rồi E7 (điểm bất thường VGG-16 BF16, 1 GPU)
#   3: E4 (scaling data parallel, k=1,2,3,4)
#   4: E5 (hội tụ: sync / ParallelSGD / oneshot, shuffled so với sorted, k=4)
#   5: E6 (Downpour SGD bất đồng bộ, W=1..8)
#   6,7,8: kiểm chứng bổ sung (xem PLAN.md);  9: model parallelism + mô phỏng mạng trong một pod
#   10: chạy bằng run_mn.sh (2 pod)
set -uo pipefail
mkdir -p /tmp/exp && tar -xzf /code/code.tgz -C /tmp/exp
cd /tmp/exp && export PYTHONPATH=/tmp/exp
Q=""; [ "${QUICK:-0}" = "1" ] && Q="--quick"
export RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
echo "WAVE=$WAVE QUICK=${QUICK:-0} RUN_ID=$RUN_ID OMP_NUM_THREADS=${OMP_NUM_THREADS:-} OUT_DIR=$OUT_DIR"
nvidia-smi -L; nvidia-smi topo -m | head -12

trun() {  # trun <số rank> <module> [args...]: torchrun một process mỗi GPU
  local n=$1; shift
  python -u -m torch.distributed.run --standalone --nproc_per_node="$n" -m "$@"
}
net_run() {  # net_run <native|socket> <lệnh...>: socket = ép NCCL đi qua stack mạng, tắt NVLink/shared memory
  local net=$1; shift
  ( export NET_LABEL=$net
    if [ "$net" = socket ]; then export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_NET=Socket; fi
    "$@" )
}
rc=0
case "$WAVE" in
  2)
    for k in 2 3 4; do echo "##### E3 k=$k"; trun $k e3_comm.allreduce_bench $Q || rc=1; done
    echo "##### E7"; CUDA_VISIBLE_DEVICES=0 python -u -m e7_bf16.vgg_bf16 $Q || rc=1 ;;
  3)
    for k in 1 2 3 4; do echo "##### E4 k=$k"; trun $k e4_dp.scaling $Q || rc=1; done ;;
  4)
    echo "##### E5 k=4"; trun 4 e5_sgd.convergence $Q || rc=1 ;;
  5)
    echo "##### E6"; python -u -m e6_async.downpour $Q || rc=1 ;;
  6)  # Kiểm chứng bổ sung: E7 (profile channels_last), E6 mở rộng (W lớn, làm chậm server)
    echo "##### E7"; CUDA_VISIBLE_DEVICES=0 python -u -m e7_bf16.vgg_bf16 $Q || rc=1
    echo "##### E6 W=12,16"; python -u -m e6_async.downpour $Q --workers 12,16 || rc=1
    for x in 10 30; do
      echo "##### E6 tau-extra=${x}ms"
      python -u -m e6_async.downpour $Q --models resnet18_cifar --workers 1,2,4,8,12,16 --tau-extra-ms $x || rc=1
    done ;;
  7)  # E5b: GroupNorm thay BatchNorm (kiểm chứng giả thuyết BN) và thêm seed 2 cho BN
    echo "##### E5 GN seeds 0,1,2"; trun 4 e5_sgd.convergence $Q --model resnet18_gn_cifar --seeds 0,1,2 || rc=1
    echo "##### E5 BN seed 2";      trun 4 e5_sgd.convergence $Q --model resnet18_cifar --seeds 2 || rc=1 ;;
  8)  # Lặp E3 và E4 5 lần để có khoảng tin cậy
    BASE=$RUN_ID
    for r in 1 2 3 4 5; do
      for k in 2 3 4; do echo "##### E3 rep=$r k=$k"; RUN_ID=${BASE}_r$r trun $k e3_comm.allreduce_bench $Q || rc=1; done
      for k in 1 2 3 4; do echo "##### E4 rep=$r k=$k"; RUN_ID=${BASE}_r$r trun $k e4_dp.scaling $Q || rc=1; done
    done ;;
  9)  # E9 (model parallelism: pipeline, tensor parallel, "không vừa một GPU") + mô phỏng mạng trong một pod
    BASE=$RUN_ID
    echo "##### kiểm tra tính đúng (chuẩn gradient phải bằng nhau giữa các S, T)"
    for n in 1 2 4; do
      trun $n e9_mp.pipeline --quick --archs mlp --check || rc=1
      trun $n e9_mp.tensor_parallel --quick --check || rc=1
    done
    for net in native socket; do
      echo "##### NCCL transport thực sự được chọn (net=$net)"
      net_run $net env NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,P2P python -u -m torch.distributed.run --standalone \
        --nproc_per_node=4 -m e3_comm.allreduce_bench --quick --ops all_reduce 2>&1 \
        | grep -E "Using network|NET/Socket|via P2P|via SHM|via NET" | sed -E 's/^[^ ]+ //; s/\[[0-9]+\]//' | sort | uniq -c | sort -rn | head -12
    done
    echo "##### E3/E4 qua stack mạng (NCCL Socket) trong một pod"
    for k in 2 4; do RUN_ID=${BASE}_socket net_run socket trun $k e3_comm.allreduce_bench $Q --ops all_reduce || rc=1; done
    for k in 1 2 4; do RUN_ID=${BASE}_socket net_run socket trun $k e4_dp.scaling $Q || rc=1; done
    for net in native socket; do
      for S in 1 2 4; do
        [ "$net" = socket ] && [ "$S" = 1 ] && continue
        echo "##### E9 net=$net S/T=$S"
        RUN_ID=${BASE}_$net net_run $net trun $S e9_mp.pipeline $Q --archs vgg16,mlp || rc=1
        RUN_ID=${BASE}_$net net_run $net trun $S e9_mp.tensor_parallel $Q || rc=1
      done
    done
    for S in 1 2 4; do
      echo "##### E9 mlp_big (không vừa một GPU) S=$S"
      RUN_ID=${BASE}_big net_run native timeout 1500 python -u -m torch.distributed.run --standalone \
        --nproc_per_node=$S -m e9_mp.pipeline $Q --archs mlp_big || rc=1
    done ;;
  *) echo "WAVE không hợp lệ: $WAVE"; exit 2 ;;
esac
echo "WAVE$WAVE FINISHED rc=$rc"
exit $rc
