#!/usr/bin/env bash
# Đợt 1: E2 rồi E1 trên một thiết bị. DEVICE=cpu|cuda, QUICK=1 để chạy thử nhanh.
set -uo pipefail
mkdir -p /tmp/exp && tar -xzf /code/code.tgz -C /tmp/exp
cd /tmp/exp && export PYTHONPATH=/tmp/exp
Q=""; [ "${QUICK:-0}" = "1" ] && Q="--quick"
echo "DEVICE=$DEVICE QUICK=${QUICK:-0} OMP_NUM_THREADS=${OMP_NUM_THREADS:-} OUT_DIR=$OUT_DIR"
[ "$DEVICE" = "cuda" ] && nvidia-smi -L

rc=0
python -u -m e2_cpu_gpu.bench --device "$DEVICE" $Q || rc=1
python -u -m e1_profile.profile_ops --device "$DEVICE" $Q || rc=1
echo "WAVE1 FINISHED device=$DEVICE rc=$rc"
exit $rc
