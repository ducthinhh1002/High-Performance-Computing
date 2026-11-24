#!/usr/bin/env bash
# Chạy một lệnh trong container GPU trên cụm Kubernetes, rồi lấy log về results/.
#   Dùng:  KUBECONFIG=<tệp kubeconfig> scripts/cluster_run.sh <tên-job> <số GPU> "<lệnh chạy trong thư mục dự án>"
#   Ví dụ: scripts/cluster_run.sh probe 1 "nvcc --version; make seq"
# Mã (src/, scripts/, Makefile) được đóng gói vào ConfigMap rồi giải nén trong container, nên không cần build image.
# Biến: NAMESPACE (mặc định hpc-dl-exp), IMAGE, DEADLINE (giây, mặc định 3600).
set -uo pipefail
NAME=${1:?tên job}; GPUS=${2:?số GPU}; CMD=${3:?lệnh}
NS=${NAMESPACE:-hpc-dl-exp}
IMAGE=${IMAGE:-pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel}   # bản devel có nvcc, gcc, make
DEADLINE=${DEADLINE:-3600}
cd "$(dirname "$0")/.."
K="kubectl -n $NS"
mkdir -p results
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
tar -czf "$TMP/code.tgz" Makefile src scripts
$K create configmap "hpc-vinhvn-$NAME" --from-file=code.tgz="$TMP/code.tgz" --dry-run=client -o yaml | $K apply -f - >/dev/null
$K delete job "vinhvn-$NAME" --ignore-not-found >/dev/null
cat <<YAML | $K apply -f - >/dev/null
apiVersion: batch/v1
kind: Job
metadata: {name: vinhvn-$NAME}
spec:
  backoffLimit: 0
  activeDeadlineSeconds: $DEADLINE
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: run
          image: $IMAGE
          command: ["bash", "-c", "mkdir -p /tmp/work && tar -xzf /code/code.tgz -C /tmp/work && cd /tmp/work && $CMD"]
          resources:
            requests: {cpu: "16", memory: 64Gi, nvidia.com/gpu: "$GPUS"}
            limits:   {cpu: "16", memory: 64Gi, nvidia.com/gpu: "$GPUS"}
          volumeMounts: [{name: code, mountPath: /code}]
      volumes: [{name: code, configMap: {name: hpc-vinhvn-$NAME}}]
YAML
echo "Đã tạo job vinhvn-$NAME, đang chờ..."
for ((t=0; t<DEADLINE+120; t+=10)); do
  st=$($K get job "vinhvn-$NAME" -o jsonpath='{.status.succeeded}/{.status.failed}' 2>/dev/null)
  case "$st" in 1/*|*/[1-9]*) break;; esac
  sleep 10
done
$K logs "job/vinhvn-$NAME" --tail=-1 > "results/$NAME.log" 2>&1
echo "trạng thái (succeeded/failed): $st; log: results/$NAME.log"
tail -n 25 "results/$NAME.log"
