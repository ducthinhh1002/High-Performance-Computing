#!/usr/bin/env bash
# Dùng: tools/submit_wave.sh <2|3|4|5> [quick]
# Đóng gói code vào ConfigMap hpc-code, tạo Job, chờ xong, lấy log về results/wave<N>/.
# Cần: KUBECONFIG trỏ tới kubeconfig của SA hpc-runner. Chỉ chạy một đợt tại một thời điểm.
set -euo pipefail
WAVE=${1:?wave 2..5}; MODE=${2:-full}
cd "$(dirname "$0")/.."
K="kubectl -n hpc-dl-exp"
case $WAVE in 2|3|4|5|6|7|8|9|10) GPUS=4; CPUS=32; DEADLINE=9000;; *) echo bad wave; exit 2;; esac
QUICK=0; SUFFIX=""; [ "$MODE" = quick ] && { QUICK=1; SUFFIX=-quick; DEADLINE=1200; }
NAME=wave$WAVE$SUFFIX

# Kiểm tra quota GPU trước khi tạo Job, tránh Job kẹt Pending.
USED=$($K get resourcequota hpc-dl-exp-quota -o jsonpath='{.status.used.requests\.nvidia\.com/gpu}' 2>/dev/null || echo 0)
[ "${USED:-0}" -gt 0 ] && { echo "Đang dùng $USED GPU trong namespace, dừng."; $K get pods; exit 1; }

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
tar --exclude='results' --exclude='__pycache__' --exclude='k8s' -czf "$TMP/code.tgz" common e1_profile e2_cpu_gpu e3_comm e4_dp e5_sgd e6_async e7_bf16 e9_mp data
$K create configmap hpc-code --from-file=code.tgz="$TMP/code.tgz" --from-file=run_wave.sh=tools/run_wave.sh \
  --from-file=run_wave1.sh=tools/run_wave1.sh --from-file=run_mn.sh=tools/run_mn.sh --dry-run=client -o yaml | $K apply -f -
$K delete job "$NAME" --ignore-not-found
if [ "$WAVE" = 10 ]; then
  # Đợt 10: Job Indexed 2 pod × 2 GPU + Service headless. SPREAD=1 buộc hai pod ở hai node khác nhau (cần cụm ≥ 2 node).
  AFF=""; [ "${SPREAD:-0}" = 1 ] && AFF='      affinity: {podAntiAffinity: {requiredDuringSchedulingIgnoredDuringExecution: [{labelSelector: {matchLabels: {app: hpc-wave-mn}}, topologyKey: kubernetes.io/hostname}]}}'
  $K delete service "$NAME-svc" --ignore-not-found
  RUNID=$(date -u +%Y%m%dT%H%M%SZ)
  sed -e "s/@NAME@/$NAME/g" -e "s/@QUICK@/$QUICK/g" -e "s/@RUNID@/$RUNID/g" -e "s/@DEADLINE@/$DEADLINE/g" \
      -e "s|@AFFINITY@|$AFF|" k8s/wave-mn-job.tmpl.yaml | ${APPLY:-$K apply -f -}
  [ "${DRYRUN:-0}" = 1 ] && exit 0
  echo "Đã tạo Job $NAME (2 pod), đang chờ..."
  $K wait --for=condition=complete --timeout=${DEADLINE}s job/"$NAME" || $K wait --for=condition=failed --timeout=60s job/"$NAME" || true
  mkdir -p results/wave$WAVE
  for i in 0 1; do $K logs -l job-name="$NAME",batch.kubernetes.io/job-completion-index=$i --tail=-1 > results/wave$WAVE/$NAME-pod$i.log 2>&1 || true; done
  cp results/wave$WAVE/$NAME-pod0.log results/wave$WAVE/$NAME.log
  echo "Log: results/wave$WAVE/$NAME.log (+ -pod1.log)"; tail -3 results/wave$WAVE/$NAME.log
  exit 0
fi
sed -e "s/@WAVE@/$WAVE/g" -e "s/@SUFFIX@/$SUFFIX/g" -e "s/@GPUS@/$GPUS/g" -e "s/@CPUS@/$CPUS/g" \
    -e "s/@QUICK@/$QUICK/g" -e "s/@DEADLINE@/$DEADLINE/g" k8s/wave-job.tmpl.yaml | $K apply -f -
echo "Đã tạo Job $NAME, đang chờ..."
$K wait --for=condition=complete --timeout=${DEADLINE}s job/"$NAME" || \
  $K wait --for=condition=failed --timeout=60s job/"$NAME" || true
mkdir -p results/wave$WAVE
$K logs job/"$NAME" > results/wave$WAVE/$NAME.log
echo "Log: results/wave$WAVE/$NAME.log"; tail -3 results/wave$WAVE/$NAME.log
