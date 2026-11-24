#!/usr/bin/env bash
# Kiểm tra tính đúng: loss và chuẩn gradient toàn cục của bản CUDA (mọi cách chia mô hình) phải khớp bản C tuần tự.
# Dùng: scripts/check_correctness.sh        (cần đã `make`; cần >= 4 GPU)   Biến: H, L, B, TOL (sai số tương đối, mặc định 1e-4)
cd "$(dirname "$0")/.."
H=${H:-128}; L=${L:-4}; B=${B:-64}; TOL=${TOL:-1e-4}
field() { echo "$1" | sed -E "s/.*$2=([^ ]+).*/\1/"; }
ref=$(bin/mlp_seq --H "$H" --L "$L" --B "$B" --check)
rl=$(field "$ref" loss); rg=$(field "$ref" gradnorm)
echo "tuần tự (C): loss=$rl gradnorm=$rg   [H=$H L=$L B=$B, ngưỡng sai số tương đối $TOL]"
fail=0; total=0
check() {   # check <đối số của mlp_cuda...>
  out=$(bin/mlp_cuda "$@" --H "$H" --L "$L" --B "$B" --check 2>&1)
  l=$(field "$out" loss); g=$(field "$out" gradnorm)
  if [ -z "$l" ] || [ -z "$g" ] || [ "$l" = "$out" ]; then echo "LỖI  $*: $out" | head -3; fail=$((fail+1)); total=$((total+1)); return; fi
  verdict=$(awk -v l="$l" -v g="$g" -v rl="$rl" -v rg="$rg" -v tol="$TOL" 'BEGIN {
      el = (l-rl)/rl; if (el<0) el=-el; eg = (g-rg)/rg; if (eg<0) eg=-eg;
      printf "%s loss_err=%.1e grad_err=%.1e", (el<tol && eg<tol) ? "ĐÚNG" : "SAI ", el, eg }')
  echo "$verdict  $*"
  case "$verdict" in ĐÚNG*) ;; *) fail=$((fail+1));; esac
  total=$((total+1))
}
check --mode pipeline --gpus 1 --micro 1
for S in 2 4; do for M in 1 2 4 8; do check --mode pipeline --gpus $S --micro $M; done; done
for T in 1 2 4; do check --mode tensor --gpus $T; done
echo "kết quả: $((total-fail))/$total cấu hình khớp bản tuần tự"
[ "$fail" -eq 0 ]
