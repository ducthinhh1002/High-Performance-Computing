#!/usr/bin/env bash
# So bản C tuần tự với PyTorch autograd fp64 (công cụ kiểm tra, cần python3 + numpy + torch). Cần đã `make seq`.
cd "$(dirname "$0")/.."
for cfg in "64 2 32" "128 3 48"; do
  set -- $cfg
  python3 scripts/verify_seq_autograd.py "$1" "$2" "$3"
  bin/mlp_seq --H "$1" --L "$2" --B "$3" --check
done
