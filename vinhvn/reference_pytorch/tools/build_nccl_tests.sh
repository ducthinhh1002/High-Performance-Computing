#!/usr/bin/env bash
# Build nccl-tests (chỉ CPU) vào hostPath, dùng đúng NCCL đi kèm PyTorch (pip nvidia-nccl-cu12)
# để kết quả khớp với NCCL mà các thí nghiệm PyTorch sẽ dùng.
set -euo pipefail
OUT=${OUT:-/hostdata/tools/nccl-tests}
NCCL_PKG=$(python -c "import nvidia.nccl, os; print(os.path.dirname(nvidia.nccl.__file__))")
echo "NCCL package: $NCCL_PKG"; ls "$NCCL_PKG"/lib "$NCCL_PKG"/include
python -c "import torch; print('torch', torch.__version__, 'nccl', torch.cuda.nccl.version())"
which nvcc && nvcc --version | tail -1

# pip chỉ có libnccl.so.2 → tạo NCCL_HOME với symlink libnccl.so để linker tìm được.
mkdir -p /tmp/nccl/lib
ln -sf "$NCCL_PKG"/include /tmp/nccl/include
ln -sf "$NCCL_PKG"/lib/libnccl.so.2 /tmp/nccl/lib/libnccl.so
ln -sf "$NCCL_PKG"/lib/libnccl.so.2 /tmp/nccl/lib/libnccl.so.2

# Image không có git: lấy commit mới nhất qua GitHub API rồi tải tarball đúng commit đó.
cd /tmp
# Ghim commit đã build lần đầu (20/11/2026); đặt NCCL_TESTS_COMMIT="" để lấy bản mới nhất.
COMMIT=${NCCL_TESTS_COMMIT-afd59abab774a6a4b492eb344bd0c845a0b7bb4d}
COMMIT=${COMMIT:-$(python -c "
import json, urllib.request as u
r = u.Request('https://api.github.com/repos/NVIDIA/nccl-tests/commits/master', headers={'User-Agent': 'hpc-dl-exp'})
print(json.load(u.urlopen(r, timeout=30))['sha'])")}
echo "nccl-tests commit: $COMMIT"
python -c "
import urllib.request as u, tarfile, io, sys
r = u.Request('https://codeload.github.com/NVIDIA/nccl-tests/tar.gz/$COMMIT', headers={'User-Agent': 'hpc-dl-exp'})
tarfile.open(fileobj=io.BytesIO(u.urlopen(r, timeout=60).read())).extractall('/tmp')"
cd /tmp/nccl-tests-$COMMIT
make -j16 MPI=0 CUDA_HOME=/usr/local/cuda NCCL_HOME=/tmp/nccl \
     NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"

mkdir -p "$OUT"
cp build/*_perf "$OUT"/
cat > "$OUT"/BUILD_INFO.txt <<EOF
nccl-tests commit: $COMMIT
built: $(date -u +%FT%TZ)
NCCL from: $NCCL_PKG (pip nvidia-nccl-cu12, same as torch)
gencode: sm_90 (H100)
run with: LD_LIBRARY_PATH=<site-packages>/nvidia/nccl/lib
EOF
ls -la "$OUT"; cat "$OUT"/BUILD_INFO.txt
