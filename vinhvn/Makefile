# Biên dịch hai chương trình của phần model parallelism.
#   make seq    bản tuần tự (C thuần)          -> bin/mlp_seq
#   make cuda   bản song song (CUDA, cần nvcc) -> bin/mlp_cuda
#   make        cả hai
# ARCH: kiến trúc GPU (mặc định sm_90 cho H100; sm_80 cho A100, sm_75 cho T4).
CC    ?= cc
NVCC  ?= nvcc
ARCH  ?= sm_90
CFLAGS ?= -O3 -march=native -Wall -Wextra

all: seq cuda

bin:
	mkdir -p bin

seq: bin/mlp_seq
bin/mlp_seq: src/mlp_seq.c src/common.h | bin
	$(CC) $(CFLAGS) -o $@ src/mlp_seq.c -lm

cuda: bin/mlp_cuda
bin/mlp_cuda: src/mlp_cuda.cu src/common.h | bin
	$(NVCC) -O3 -arch=$(ARCH) -std=c++14 -o $@ src/mlp_cuda.cu

clean:
	rm -rf bin
.PHONY: all seq cuda clean
