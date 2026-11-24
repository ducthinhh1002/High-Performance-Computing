#!/usr/bin/env python3
"""Kiểm tra độc lập bản tuần tự bằng PyTorch autograd (fp64). CHỈ LÀ CÔNG CỤ KIỂM TRA, không thuộc mã nộp
(bài nộp không dùng PyTorch). Dựng lại đúng trọng số và đầu vào từ hàm băm splitmix64 của src/common.h, tính loss và
chuẩn gradient toàn cục bằng autograd, để so với `bin/mlp_seq --check`.
Dùng:  python3 scripts/verify_seq_autograd.py <H> <L> <B>      (cần numpy và torch)
       scripts/verify_seq_autograd.sh                          (chạy vài cấu hình và so với bản C) """
import sys

import numpy as np
import torch

H, L, B, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), 1
F = 4 * H
M64 = np.uint64
def mix64(z):
    z = (z + M64(0x9E3779B97F4A7C15))
    z = (z ^ (z >> M64(30))) * M64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> M64(27))) * M64(0x94D049BB133111EB)
    return z ^ (z >> M64(31))
def urand(tag, layer, idx):
    idx = idx.astype(np.uint64)
    key = idx + (M64(layer) << M64(36)) + (M64(tag) << M64(56))
    h = mix64(key ^ (M64(seed) * M64(0xD1B54A32D192ED03)))
    m = (h >> M64(40)).astype(np.uint32).astype(np.float32)
    return m * np.float32(1.0 / 8388608.0) - np.float32(1.0)
with np.errstate(over="ignore"):
    s1 = np.float32(1.0) / np.sqrt(np.float32(H)); s2 = np.float32(1.0) / np.sqrt(np.float32(F)); sb = np.float32(0.1)
    W1 = [torch.tensor((s1 * urand(1, l, np.arange(H * F))).reshape(H, F), dtype=torch.float64, requires_grad=True) for l in range(L)]
    B1 = [torch.tensor(sb * urand(2, l, np.arange(F)), dtype=torch.float64, requires_grad=True) for l in range(L)]
    W2 = [torch.tensor((s2 * urand(3, l, np.arange(F * H))).reshape(F, H), dtype=torch.float64, requires_grad=True) for l in range(L)]
    X = torch.tensor(urand(4, 0, np.arange(B * H)).reshape(B, H), dtype=torch.float64)
x = X
for l in range(L):
    x = x + torch.relu(x @ W1[l] + B1[l]) @ W2[l]
loss = 0.5 * (x ** 2).mean()
loss.backward()
g2 = sum((p.grad ** 2).sum() for p in W1 + B1 + W2)
print(f"REF   impl=autograd-fp64 H={H} L={L} B={B} loss={loss.item():.9e} gradnorm={g2.sqrt().item():.9e}")
