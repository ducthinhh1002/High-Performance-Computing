"""Khởi tạo torch.distributed dùng chung cho E3, E4, E5 (chạy bằng torchrun).

- GPU: backend nccl, mỗi rank một GPU (LOCAL_RANK).
- Không có GPU (kiểm tra code trên máy cá nhân): backend gloo, device cpu.
- Chạy không qua torchrun (RANK/WORLD_SIZE không có): world_size = 1.
"""
import os

import torch
import torch.distributed as dist


def init_dist():
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    use_cuda = torch.cuda.is_available() and os.environ.get("FORCE_CPU", "0") != "1"
    if use_cuda:
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")
    if world > 1 or os.environ.get("FORCE_INIT", "0") == "1":
        dist.init_process_group("nccl" if use_cuda else "gloo", rank=rank, world_size=world)
    # Mặc định TF32 tắt cho matmul; cudnn mặc định bật nên tắt để khớp Đợt 1 (FP32 thật).
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return rank, world, local, device


def barrier():
    if dist.is_initialized():
        dist.barrier()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cleanup():
    if dist.is_initialized():
        dist.barrier()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dist.destroy_process_group()
