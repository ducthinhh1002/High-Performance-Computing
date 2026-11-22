"""Khởi tạo torch.distributed dùng chung cho E3, E4, E5 (chạy bằng torchrun).

- GPU: backend nccl, mỗi rank một GPU (LOCAL_RANK).
- Không có GPU (kiểm tra code trên máy cá nhân): backend gloo, device cpu.
- Chạy không qua torchrun (RANK/WORLD_SIZE không có): world_size = 1.
"""
import os
import sys
import time
from datetime import timedelta

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
        # DIST_TIMEOUT_S: thời gian tối đa một collective được chờ trước khi báo lỗi (mặc định 600 s như PyTorch).
        dist.init_process_group("nccl" if use_cuda else "gloo", rank=rank, world_size=world,
                                timeout=timedelta(seconds=int(os.environ.get("DIST_TIMEOUT_S", "600"))))
    # Mặc định TF32 tắt cho matmul; cudnn mặc định bật nên tắt để khớp Đợt 1 (FP32 thật).
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return rank, world, local, device


def barrier():
    if dist.is_initialized():
        # NCCL không tự biết rank dùng GPU nào và đoán theo rank % số GPU: sai khi mỗi pod chỉ chạy một phần
        # số rank (rank toàn cục 1 nằm ở GPU cục bộ 0), làm hai rank tạo communicator khác nhau và treo.
        if dist.get_backend() == "nccl":
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cleanup():
    if dist.is_initialized():
        barrier()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        time.sleep(float(os.environ.get("TEARDOWN_SLEEP_S", "0")))   # chẩn đoán đua lúc thoát (Đợt 10)
        if os.environ.get("FAST_EXIT", "0") == "1":
            # Đợt 10: huỷ communicator điểm-điểm giữa hai pod có thể treo dù công việc đã xong. Mọi tệp kết quả đã đóng và
            # mọi rank đã qua barrier, nên thoát thẳng (không gọi destroy_process_group) là an toàn.
            sys.stdout.flush(); sys.stderr.flush()
            os._exit(0)
        dist.destroy_process_group()
