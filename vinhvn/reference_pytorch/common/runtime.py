"""Cấu hình runtime dùng chung: số luồng CPU khớp với quota của container.

Trong pod, os.cpu_count() và nproc trả về số lõi của cả node (128), không phải
quota của container (vd 16). PyTorch mặc định mở 64 luồng → tranh nhau quota,
bị CFS throttle, số đo CPU sai. Luôn gọi configure_threads() đầu mỗi script.
"""
import os


def cgroup_cpu_quota():
    """Số CPU theo cgroup (v2: cpu.max, v1: cfs_quota/period), None nếu không giới hạn."""
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            quota = int(f.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            period = int(f.read())
        if quota > 0:
            return max(1, quota // period)
    except (OSError, ValueError):
        pass
    return None


def configure_threads(n=None):
    """Đặt số luồng intra-op của PyTorch. Ưu tiên: tham số n > OMP_NUM_THREADS > quota cgroup."""
    import torch

    if n is None:
        env = os.environ.get("OMP_NUM_THREADS")
        n = int(env) if env else (cgroup_cpu_quota() or os.cpu_count())
    torch.set_num_threads(n)
    return {"torch_threads": torch.get_num_threads(), "cgroup_quota": cgroup_cpu_quota(),
            "omp_env": os.environ.get("OMP_NUM_THREADS"), "os_cpu_count": os.cpu_count()}
