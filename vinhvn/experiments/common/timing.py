"""Đo thời gian và ghi kết quả dùng chung cho mọi thí nghiệm.

Quy tắc đo (theo tài liệu kế hoạch): warm-up trước, đồng bộ GPU trước và sau mỗi lần,
báo median kèm p25/p75. Số lần lặp tự co lại khi phép đo chậm (CPU) để không kéo dài Job.
"""
import csv
import json
import os
import platform
import statistics
import subprocess
import time

import torch


def sync_fn(device):
    if str(device).startswith("cuda"):
        return torch.cuda.synchronize
    return lambda: None


def bench(fn, device, warmup=3, min_reps=3, max_reps=20, budget_s=5.0):
    """Chạy fn nhiều lần, trả về dict thời gian (giây). Dừng khi đủ max_reps hoặc hết budget."""
    sync = sync_fn(device)
    for _ in range(warmup):
        fn()
    sync()
    ts, t_start = [], time.perf_counter()
    while len(ts) < max_reps:
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t0)
        if len(ts) >= min_reps and time.perf_counter() - t_start > budget_s:
            break
    q = statistics.quantiles(ts, n=4) if len(ts) >= 2 else [ts[0]] * 3
    return {"median_s": statistics.median(ts), "p25_s": q[0], "p75_s": q[2], "reps": len(ts)}


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"ERR {e!r}"


def env_info(device):
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "torch_threads": torch.get_num_threads(),
        "omp_env": os.environ.get("OMP_NUM_THREADS"),
        "cpu_model": sh("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2").strip(),
        "blas": "mkl" if "BLAS_INFO=mkl" in torch.__config__.show() else "other",
        "device": str(device),
        "pod": os.environ.get("HOSTNAME"),
        "net_env": {k: v for k, v in os.environ.items()
                    if k.startswith("NCCL_") or k in ("NET_LABEL", "NODE_NAME", "MASTER_ADDR", "NNODES")},
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if str(device).startswith("cuda"):
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["driver"] = sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader")
    return info


class CsvLog:
    """Ghi từng dòng kết quả ra CSV ngay khi đo xong (Job chết giữa chừng vẫn giữ được phần đã đo)."""

    def __init__(self, path, fields):
        self.path, self.fields = path, fields
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    def add(self, **row):
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fields, extrasaction="ignore").writerow(row)
        print("  " + ", ".join(f"{k}={_fmt(row.get(k))}" for k in self.fields), flush=True)


def _fmt(v):
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def dump_dir_to_log(out_dir, max_bytes=2_000_000):
    """In các file kết quả nhỏ ra log, để lấy về được bằng `kubectl logs`."""
    for name in sorted(os.listdir(out_dir)):
        p = os.path.join(out_dir, name)
        if os.path.isfile(p) and os.path.getsize(p) < max_bytes and name.endswith((".csv", ".json", ".txt")):
            print(f"\n----- BEGIN FILE {name} -----")
            with open(p, errors="replace") as f:
                print(f.read().rstrip("\n"))
            print(f"----- END FILE {name} -----", flush=True)
