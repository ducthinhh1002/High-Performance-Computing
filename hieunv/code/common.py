"""Shared helpers for the benchmarks: devices, threads, timing, machine info, CSV output."""
import csv
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent

# torch picks one intra-op thread per physical core by default. Capture it before
# any benchmark calls torch.set_num_threads(), so "max" always means this value.
DEFAULT_THREADS = torch.get_num_threads()

CSV_FIELDS = [
    "exp", "impl", "device", "threads", "size",
    "mean_ms", "std_ms", "min_ms", "median_ms", "warmup", "repeats",
    "gflops", "gbps", "machine", "timestamp",
]


def parse_ints(text):
    return [int(v) for v in text.split(",") if v.strip()]


def resolve_threads(spec):
    """'1,max' -> [1, DEFAULT_THREADS] (duplicates removed, order kept)."""
    values = []
    for item in spec.split(","):
        item = item.strip()
        if item == "max":
            values.append(DEFAULT_THREADS)
        elif item:
            values.append(int(item))
    return list(dict.fromkeys(values))


def get_device(name):
    if name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is not available on this machine.")
        # The paper's GPU (Kepler) only had plain FP32. Turn off TF32 shortcuts so
        # newer GPUs compute in true FP32 too.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = True
    elif name != "cpu":
        raise SystemExit(f"Unknown device '{name}' (use cpu or cuda).")
    return torch.device(name)


def available_devices():
    return ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def bench(fn, device, warmup=3, repeat=10):
    """Run fn `warmup` times untimed, then `repeat` timed runs (wall clock, ms).

    GPU work is asynchronous, so the device is synchronized before starting and
    after finishing each timed run: the time covers the full kernel execution.
    """
    for _ in range(warmup):
        fn()
    sync(device)
    times = []
    for _ in range(repeat):
        sync(device)
        start = time.perf_counter()
        fn()
        sync(device)
        times.append((time.perf_counter() - start) * 1000.0)
    return {
        "mean_ms": statistics.fmean(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min_ms": min(times),
        "median_ms": statistics.median(times),
    }


def make_row(exp, impl, device, threads, size, stats, warmup, repeat, flops, nbytes, machine):
    seconds = stats["mean_ms"] / 1000.0
    return {
        "exp": exp,
        "impl": impl,
        "device": device.type,
        "threads": threads,
        "size": size,
        **{k: round(v, 6) for k, v in stats.items()},
        "warmup": warmup,
        "repeats": repeat,
        "gflops": round(flops / seconds / 1e9, 3) if flops else "",
        "gbps": round(nbytes / seconds / 1e9, 3) if nbytes else "",
        "machine": machine,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def append_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def cpu_name():
    system = platform.system()
    if system == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    elif system == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def env_info(machine):
    info = {
        "machine": machine,
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu": cpu_name(),
        "cpu_logical_cores": os.cpu_count(),
        "torch_default_threads": DEFAULT_THREADS,
        "mkl": torch.backends.mkl.is_available(),
        "mkldnn": torch.backends.mkldnn.is_available(),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update({
            "gpu": props.name,
            "gpu_count": torch.cuda.device_count(),
            "gpu_memory_gb": round(props.total_memory / 1024**3, 1),
            "gpu_sm_count": props.multi_processor_count,
            "gpu_compute_capability": f"{props.major}.{props.minor}",
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        })
    return info


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
