"""Shared helpers for the benchmarks: devices, threads, timing, machine info, CSV output."""
import csv
import json
import os
import platform
import random
import statistics
import subprocess
import sys
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
    "mean_ms", "std_ms", "min_ms", "median_ms", "issue_ms", "warmup", "repeats",
    "gflops", "gbps", "order", "t_start", "t_end", "machine", "timestamp",
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

    issue_ms is the part until fn() returns, i.e. before the final synchronize. On a GPU
    that is the host's launch work (as long as the launches fit in the CUDA queue), so
    issue_ms ~ mean_ms means the run is bound by the CPU launching kernels, and
    issue_ms << mean_ms means it is bound by the GPU executing them.
    """
    for _ in range(warmup):
        fn()
    sync(device)
    times, issue = [], []
    t_start = time.time()
    for _ in range(repeat):
        sync(device)
        start = time.perf_counter()
        fn()
        issue.append((time.perf_counter() - start) * 1000.0)
        sync(device)
        times.append((time.perf_counter() - start) * 1000.0)
    return {
        "issue_ms": statistics.fmean(issue),
        "t_start": t_start,
        "t_end": time.time(),
        "mean_ms": statistics.fmean(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min_ms": min(times),
        "median_ms": statistics.median(times),
    }


def ordered(items, seed):
    """Measurement order. Seed 0 keeps it ascending; any other seed shuffles reproducibly.

    With an ascending order "larger batch" and "later in time" cannot be told apart, e.g.
    a GPU raising its clocks during the sweep would look like a batch-size effect.
    """
    items = list(items)
    if seed:
        random.Random(seed).shuffle(items)
    return items


def gpu_warmup(device, seconds):
    """Keep the GPU busy so it reaches its sustained clocks before anything is timed."""
    if device.type != "cuda" or seconds <= 0:
        return
    a = torch.randn(4096, 4096, device=device)
    b = torch.empty_like(a)
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        torch.mm(a, a, out=b)
        torch.cuda.synchronize(device)
    del a, b


class GpuClockLog:
    """Log GPU 0 clocks/power/temperature (gpu_sampler.py, every 20 ms) while the block runs."""

    def __init__(self, path, enabled, log=print):
        self.path, self.enabled, self.log, self.proc = Path(path), enabled, log, None

    def __enter__(self):
        if self.enabled:
            self.err = self.path.with_suffix(".err")
            self.err_fh = self.err.open("w", encoding="utf-8")
            self.proc = subprocess.Popen(
                [sys.executable, str(Path(__file__).with_name("gpu_sampler.py")), str(self.path), "20"],
                stderr=self.err_fh)
            time.sleep(1)
        return self

    def __exit__(self, *exc):
        if self.proc is None:
            return
        self.proc.terminate()
        self.proc.wait(timeout=10)
        self.err_fh.close()
        samples = max(len(self.path.read_text().splitlines()) - 1, 0) if self.path.exists() else 0
        error = self.err.read_text(encoding="utf-8").strip().splitlines()
        self.log(f"gpu clock log: {samples} samples" + (f" | ERROR: {error[-1]}" if error else ""))


_order = 0


def make_row(exp, impl, device, threads, size, stats, warmup, repeat, flops, nbytes, machine):
    global _order
    _order += 1
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
        "order": _order,
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
        "malloc_env": {k: v for k, v in os.environ.items() if k.startswith("MALLOC_")},
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
