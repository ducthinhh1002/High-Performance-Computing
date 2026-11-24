"""Pilot 1 GPU cho phương án thực nghiệm Parallel and Distributed DL.

Kiểm tra: môi trường, topology GPU, im2col (E1), benchmark nhỏ CPU/GPU (E2),
khả năng tải dataset từ Internet, và ghi kết quả ra PVC.
Mỗi bước độc lập: một bước lỗi không làm dừng các bước sau.
"""
import json
import os
import platform
import statistics
import subprocess
import time
import traceback
from datetime import datetime, timezone

import torch
import torch.nn.functional as F

OUT_ROOT = os.environ.get("OUT_DIR", "/data/results/pilot")
DATA_DIR = os.environ.get("DATA_DIR", "/data/datasets")
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT = os.path.join(OUT_ROOT, RUN_ID)
os.makedirs(OUT, exist_ok=True)

summary = {"run_id": RUN_ID, "steps": {}}


def _cgroup_cpu_quota():
    # Bản rút gọn của experiments/common/runtime.py (pilot chỉ mount 1 file qua ConfigMap).
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    return None


# Số luồng phải khớp quota CPU của container, không phải số lõi của node.
_threads = int(os.environ.get("OMP_NUM_THREADS") or _cgroup_cpu_quota() or os.cpu_count())
torch.set_num_threads(_threads)
summary["threads"] = {"torch_threads": torch.get_num_threads(), "cgroup_quota": _cgroup_cpu_quota(),
                      "omp_env": os.environ.get("OMP_NUM_THREADS"), "os_cpu_count": os.cpu_count()}
print("threads:", summary["threads"], flush=True)


def step(name):
    def deco(fn):
        print(f"\n===== {name} =====", flush=True)
        t0 = time.time()
        try:
            res = fn()
            summary["steps"][name] = {"ok": True, "sec": round(time.time() - t0, 1), "result": res}
            print(f"[OK] {name} ({time.time() - t0:.1f}s)", flush=True)
        except Exception as e:  # noqa: BLE001
            summary["steps"][name] = {"ok": False, "sec": round(time.time() - t0, 1), "error": repr(e)}
            traceback.print_exc()
            print(f"[FAIL] {name}: {e!r}", flush=True)
        return fn
    return deco


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"ERR {e!r}"


def bench(fn, sync, warmup=3, reps=10):
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(reps):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


gpu_sync = torch.cuda.synchronize
cpu_sync = lambda: None  # noqa: E731


@step("env")
def _env():
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "nccl": ".".join(map(str, torch.cuda.nccl.version())) if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cpu_threads_torch": torch.get_num_threads(),
        "cpu_model": sh("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"),
        "nproc": sh("nproc"),
        "nvidia_smi_L": sh("nvidia-smi -L"),
        "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
    }
    with open(os.path.join(OUT, "torch_config.txt"), "w") as f:
        f.write(torch.__config__.show())
    with open(os.path.join(OUT, "nvidia_smi_topo.txt"), "w") as f:
        f.write(sh("nvidia-smi topo -m"))
    print(json.dumps(info, indent=2))
    return info


@step("im2col_check")
def _im2col():
    torch.manual_seed(0)
    x = torch.randn(4, 1, 224, 224, device="cuda")
    w = torch.randn(96, 1, 11, 11, device="cuda")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    cols = F.unfold(x, kernel_size=11, stride=4)               # (B, 121, L)
    out = (w.view(96, -1) @ cols).view(4, 96, 54, 54)
    ref = F.conv2d(x, w, stride=4)
    err = (out - ref).abs().max().item()
    res = {"unfold_shape_per_image": [cols.shape[1], cols.shape[2]],
           "patches_per_image": cols.shape[2], "max_abs_err": err, "match": err < 1e-3}
    print(res)
    return res


@step("bench_conv_cpu_vs_gpu")
def _conv():
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    rows = []
    w_cpu = torch.randn(96, 1, 11, 11)
    w_gpu = w_cpu.cuda()
    for b in [1, 8, 32, 128]:
        x_cpu = torch.randn(b, 1, 224, 224)
        x_gpu = x_cpu.cuda()
        t_cpu = bench(lambda: F.conv2d(x_cpu, w_cpu, stride=4), cpu_sync, warmup=2, reps=5)
        t_gpu = bench(lambda: F.conv2d(x_gpu, w_gpu, stride=4), gpu_sync)
        rows.append({"batch": b, "cpu_ms": t_cpu * 1e3, "gpu_ms": t_gpu * 1e3, "speedup": t_cpu / t_gpu})
        print(rows[-1], flush=True)
    with open(os.path.join(OUT, "bench_conv.csv"), "w") as f:
        f.write("batch,cpu_ms,gpu_ms,speedup\n")
        for r in rows:
            f.write(f"{r['batch']},{r['cpu_ms']:.4f},{r['gpu_ms']:.4f},{r['speedup']:.2f}\n")
    return rows


@step("bench_matmul_cpu_vs_gpu")
def _mm():
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = []
    for n in [1024, 4096]:
        a_cpu, b_cpu = torch.randn(n, n), torch.randn(n, n)
        a_gpu, b_gpu = a_cpu.cuda(), b_cpu.cuda()
        t_cpu = bench(lambda: a_cpu @ b_cpu, cpu_sync, warmup=1, reps=3)
        t_gpu = bench(lambda: a_gpu @ b_gpu, gpu_sync)
        tflops = 2 * n**3 / t_gpu / 1e12
        rows.append({"n": n, "cpu_ms": t_cpu * 1e3, "gpu_ms": t_gpu * 1e3,
                     "speedup": t_cpu / t_gpu, "gpu_fp32_tflops": tflops})
        print(rows[-1], flush=True)
    with open(os.path.join(OUT, "bench_matmul.csv"), "w") as f:
        f.write("n,cpu_ms,gpu_ms,speedup,gpu_fp32_tflops\n")
        for r in rows:
            f.write(f"{r['n']},{r['cpu_ms']:.4f},{r['gpu_ms']:.4f},{r['speedup']:.2f},{r['gpu_fp32_tflops']:.2f}\n")
    return rows


@step("dataset_readonly_mount")
def _data():
    # Không tải trong Job GPU: dataset phải được Job dataset-prep chuẩn bị sẵn
    # (npz từ Hugging Face, xem experiments/common/data.py).
    import numpy as np
    d = os.path.join(DATA_DIR, "cifar10")
    t0 = time.time()
    with np.load(os.path.join(d, "train.npz")) as z:
        x, y = z["x"], z["y"]
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    res = {"cifar10_train_shape": list(x.shape), "labels": len(y),
           "load_sec": round(time.time() - t0, 2), "revision": meta["revision"],
           "readonly": not os.access(DATA_DIR, os.W_OK)}
    print(res)
    return res


@step("pvc_write_speed")
def _pvc():
    path = os.path.join(OUT, "io_test.bin")
    buf = os.urandom(64 * 1024 * 1024)
    t0 = time.time()
    with open(path, "wb") as f:
        for _ in range(8):
            f.write(buf)
        f.flush()
        os.fsync(f.fileno())
    mbps = 512 / (time.time() - t0)
    os.remove(path)
    return {"write_MBps": round(mbps, 1)}


with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2, default=str)
# In toàn bộ file kết quả nhỏ ra log, để lấy được kể cả khi không có PVC.
for name in sorted(os.listdir(OUT)):
    p = os.path.join(OUT, name)
    if os.path.isfile(p) and os.path.getsize(p) < 200_000:
        print(f"\n----- BEGIN FILE {name} -----")
        with open(p, errors="replace") as f:
            print(f.read())
        print(f"----- END FILE {name} -----", flush=True)
ok = sum(s["ok"] for s in summary["steps"].values())
print(f"\n===== PILOT DONE: {ok}/{len(summary['steps'])} steps OK, results in {OUT} =====", flush=True)
