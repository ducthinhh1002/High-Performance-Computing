"""E1: Convolution và GEMM có phải bottleneck?

1. im2col: tự cài conv bằng unfold + matmul, so khớp với conv2d, và đo riêng thời gian
   unfold và matmul (bài §3.1 khẳng định matmul là phần tốn kém nhất).
2. Phân rã thời gian theo nhóm phép tính (conv, gemm, norm, act/pool, optimizer, …) bằng
   torch.profiler cho 3 mô hình, ở hai chế độ: chỉ forward, và cả bước huấn luyện.
   Dùng self time (không cộng dồn op lồng nhau). Backward+optimizer = step − forward.
3. (GPU) Liệt kê kernel cuDNN thực sự chạy cho lớp conv của bài, để xem có dùng im2col
   hay implicit GEMM / Winograd / FFT.
"""
import argparse
import os
import re
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from common.models import build, n_params
from common.runtime import configure_threads
from common.timing import CsvLog, bench, dump_dir_to_log, env_info, write_json

CATEGORIES = [  # thứ tự quan trọng: khớp mẫu đầu tiên
    ("conv", r"conv|cudnn_convolution|mkldnn_convolution"),
    ("gemm", r"addmm|\bmm\b|aten::mm|bmm|matmul|linear|gemm"),
    ("norm", r"batch_norm|layer_norm"),
    ("act_pool_dropout", r"relu|threshold|max_pool|avg_pool|adaptive|dropout|pool"),
    ("loss_softmax", r"cross_entropy|nll_loss|log_softmax|softmax"),
    ("optimizer", r"_foreach|sgd|optimizer|lerp|addcmul|addcdiv"),
    ("elementwise_add", r"aten::add|aten::mul|aten::div|aten::sub"),
    ("memory_copy", r"copy_|aten::to|contiguous|clone|empty|zero_|fill_|cat"),
]


def categorize(name):
    low = name.lower()
    for cat, pat in CATEGORIES:
        if re.search(pat, low):
            return cat
    return "other"


def im2col_section(dev, batches, log, out):
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    w = torch.randn(96, 1, 11, 11, device=dev)
    wm = w.view(96, -1)                                   # 96 × 121
    for b in batches:
        x = torch.randn(b, 1, 224, 224, device=dev)
        cols = F.unfold(x, kernel_size=11, stride=4)      # b × 121 × 2916
        out_im2col = (wm @ cols).view(b, 96, 54, 54)
        ref = F.conv2d(x, w, stride=4)
        err = (out_im2col - ref).abs().max().item()
        t_unfold = bench(lambda: F.unfold(x, kernel_size=11, stride=4), dev)
        t_mm = bench(lambda: wm @ cols, dev)
        t_conv = bench(lambda: F.conv2d(x, w, stride=4), dev)
        t_total = t_unfold["median_s"] + t_mm["median_s"]
        log.add(device=dev, batch=b, cols_shape=f"{cols.shape[1]}x{cols.shape[2]}", max_abs_err=err,
                unfold_ms=t_unfold["median_s"] * 1e3, matmul_ms=t_mm["median_s"] * 1e3,
                im2col_total_ms=t_total * 1e3, matmul_share=t_mm["median_s"] / t_total,
                conv2d_ms=t_conv["median_s"] * 1e3, conv2d_vs_im2col=t_conv["median_s"] / t_total)
        del x, cols, out_im2col, ref


def run_profile(fn, dev, steps):
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev == "cuda" else [])
    with profile(activities=acts) as prof:
        for _ in range(steps):
            fn()
        if dev == "cuda":
            torch.cuda.synchronize()
    return prof


def self_time_us(evt, dev):
    if dev == "cuda":
        return getattr(evt, "self_device_time_total", None) or getattr(evt, "self_cuda_time_total", 0)
    return evt.self_cpu_time_total


def breakdown(prof, dev, steps):
    """Tổng self time theo nhóm (ms mỗi bước). Trên GPU chỉ tính op cấp aten để không đếm trùng kernel."""
    acc = defaultdict(float)
    for e in prof.key_averages():
        if dev == "cuda" and not e.key.startswith("aten::"):
            continue
        if dev == "cpu" and not (e.key.startswith("aten::") or "Optimizer" in e.key):
            continue
        t = self_time_us(e, dev)
        if t:
            acc[categorize(e.key)] += t / 1e3 / steps
    return acc


def top_kernels(prof, n=15):
    rows = []
    for e in prof.key_averages():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            rows.append((e.key, (getattr(e, "self_device_time_total", None) or e.self_cuda_time_total), e.count))
    rows.sort(key=lambda r: -r[1])
    return rows[:n]


def profile_section(dev, models, steps, log_ops, log_phase, out):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    kernels = {}
    for name, b in models:
        model, shape, ncls = build(name, dev)
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, foreach=True)
        loss_fn = nn.CrossEntropyLoss()
        x = torch.randn(b, *shape, device=dev)
        y = torch.randint(0, ncls, (b,), device=dev)

        def forward():
            loss_fn(model(x), y)

        def step():
            opt.zero_grad(set_to_none=True)
            loss_fn(model(x), y).backward()
            opt.step()

        # Thời gian thật (không có overhead profiler) cho từng pha.
        t_fwd = bench(forward, dev, warmup=3, budget_s=8.0)["median_s"]
        t_step = bench(step, dev, warmup=3, budget_s=8.0)["median_s"]
        log_phase.add(model=name, device=dev, batch=b, params=n_params(model),
                      forward_ms=t_fwd * 1e3, step_ms=t_step * 1e3,
                      backward_opt_ms=(t_step - t_fwd) * 1e3, forward_share=t_fwd / t_step)

        res = {}
        for mode, fn in (("forward", forward), ("step", step)):
            prof = run_profile(fn, dev, steps)
            res[mode] = breakdown(prof, dev, steps)
            if dev == "cuda" and mode == "forward" and name == "alexnet_gray":
                kernels[name] = top_kernels(prof)
        cats = sorted(set(res["forward"]) | set(res["step"]))
        tot_step = sum(res["step"].values())
        for c in cats:
            f, s = res["forward"].get(c, 0.0), res["step"].get(c, 0.0)
            log_ops.add(model=name, device=dev, batch=b, category=c, forward_ms=f,
                        backward_opt_ms=max(s - f, 0.0), step_ms=s, step_share=s / tot_step if tot_step else 0)
        del model, opt, x, y
        if dev == "cuda":
            torch.cuda.empty_cache()

    if kernels:
        with open(os.path.join(out, "e1_alexnet_forward_top_kernels.csv"), "w") as f:
            f.write("kernel,self_device_us_total,count\n")
            for k, t, c in kernels["alexnet_gray"]:
                f.write(f"\"{k}\",{t:.1f},{c}\n")
        print("top GPU kernels (alexnet forward):")
        for k, t, c in kernels["alexnet_gray"]:
            print(f"  {t/1e3:8.2f} ms  x{c:<4d} {k[:110]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], required=True)
    ap.add_argument("--out", default=os.environ.get("OUT_DIR", "/data/results") + "/e1")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    threads = configure_threads()
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = os.path.join(a.out, f"{run_id}_{a.device}{'_quick' if a.quick else ''}")
    os.makedirs(out, exist_ok=True)
    batches = [1, 8] if a.quick else [1, 8, 32, 128, 256]
    models = ([("alexnet_gray", 4), ("resnet18_cifar", 8), ("vgg16_cifar", 4)] if a.quick
              else [("alexnet_gray", 128), ("resnet18_cifar", 128), ("vgg16_cifar", 128)])
    steps = 2 if a.quick else 5
    meta = {"experiment": "E1", "env": env_info(a.device), "threads": threads, "quick": a.quick,
            "im2col_batches": batches, "profile_models": models, "profile_steps": steps,
            "categories": CATEGORIES, "tf32": False}
    write_json(os.path.join(out, "meta.json"), meta)
    print("E1", meta, flush=True)
    t0 = time.time()

    print("\n== im2col", flush=True)
    log_i = CsvLog(os.path.join(out, "e1_im2col.csv"),
                   ["device", "batch", "cols_shape", "max_abs_err", "unfold_ms", "matmul_ms", "im2col_total_ms",
                    "matmul_share", "conv2d_ms", "conv2d_vs_im2col"])
    im2col_section(a.device, batches, log_i, out)

    print("\n== profile", flush=True)
    log_p = CsvLog(os.path.join(out, "e1_phase.csv"),
                   ["model", "device", "batch", "params", "forward_ms", "step_ms", "backward_opt_ms", "forward_share"])
    log_o = CsvLog(os.path.join(out, "e1_ops.csv"),
                   ["model", "device", "batch", "category", "forward_ms", "backward_opt_ms", "step_ms", "step_share"])
    profile_section(a.device, models, steps, log_o, log_p, out)

    meta["elapsed_s"] = round(time.time() - t0, 1)
    write_json(os.path.join(out, "meta.json"), meta)
    print(f"\nE1 DONE ({a.device}) in {meta['elapsed_s']}s → {out}", flush=True)
    dump_dir_to_log(out)


if __name__ == "__main__":
    main()
