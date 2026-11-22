"""E2: GPU so với CPU.

Tái lập (bài gốc §3.2, Hình 1–4):
  conv_forward  lớp conv của bài (96 kernel 11×11 stride 4, ảnh 1×224×224) theo batch size
  matvec        lớp FC như bài mô tả: nhân ma trận n×n với vector (O(n²), Hình 3–4)
Mở rộng:
  matmul        nhân ma trận n×n × n×n (FC với batch), kèm TFLOPS
  train_step    forward+backward+optimizer của 3 mô hình theo batch size (ảnh/giây)
  precision     (chỉ GPU) FP32 / TF32 / BF16 cho matmul và train_step
  h2d           (chỉ GPU) băng thông CPU→GPU, bộ nhớ thường so với pinned

Chạy riêng cho từng thiết bị: --device cpu (Job chỉ CPU) và --device cuda (Job 1 GPU),
ghép kết quả sau. Mặc định tắt TF32 để so công bằng với CPU FP32.
"""
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.models import MODELS, build, n_params
from common.runtime import configure_threads
from common.timing import CsvLog, bench, dump_dir_to_log, env_info, write_json

FULL = {
    "conv_batches": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    "mat_ns": [256, 512, 1024, 2048, 4096, 8192, 16384],
    "train_batches": {"alexnet_gray": [1, 16, 64, 128, 256],
                      "resnet18_cifar": [32, 128, 512],
                      "vgg16_cifar": [32, 128, 512]},
    "h2d_mb": [1, 4, 16, 64, 256, 1024],
}
QUICK = {
    "conv_batches": [1, 8],
    "mat_ns": [256, 1024],
    "train_batches": {"alexnet_gray": [4], "resnet18_cifar": [16], "vgg16_cifar": [8]},
    "h2d_mb": [1, 16],
}


def set_precision(mode):
    """fp32 = TF32 tắt; tf32 = bật TF32 cho matmul và cuDNN; bf16 dùng autocast riêng."""
    tf32 = mode == "tf32"
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32


def conv_forward(dev, cfg, log):
    w = torch.randn(96, 1, 11, 11, device=dev)
    for b in cfg["conv_batches"]:
        x = torch.randn(b, 1, 224, 224, device=dev)
        r = bench(lambda: F.conv2d(x, w, stride=4), dev)
        log.add(test="conv_forward", device=dev, precision="fp32", size=b, **r,
                throughput=b / r["median_s"], unit="img/s")
        del x


def matvec(dev, cfg, log):
    for n in cfg["mat_ns"]:
        a, v = torch.randn(n, n, device=dev), torch.randn(n, device=dev)
        r = bench(lambda: torch.mv(a, v), dev)
        gbps = (n * n * 4) / r["median_s"] / 1e9  # bị giới hạn bởi băng thông bộ nhớ: đọc cả ma trận
        log.add(test="matvec", device=dev, precision="fp32", size=n, **r, throughput=gbps, unit="GB/s")
        del a, v


def matmul(dev, cfg, log, precision="fp32"):
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    for n in cfg["mat_ns"]:
        a, b = torch.randn(n, n, device=dev, dtype=dtype), torch.randn(n, n, device=dev, dtype=dtype)
        r = bench(lambda: a @ b, dev, budget_s=8.0)
        log.add(test="matmul", device=dev, precision=precision, size=n, **r,
                throughput=2 * n**3 / r["median_s"] / 1e12, unit="TFLOPS")
        del a, b


def train_step(dev, cfg, log, precision="fp32"):
    for name, batches in cfg["train_batches"].items():
        model, shape, ncls = build(name, dev)
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, foreach=True)
        loss_fn = nn.CrossEntropyLoss()
        for b in batches:
            x = torch.randn(b, *shape, device=dev)
            y = torch.randint(0, ncls, (b,), device=dev)

            def fwd():
                if precision == "bf16":
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        return loss_fn(model(x), y)
                return loss_fn(model(x), y)

            def forward_only():
                with torch.no_grad():
                    fwd()

            def step():
                opt.zero_grad(set_to_none=True)
                fwd().backward()
                opt.step()

            for phase, fn in (("forward", forward_only), ("step", step)):
                r = bench(fn, dev, warmup=2, budget_s=8.0)
                log.add(test=f"train_{phase}", device=dev, precision=precision, size=b, model=name,
                        params=n_params(model), **r, throughput=b / r["median_s"], unit="img/s")
            del x, y
        del model, opt
        if dev == "cuda":
            torch.cuda.empty_cache()


def h2d(cfg, log):
    for mb in cfg["h2d_mb"]:
        n = mb * 1024 * 1024 // 4
        for pinned in (False, True):
            src = torch.empty(n, dtype=torch.float32, pin_memory=pinned)
            dst = torch.empty(n, dtype=torch.float32, device="cuda")
            r = bench(lambda: dst.copy_(src, non_blocking=pinned), "cuda")
            log.add(test="h2d_pinned" if pinned else "h2d_pageable", device="cuda", precision="-",
                    size=mb, **r, throughput=mb * 1024 * 1024 / r["median_s"] / 1e9, unit="GB/s")
            del src, dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], required=True)
    ap.add_argument("--out", default=os.environ.get("OUT_DIR", "/data/results") + "/e2")
    ap.add_argument("--quick", action="store_true", help="kích thước nhỏ, để kiểm tra code")
    a = ap.parse_args()

    threads = configure_threads()
    cfg = QUICK if a.quick else FULL
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = os.path.join(a.out, f"{run_id}_{a.device}{'_quick' if a.quick else ''}")
    os.makedirs(out, exist_ok=True)
    meta = {"experiment": "E2", "env": env_info(a.device), "threads": threads, "quick": a.quick,
            "config": cfg, "models": {k: list(v[1]) for k, v in MODELS.items()}}
    write_json(os.path.join(out, "meta.json"), meta)
    print("E2", meta, flush=True)

    torch.backends.cudnn.benchmark = True
    fields = ["test", "device", "precision", "model", "params", "size", "median_s", "p25_s", "p75_s",
              "reps", "throughput", "unit"]
    log = CsvLog(os.path.join(out, "e2_results.csv"), fields)
    t0 = time.time()

    set_precision("fp32")
    print("\n== conv_forward (Hình 1–2)", flush=True); conv_forward(a.device, cfg, log)
    print("\n== matvec (Hình 3–4)", flush=True); matvec(a.device, cfg, log)
    print("\n== matmul", flush=True); matmul(a.device, cfg, log)
    print("\n== train_step", flush=True); train_step(a.device, cfg, log)

    if a.device == "cuda":
        for p in ("tf32", "bf16"):
            set_precision(p)
            print(f"\n== precision {p}", flush=True)
            matmul("cuda", cfg, log, precision=p)
            train_step("cuda", cfg, log, precision=p)
        set_precision("fp32")
        print("\n== h2d", flush=True); h2d(cfg, log)

    meta["elapsed_s"] = round(time.time() - t0, 1)
    write_json(os.path.join(out, "meta.json"), meta)
    print(f"\nE2 DONE ({a.device}) in {meta['elapsed_s']}s → {out}", flush=True)
    dump_dir_to_log(out)


if __name__ == "__main__":
    main()
