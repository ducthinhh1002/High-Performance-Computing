"""E7: kiểm chứng điểm bất thường của Đợt 1: VGG-16 với BF16 chậm hơn FP32 khoảng 2 lần.

Giả thuyết H1 (chưa kiểm chứng): backward của AdaptiveAvgPool2d (phóng 1×1 lên 7×7 với ảnh
32×32) dùng atomic add trên BF16 nên chậm. Giả thuyết đối lập H2: layout NCHW làm cuDNN
chèn phép chuyển vị khi chạy BF16. Mỗi biến thể đổi đúng một yếu tố:

  base            vgg16_cifar như Đợt 1 (autocast BF16 so với FP32)
  no_avgpool      thay AdaptiveAvgPool2d bằng Flatten, FC đầu 512→4096     → kiểm H1
  channels_last   model + đầu vào NHWC                                     → kiểm H2
  pure_bf16       model.to(bfloat16), không autocast                        → loại trừ chi phí cast
  resnet18        đối chứng (không có AdaptiveAvgPool 1×1→7×7)

Đo thời gian bước huấn luyện (forward+backward+optimizer) và tách forward/backward/optimizer
bằng cuda event; thêm torch.profiler lấy top kernel CUDA của biến thể base ở FP32 và BF16.
Cần 1 GPU.
"""
import argparse
import os
from datetime import datetime, timezone

import torch
import torch.nn as nn

from common.models import build
from common.timing import CsvLog, bench, dump_dir_to_log, env_info, write_json

STEP_FIELDS = ["variant", "precision", "batch", "step_s", "fwd_s", "bwd_s", "opt_s", "img_s"]
PROF_FIELDS = ["variant", "precision", "rank", "kernel", "self_cuda_ms", "calls"]


def make(variant, dev):
    name = "resnet18_cifar" if variant == "resnet18" else "vgg16_cifar"
    model, shape, ncls = build(name, dev)
    if variant == "no_avgpool":
        model.avgpool = nn.Flatten()
        model.classifier[0] = nn.Linear(512, 4096).to(dev)
    if variant == "channels_last":
        model = model.to(memory_format=torch.channels_last)
    return model, shape, ncls


def make_step(model, shape, ncls, batch, precision, variant, dev):
    if precision == "pure_bf16":
        model = model.to(torch.bfloat16)
    x = torch.randn(batch, *shape, device=dev)
    if precision == "pure_bf16":
        x = x.to(torch.bfloat16)
    if variant == "channels_last":
        x = x.contiguous(memory_format=torch.channels_last)
    y = torch.randint(0, ncls, (batch,), device=dev)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, foreach=True)
    loss_fn = nn.CrossEntropyLoss()
    use_ac = precision == "bf16"

    def fwd():
        if use_ac:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return loss_fn(model(x), y)
        return loss_fn(model(x), y)

    def step():
        opt.zero_grad(set_to_none=True)
        fwd().backward()
        opt.step()

    def phases(reps=10):
        """Trung bình forward / backward / optimizer bằng cuda event."""
        tot = [0.0, 0.0, 0.0]
        for _ in range(reps):
            opt.zero_grad(set_to_none=True)
            e = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            e[0].record()
            loss = fwd()
            e[1].record()
            loss.backward()
            e[2].record()
            opt.step()
            e[3].record()
            torch.cuda.synchronize()
            for i in range(3):
                tot[i] += e[i].elapsed_time(e[i + 1]) / 1000
        return [t / reps for t in tot]

    return step, phases


def profile_kernels(variant, precision, batch, dev, log, topn=12):
    model, shape, ncls = make(variant, dev)
    step, _ = make_step(model, shape, ncls, batch, precision, variant, dev)
    for _ in range(3):
        step()
    torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            step()
        torch.cuda.synchronize()
    rows = []
    for ev in prof.key_averages():
        t = getattr(ev, "self_device_time_total", None) or getattr(ev, "self_cuda_time_total", 0)
        if t > 0:
            rows.append((ev.key, t / 1000 / 5, ev.count // 5))
    rows.sort(key=lambda r: -r[1])
    for i, (key, ms, calls) in enumerate(rows[:topn]):
        log.add(variant=variant, precision=precision, rank=i + 1, kernel=key[:120], self_cuda_ms=ms, calls=calls)
    del model, step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("OUT_DIR", "/data/results") + "/e7")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("E7 cần GPU (autocast BF16 và cuda event).")
    dev = "cuda"
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = os.path.join(a.out, f"{run_id}{'_quick' if a.quick else ''}")
    os.makedirs(out, exist_ok=True)
    write_json(os.path.join(out, "meta.json"), {"experiment": "E7", "env": env_info(dev), "quick": a.quick})

    batches = [32] if a.quick else [32, 128, 512]
    plan = [("base", ["fp32", "bf16"]), ("no_avgpool", ["fp32", "bf16"]),
            ("channels_last", ["fp32", "bf16"]), ("base", ["pure_bf16"]), ("resnet18", ["fp32", "bf16"])]
    log = CsvLog(os.path.join(out, "e7_steps.csv"), STEP_FIELDS)
    for variant, precisions in plan:
        for precision in precisions:
            for b in batches:
                try:
                    model, shape, ncls = make(variant, dev)
                    step, phases = make_step(model, shape, ncls, b, precision, variant, dev)
                    r = bench(step, dev, warmup=3, budget_s=4.0 if a.quick else 8.0)
                    fwd_s, bwd_s, opt_s = phases()
                    label = "pure_bf16" if precision == "pure_bf16" else precision
                    log.add(variant=variant, precision=label, batch=b, step_s=r["median_s"],
                            fwd_s=fwd_s, bwd_s=bwd_s, opt_s=opt_s, img_s=b / r["median_s"])
                except torch.cuda.OutOfMemoryError:
                    print(f"  OOM {variant} {precision} b={b}", flush=True)
                finally:
                    model = step = phases = None
                    torch.cuda.empty_cache()

    plog = CsvLog(os.path.join(out, "e7_kernels.csv"), PROF_FIELDS)
    pb = 32 if a.quick else 256
    for variant, precision in [("base", "fp32"), ("base", "bf16"), ("no_avgpool", "bf16"),
                         ("channels_last", "fp32"), ("channels_last", "bf16")]:
        profile_kernels(variant, precision, pb, dev, plog)
    dump_dir_to_log(out)


if __name__ == "__main__":
    main()
