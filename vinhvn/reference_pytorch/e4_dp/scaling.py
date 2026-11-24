"""E4: data parallelism đồng bộ, đo scaling theo số GPU (bài §5.1, Hình 5).

Mỗi rank giữ một bản sao mô hình, tính gradient trên phần dữ liệu của mình, rồi all-reduce
trung bình gradient trước khi cập nhật: đây chính là một bước ParallelSGD/ADMM.SGD đồng bộ.
Dữ liệu tổng hợp ngẫu nhiên nằm sẵn trên GPU để chỉ đo tính toán và giao tiếp.

Ba chế độ (mỗi k một lần chạy torchrun, k = WORLD_SIZE):
  local   mỗi rank huấn luyện riêng, không giao tiếp  → T_compute, mốc để tính chi phí giao tiếp lộ ra
  manual  backward xong mới gộp toàn bộ gradient thành một vector và all-reduce một lần
          (đúng như bài: tính xong rồi mới đồng bộ, không chồng lấp)  → tách được t_bwd/t_comm/t_opt
  ddp     DistributedDataParallel (all-reduce theo bucket, chồng lấp với backward)

Hai kiểu scaling:
  weak    batch mỗi GPU cố định (B_gpu), batch tổng = k·B_gpu
  strong  batch tổng cố định (B_glob), mỗi GPU B_glob/k
"""
import argparse
import os
import statistics
import time
from datetime import datetime, timezone

import torch
import torch.distributed as dist
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.nn.parallel import DistributedDataParallel as DDP

from common.dist import barrier, cleanup, init_dist, sync
from common.models import build, n_params
from common.timing import CsvLog, dump_dir_to_log, env_info, write_json

FIELDS = ["model", "params", "k", "mode", "scaling", "b_gpu", "b_global", "step_s", "p25_s", "p75_s",
          "t_fwd_s", "t_bwd_s", "t_comm_s", "t_opt_s", "img_s", "img_s_per_gpu", "steps"]

FULL = {"models": {"resnet18_cifar": 128, "vgg16_cifar": 128, "alexnet_gray": 128},
        "strong_global": {"resnet18_cifar": 512, "vgg16_cifar": 512, "alexnet_gray": 512},
        "warmup": 10, "steps": 40}
QUICK = {"models": {"resnet18_cifar": 8}, "strong_global": {"resnet18_cifar": 16}, "warmup": 2, "steps": 4}


def run_config(model_name, mode, b_gpu, k, device, warmup, steps):
    model, shape, ncls = build(model_name, device)
    p = n_params(model)
    x = torch.randn(b_gpu, *shape, device=device)
    y = torch.randint(0, ncls, (b_gpu,), device=device)
    loss_fn = nn.CrossEntropyLoss()
    net = (DDP(model, device_ids=[device.index] if device.type == "cuda" else None,
               bucket_cap_mb=int(os.environ.get("DDP_BUCKET_MB", "25")),
               broadcast_buffers=os.environ.get("DDP_BCAST_BUF", "1") == "1")
           if mode == "ddp" else model)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, foreach=True)
    params = [q for q in model.parameters()]

    def one_step():
        """Trả về (t_total, t_fwd, t_bwd, t_comm, t_opt) đo trên rank này."""
        sync(device)
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(net(x), y)
        sync(device)
        t1 = time.perf_counter()
        loss.backward()
        sync(device)
        t2 = time.perf_counter()
        if mode == "manual":
            flat = _flatten_dense_tensors([q.grad for q in params])
            dist.all_reduce(flat)
            flat.div_(k)
            for q, g in zip(params, _unflatten_dense_tensors(flat, [q.grad for q in params])):
                q.grad.copy_(g)
            sync(device)
        t3 = time.perf_counter()
        opt.step()
        sync(device)
        t4 = time.perf_counter()
        return t4 - t0, t1 - t0, t2 - t1, t3 - t2, t4 - t3

    for _ in range(warmup):
        one_step()
    barrier()
    rows = [one_step() for _ in range(steps)]
    barrier()
    del model, net, opt, x, y
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return p, rows


def med(rows, i):
    return statistics.median(r[i] for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("OUT_DIR", "/data/results") + "/e4")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    cfg = QUICK if a.quick else FULL

    rank, k, _, device = init_dist()
    run_id = os.environ.get("RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = os.path.join(a.out, f"{run_id}_k{k}{'_quick' if a.quick else ''}")
    if rank == 0:
        os.makedirs(out, exist_ok=True)
        write_json(os.path.join(out, "meta.json"), {"experiment": "E4", "k": k, "env": env_info(device), "cfg": cfg})
    log = CsvLog(os.path.join(out, "scaling.csv"), FIELDS) if rank == 0 else None

    modes = ["local", "manual", "ddp"] if k > 1 else ["local"]
    for model_name, b_weak in cfg["models"].items():
        for scaling, b_gpu in (("weak", b_weak), ("strong", cfg["strong_global"][model_name] // k)):
            if b_gpu < 1:
                continue
            for mode in modes:
                try:
                    p, rows = run_config(model_name, mode, b_gpu, k, device, cfg["warmup"], cfg["steps"])
                except torch.cuda.OutOfMemoryError:
                    if rank == 0:
                        print(f"  OOM {model_name} {mode} b={b_gpu}", flush=True)
                    continue
                if rank == 0:
                    q = statistics.quantiles([r[0] for r in rows], n=4)
                    step = med(rows, 0)
                    # 'local' không đồng bộ giữa các rank nên img_s tính theo k rank chạy song song.
                    log.add(model=model_name, params=p, k=k, mode=mode, scaling=scaling, b_gpu=b_gpu,
                            b_global=b_gpu * k, step_s=step, p25_s=q[0], p75_s=q[2],
                            t_fwd_s=med(rows, 1), t_bwd_s=med(rows, 2), t_comm_s=med(rows, 3),
                            t_opt_s=med(rows, 4), img_s=b_gpu * k / step, img_s_per_gpu=b_gpu / step,
                            steps=len(rows))
    if rank == 0:
        dump_dir_to_log(out)
    barrier()
    cleanup()


if __name__ == "__main__":
    main()
