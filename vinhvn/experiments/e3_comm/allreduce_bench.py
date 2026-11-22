"""E3: chi phí giao tiếp trong một node, so với mô hình chi phí của bài (§5.1.1).

Bài đưa ra: gộp k máy theo kiểu BitTorrent tốn O(L·k) + O(p·k/B) về chi phí và
O(p·log k) về thời gian. NCCL dùng ring/tree, nên thời gian all-reduce lý thuyết là
  t(n) = α + 2(k-1)/k · n / B        (ring, n byte)
Script đo t(n) cho all_reduce / broadcast / reduce với k = WORLD_SIZE, rồi khớp α và B
bằng bình phương tối thiểu để đối chiếu với mô hình của bài (xem PLAN.md, E3).

Chạy: torchrun --nproc_per_node=K -m e3_comm.allreduce_bench   (lặp K = 2, 3, 4)
Kích thước thử: luỹ thừa của 4 từ 4 KiB đến 1 GiB, cộng kích thước gradient thật
của ResNet-18, AlexNet(ảnh xám), VGG-16.
"""
import argparse
import os
import statistics
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.distributed as dist

from common.dist import barrier, cleanup, init_dist, sync
from common.models import MODELS, build, n_params
from common.timing import CsvLog, dump_dir_to_log, env_info, write_json

FIELDS = ["op", "k", "label", "bytes", "median_s", "p25_s", "p75_s", "algbw_GBps", "busbw_GBps", "trials"]


def model_sizes():
    out = {}
    for name in MODELS:
        m, _, _ = build(name, "cpu")
        out[name] = n_params(m) * 4
        del m
    return out


def run_op(op, buf, k):
    if op == "all_reduce":
        dist.all_reduce(buf)
    elif op == "broadcast":
        dist.broadcast(buf, src=0)
    elif op == "reduce":
        dist.reduce(buf, dst=0)


def busbw_factor(op, k):
    # Cùng quy ước với nccl-tests: nhân algbw để so được với băng thông phần cứng.
    return 2 * (k - 1) / k if op == "all_reduce" else 1.0


def measure(op, nbytes, device, k, iters, trials, warmup):
    buf = torch.ones(max(1, nbytes // 4), dtype=torch.float32, device=device)
    for _ in range(warmup):
        run_op(op, buf, k)
    sync(device)
    ts = []
    for _ in range(trials):
        barrier()
        sync(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            run_op(op, buf, k)
        sync(device)
        ts.append((time.perf_counter() - t0) / iters)
    del buf
    return ts


def fit_alpha_beta(rows, k):
    """Khớp t = α + c·n/B bằng bình phương tối thiểu trọng số theo 1/t (để cỡ nhỏ không bị bỏ qua)."""
    n = np.array([r["bytes"] for r in rows], dtype=float)
    t = np.array([r["median_s"] for r in rows], dtype=float)
    c = busbw_factor(rows[0]["op"], k)
    x = np.stack([np.ones_like(n), c * n], axis=1)
    w = 1.0 / t
    sol, *_ = np.linalg.lstsq(x * w[:, None], t * w, rcond=None)
    alpha, inv_b = float(sol[0]), float(sol[1])
    pred = x @ sol
    return {"alpha_s": alpha, "B_GBps": (1.0 / inv_b / 1e9) if inv_b > 0 else None,
            "max_rel_err": float(np.max(np.abs(pred - t) / t))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("OUT_DIR", "/data/results") + "/e3")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--ops", default="all_reduce,broadcast,reduce")
    a = ap.parse_args()

    rank, k, _, device = init_dist()
    run_id = os.environ.get("RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = os.path.join(a.out, f"{run_id}_k{k}{'_quick' if a.quick else ''}")
    if rank == 0:
        os.makedirs(out, exist_ok=True)

    top = 22 if a.quick else 30                      # 4 MiB nhanh, 1 GiB đầy đủ
    sizes = [("pow4", 4 ** e) for e in range(6, top // 2 + 1)]      # 4 KiB … 1 GiB
    if not a.quick:
        sizes += [(f"grad_{n}", b) for n, b in model_sizes().items()]
    sizes.sort(key=lambda s: s[1])
    iters_for = lambda b: 50 if b <= 1 << 20 else (20 if b <= 1 << 26 else 5)  # noqa: E731

    log = CsvLog(os.path.join(out, "comm.csv"), FIELDS) if rank == 0 else None
    by_op = {}
    for op in a.ops.split(","):
        rows = []
        for label, nbytes in sizes:
            ts = measure(op, nbytes, device, k, iters_for(nbytes), 3 if a.quick else 5, 3)
            if rank == 0:
                q = statistics.quantiles(ts, n=4) if len(ts) >= 2 else [ts[0]] * 3
                med = statistics.median(ts)
                algbw = nbytes / med / 1e9
                row = dict(op=op, k=k, label=label, bytes=nbytes, median_s=med, p25_s=q[0], p75_s=q[2],
                           algbw_GBps=algbw, busbw_GBps=algbw * busbw_factor(op, k), trials=len(ts))
                log.add(**row)
                rows.append(row)
        if rank == 0 and len(rows) >= 3:
            # Chỉ khớp các cỡ ≥ 1 MiB để pha băng thông chiếm ưu thế; cỡ nhỏ cho α.
            by_op[op] = fit_alpha_beta(rows, k)
    if rank == 0:
        write_json(os.path.join(out, "fit.json"), {"experiment": "E3", "k": k, "env": env_info(device),
                                                   "backend": dist.get_backend() if dist.is_initialized() else None,
                                                   "fit": by_op, "quick": a.quick})
        dump_dir_to_log(out)
    barrier()
    cleanup()


if __name__ == "__main__":
    main()
