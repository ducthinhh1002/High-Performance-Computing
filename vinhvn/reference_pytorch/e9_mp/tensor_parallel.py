"""E9b: tensor parallelism kiểu Megatron (chia từng lớp ra nhiều GPU), bài §2 (model parallelism).

Khác pipeline: mỗi lớp được chia theo chiều rộng, nên phải giao tiếp **mỗi lớp, mỗi bước** (khối MLP Megatron:
cột song song rồi hàng song song, một all-reduce ở forward và một ở backward), thay vì chỉ truyền ở biên stage.
Dùng để đo chi phí giao tiếp của model parallelism qua NVLink so với qua mạng (khi ép NCCL đi qua Socket hoặc
khi các rank ở các pod khác nhau).

Mô hình: L khối [x + W2·ReLU(W1·x)], W1: H→4H chia theo cột, W2: 4H→H chia theo hàng, T = WORLD_SIZE.
Tất cả rank nhận cùng đầu vào; mỗi rank giữ 1/T trọng số. T=1 là mốc đối chứng (toàn bộ mô hình trên một GPU).
Chạy: torchrun --nproc_per_node=T -m e9_mp.tensor_parallel
"""
import argparse
import os
import time
from datetime import datetime, timezone

import torch
import torch.distributed as dist
import torch.nn as nn

from common.dist import barrier, cleanup, init_dist, sync
from common.timing import CsvLog, dump_dir_to_log, env_info, write_json

FIELDS = ["T", "L", "H", "tokens", "params_M_total", "params_M_per_rank", "step_s", "tokens_s",
          "allreduce_msg_MB", "allreduce_per_step", "comm_GB_per_step_algo", "peak_mem_GB", "status"]
FULL = {"L": 8, "H": 8192, "tokens": 4096}
QUICK = {"L": 2, "H": 64, "tokens": 32}


class _CopyToTP(torch.autograd.Function):
    """f: forward là identity, backward cộng gradient đầu vào qua các rank."""

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, g):
        g = g.contiguous()
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(g)
        return g


class _ReduceFromTP(torch.autograd.Function):
    """g: forward cộng đầu ra từng phần qua các rank, backward là identity."""

    @staticmethod
    def forward(ctx, x):
        x = x.contiguous().clone()
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, g):
        return g


class Block(nn.Module):
    def __init__(self, H, T, device):
        super().__init__()
        ffn = 4 * H // T
        self.w1 = nn.Linear(H, ffn, device=device)
        self.w2 = nn.Linear(ffn, H, bias=False, device=device)

    def forward(self, x):
        h = _CopyToTP.apply(x)
        h = self.w2(torch.relu(self.w1(h)))
        return x + _ReduceFromTP.apply(h)


def load_sharded(model, T, rank, H):
    """Đặt trọng số = lát của cùng một trọng số đầy đủ (theo seed), để kết quả không phụ thuộc T."""
    ffn = 4 * H // T
    for i, blk in enumerate(model):
        g = torch.Generator(device="cpu").manual_seed(500 + i)
        w1 = torch.randn(4 * H, H, generator=g) * 0.02
        b1 = torch.randn(4 * H, generator=g) * 0.02
        w2 = torch.randn(H, 4 * H, generator=g) * 0.02
        with torch.no_grad():
            blk.w1.weight.copy_(w1[rank * ffn:(rank + 1) * ffn])
            blk.w1.bias.copy_(b1[rank * ffn:(rank + 1) * ffn])
            blk.w2.weight.copy_(w2[:, rank * ffn:(rank + 1) * ffn])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("OUT_DIR", "/data/results") + "/e9")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--check", action="store_true", help="khởi tạo từ trọng số đầy đủ cắt lát, in loss và chuẩn gradient để so các T")
    a = ap.parse_args()
    cfg = QUICK if a.quick else FULL
    rank, T, _, device = init_dist()
    run_id = os.environ.get("RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = os.path.join(a.out, f"{run_id}_tp_T{T}{'_quick' if a.quick else ''}")
    if rank == 0:
        os.makedirs(out, exist_ok=True)
        write_json(os.path.join(out, "meta.json"), {"experiment": "E9b", "T": T, "env": env_info(device), "cfg": cfg})
    log = CsvLog(os.path.join(out, "tensor_parallel.csv"), FIELDS) if rank == 0 else None

    L, H, tok = cfg["L"], cfg["H"], cfg["tokens"]
    torch.manual_seed(1234 + rank)
    status = "ok"
    step = float("nan")
    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model = nn.Sequential(*[Block(H, T, device) for _ in range(L)])
        opt = torch.optim.SGD(model.parameters(), lr=1e-5, foreach=True)
        g = torch.Generator(device="cpu").manual_seed(7)
        x = torch.randn(tok, H, generator=g).to(device)       # cùng đầu vào trên mọi rank
        if a.check:
            load_sharded(model, T, rank, H)
            loss = model(x).float().pow(2).mean()
            loss.backward()
            sq = torch.tensor([sum(float(q.grad.double().pow(2).sum()) for q in model.parameters())],
                              device=device, dtype=torch.float64)
            if T > 1:
                dist.all_reduce(sq)                       # mỗi rank giữ lát riêng: tổng bình phương = của toàn bộ
            if rank == 0:
                print(f"CHECK T={T} loss={loss.item():.8f} global_grad_norm={sq.item() ** 0.5:.8f}", flush=True)
            cleanup()
            return

        def one():
            opt.zero_grad(set_to_none=True)
            model(x).float().pow(2).mean().backward()
            opt.step()
        for _ in range(2):
            one()
        sync(device); barrier()
        t0 = time.perf_counter()
        for _ in range(a.steps):
            one()
        sync(device); barrier()
        step = (time.perf_counter() - t0) / a.steps
    except torch.cuda.OutOfMemoryError:
        status = "OOM"
    if rank == 0:
        msg = tok * H * 4 / 1e6
        n_ar = 2 * L if T > 1 else 0                         # mỗi khối: 1 all-reduce forward + 1 backward
        per_rank = L * (H * 4 * H // T * 2 + 4 * H // T) / 1e6
        log.add(T=T, L=L, H=H, tokens=tok, params_M_total=L * (8 * H * H + 4 * H) / 1e6, params_M_per_rank=per_rank,
                step_s=step, tokens_s=tok / step if status == "ok" else float("nan"), allreduce_msg_MB=msg,
                allreduce_per_step=n_ar,
                comm_GB_per_step_algo=n_ar * msg * 2 * (T - 1) / T / 1e3 if T > 1 else 0.0,
                peak_mem_GB=(torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else 0.0, status=status)
        dump_dir_to_log(out)
    barrier()
    cleanup()


if __name__ == "__main__":
    main()
