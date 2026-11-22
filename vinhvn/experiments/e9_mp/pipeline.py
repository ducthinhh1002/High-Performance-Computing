"""E9a: model parallelism kiểu pipeline (GPipe), bài §2.

Bài nói: model parallelism chỉ dùng khi mô hình không vừa một máy, "không phải để tăng tốc huấn luyện"; mỗi lớp
cần truyền đầu ra từ máy này sang máy kia tuần tự. Script đo ba điều:
  1. Tăng tốc theo số stage S và số micro-batch M so với S=1 (và so với data parallel ở E4).
  2. Bong bóng pipeline: lý thuyết (S-1)/(M+S-1) so với đo (hiệu suất = T(S=1) / (S·T(S))).
  3. Bộ nhớ đỉnh mỗi GPU và byte truyền giữa các stage, và trường hợp "không vừa một GPU" (arch mlp_big).

Mỗi rank là một stage (rank = chỉ số stage), chạy bằng torchrun, S = WORLD_SIZE. Lịch GPipe: forward lần lượt M micro-batch
rồi backward ngược lại, gradient cộng dồn, một bước SGD cuối. Giữa hai stage dùng dist.send/recv (NCCL: NVLink, hoặc
mạng khi hai stage ở hai pod khác nhau / khi ép NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1).
Kiến trúc (--archs):
  vgg16     VGG-16 CIFAR (134M tham số), chia theo MAC cân bằng (quy hoạch động)
  mlp       16 lớp Linear 8192×8192 (1,07 tỉ tham số), mỗi lớp cùng chi phí → chia đều
  mlp_big   45 lớp 16384×16384 (12,1 tỉ tham số, FP32): gradient + trọng số ≈ 97 GB > 80 GB, không vừa một H100
"""
import argparse
import os
import statistics
import time
from datetime import datetime, timezone

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from common.dist import barrier, cleanup, init_dist, sync
from common.timing import CsvLog, dump_dir_to_log, env_info, write_json

FIELDS = ["arch", "S", "M", "b_global", "b_micro", "params_M", "step_s", "samples_s", "peak_mem_GB_max",
          "peak_mem_GB_min", "sent_MB_per_step", "bubble_theory", "status"]

ARCHS = {
    "vgg16": dict(b_global=512, Ms=[1, 2, 4, 8, 16]),
    "mlp": dict(b_global=4096, Ms=[1, 2, 4, 8, 16], H=8192, L=16),
    "mlp_big": dict(b_global=256, Ms=[4, 8], H=16384, L=45),
}
QUICK = {
    "vgg16": dict(b_global=16, Ms=[1, 2]),
    "mlp": dict(b_global=16, Ms=[1, 2], H=64, L=4),
    "mlp_big": dict(b_global=16, Ms=[2], H=64, L=6),
}


def linear_partition(costs, S):
    """Chia dãy chi phí thành S đoạn liên tiếp, không rỗng, tối thiểu hoá đoạn nặng nhất (quy hoạch động)."""
    n = len(costs)
    assert n >= S, "ít lớp hơn số stage"
    pre = [0.0]
    for c in costs:
        pre.append(pre[-1] + c)
    INF = float("inf")
    dp = [[INF] * (n + 1) for _ in range(S + 1)]
    cut = [[0] * (n + 1) for _ in range(S + 1)]
    dp[0][0] = 0.0
    for s in range(1, S + 1):
        for i in range(s, n + 1):
            for j in range(s - 1, i):
                v = max(dp[s - 1][j], pre[i] - pre[j])
                if v < dp[s][i]:
                    dp[s][i], cut[s][i] = v, j
    bounds, i = [], n
    for s in range(S, 0, -1):
        j = cut[s][i]
        bounds.append((j, i))
        i = j
    return bounds[::-1]


class Spec:
    """Mô tả kiến trúc: chi phí từng lớp, shape đầu ra từng lớp, hàm dựng các lớp [a, b)."""

    def __init__(self, arch, cfg, device):
        self.arch, self.cfg, self.device = arch, cfg, device
        if arch == "vgg16":
            torch.manual_seed(0)
            m = torchvision.models.vgg16(num_classes=10)
            self.layers = list(m.features) + [m.avgpool, nn.Flatten()] + list(m.classifier)
            for l in self.layers:
                if hasattr(l, "inplace"):
                    l.inplace = False            # lá có requires_grad ở biên stage: không được sửa tại chỗ
            self.in_shape = (3, 32, 32)
            self.costs, self.out_shapes = [], []
            x = torch.zeros(1, *self.in_shape)
            with torch.no_grad():
                for l in self.layers:
                    y = l(x)
                    if isinstance(l, nn.Conv2d):
                        c = y.numel() * l.in_channels * l.kernel_size[0] * l.kernel_size[1]
                    elif isinstance(l, nn.Linear):
                        c = l.in_features * l.out_features
                    else:
                        c = 0.0
                    self.costs.append(float(c) + 1.0)
                    self.out_shapes.append(tuple(y.shape[1:]))
                    x = y
            self.params = sum(p.numel() for p in m.parameters())
        else:
            H, L = cfg["H"], cfg["L"]
            self.in_shape = (H,)
            self.costs = [float(H * H)] * L + [float(H * 10)]
            self.out_shapes = [(H,)] * L + [(10,)]
            self.params = L * (H * H + H) + H * 10 + 10

    def build(self, a, b):
        if self.arch == "vgg16":
            return nn.Sequential(*self.layers[a:b]).to(self.device)
        H = self.cfg["H"]
        mods = []
        for i in range(a, b):
            torch.manual_seed(1000 + i)              # cùng trọng số cho mọi cách chia stage (--check)
            if i < len(self.costs) - 1:
                mods += [nn.Linear(H, H, device=self.device), nn.ReLU()]
            else:
                mods += [nn.Linear(H, 10, device=self.device)]
        return nn.Sequential(*mods)

    def shape_in(self, first):
        return self.in_shape if first == 0 else self.out_shapes[first - 1]


def run_step(stage, rank, S, M, xs, ys, shape_in, bm, device, sent):
    ins, outs = [], []
    for i in range(M):
        if rank == 0:
            xin = xs[i]
        else:
            xin = torch.empty((bm, *shape_in), device=device)
            dist.recv(xin, src=rank - 1)
            xin.requires_grad_(True)
        out = stage(xin)
        if rank < S - 1:
            t = out.detach().contiguous()
            dist.send(t, dst=rank + 1)
            sent[0] += t.numel() * t.element_size()
        ins.append(xin)
        outs.append(out)
    for i in reversed(range(M)):
        if rank == S - 1:
            (F.cross_entropy(outs[i], ys[i]) / M).backward()
        else:
            g = torch.empty_like(outs[i])
            dist.recv(g, src=rank + 1)
            outs[i].backward(g)
        if rank > 0:
            gi = ins[i].grad.contiguous()
            dist.send(gi, dst=rank - 1)
            sent[0] += gi.numel() * gi.element_size()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("OUT_DIR", "/data/results") + "/e9")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--archs", default="vgg16,mlp")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--check", action="store_true", help="in chuẩn gradient toàn cục (mlp) để so các S với nhau")
    a = ap.parse_args()
    cfgs = QUICK if a.quick else ARCHS
    rank, S, _, device = init_dist()
    run_id = os.environ.get("RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = os.path.join(a.out, f"{run_id}_pp_S{S}{'_quick' if a.quick else ''}")
    if rank == 0:
        os.makedirs(out, exist_ok=True)
        write_json(os.path.join(out, "meta.json"), {"experiment": "E9a", "S": S, "env": env_info(device),
                                                    "archs": a.archs, "quick": a.quick})
    log = CsvLog(os.path.join(out, "pipeline.csv"), FIELDS) if rank == 0 else None

    for arch in a.archs.split(","):
        cfg = cfgs[arch]
        spec = Spec(arch, cfg, device)
        status = "ok"
        try:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            bounds = linear_partition(spec.costs, S)
            lo, hi = bounds[rank]
            stage = spec.build(lo, hi)
            shape_in = spec.shape_in(lo)
            params = [p for p in stage.parameters()]
            opt = torch.optim.SGD(params, lr=1e-4, foreach=True)
            if rank == 0 and S > 1:
                print(f"[{arch}] stage split (layer ranges): {bounds}", flush=True)
        except torch.cuda.OutOfMemoryError:
            status, stage = "OOM(build)", None
        # Mọi rank phải cùng biết có OOM hay không, tránh một rank treo ở send/recv.
        flag = torch.tensor([1.0 if status != "ok" else 0.0], device=device)
        if S > 1:
            dist.all_reduce(flag)
        if flag.item() > 0:
            if rank == 0:
                log.add(arch=arch, S=S, M=0, b_global=cfg["b_global"], b_micro=0, params_M=spec.params / 1e6,
                        step_s=float("nan"), samples_s=float("nan"), peak_mem_GB_max=float("nan"),
                        peak_mem_GB_min=float("nan"), sent_MB_per_step=0, bubble_theory=float("nan"), status="OOM(build)")
            del stage
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        if a.check and arch != "vgg16":                # vgg có Dropout nên không tất định
            for M in cfg["Ms"]:
                bg, bm = cfg["b_global"], cfg["b_global"] // M
                g = torch.Generator(device="cpu").manual_seed(1)
                gy = torch.Generator(device="cpu").manual_seed(2)
                xs = ([t.to(device) for t in torch.randn(bg, *spec.in_shape, generator=g).chunk(M)] if rank == 0 else None)
                ys = ([t.to(device) for t in torch.randint(0, 10, (bg,), generator=gy).chunk(M)] if rank == S - 1 else None)
                for q in params:
                    q.grad = None
                run_step(stage, rank, S, M, xs, ys, shape_in, bm, device, [0])
                sq = torch.tensor([sum(float(q.grad.double().pow(2).sum()) for q in params)], device=device, dtype=torch.float64)
                if S > 1:
                    dist.all_reduce(sq)
                if rank == 0:
                    print(f"CHECK arch={arch} S={S} M={M} global_grad_norm={sq.item() ** 0.5:.8f}", flush=True)
            continue
        for M in cfg["Ms"]:
            bg = cfg["b_global"]
            if bg % M:
                continue
            bm = bg // M
            g = torch.Generator(device="cpu").manual_seed(1)
            gy = torch.Generator(device="cpu").manual_seed(2)
            xs = ([t.to(device) for t in torch.randn(bg, *spec.in_shape, generator=g).chunk(M)]
                  if rank == 0 else None)
            ys = ([t.to(device) for t in torch.randint(0, 10, (bg,), generator=gy).chunk(M)]
                  if rank == S - 1 else None)
            sent = [0]
            st = "ok"
            try:
                def one():
                    opt.zero_grad(set_to_none=True)
                    run_step(stage, rank, S, M, xs, ys, shape_in, bm, device, sent)
                    opt.step()
                for _ in range(2):
                    one()
                sync(device); barrier()
                sent[0] = 0
                t0 = time.perf_counter()
                for _ in range(a.steps):
                    one()
                sync(device); barrier()
                step = (time.perf_counter() - t0) / a.steps
            except torch.cuda.OutOfMemoryError:
                st, step = "OOM(step)", float("nan")
            peak = (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else 0.0
            pk = torch.tensor([peak], device=device)
            mx, mn = pk.clone(), pk.clone()
            if S > 1 and st == "ok":
                dist.all_reduce(mx, op=dist.ReduceOp.MAX)
                dist.all_reduce(mn, op=dist.ReduceOp.MIN)
                tot = torch.tensor([sent[0] / a.steps / 1e6], device=device)
                dist.all_reduce(tot)
            else:
                tot = torch.tensor([0.0], device=device)
            if rank == 0:
                log.add(arch=arch, S=S, M=M, b_global=bg, b_micro=bm, params_M=spec.params / 1e6, step_s=step,
                        samples_s=bg / step if st == "ok" else float("nan"), peak_mem_GB_max=mx.item(),
                        peak_mem_GB_min=mn.item(), sent_MB_per_step=tot.item(),
                        bubble_theory=(S - 1) / (M + S - 1), status=st)
            if st != "ok":
                break
        del stage, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if rank == 0:
        dump_dir_to_log(out)
    barrier()
    cleanup()


if __name__ == "__main__":
    main()
