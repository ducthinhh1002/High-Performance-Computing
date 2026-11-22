"""E5: hội tụ của các thuật toán cập nhật đồng bộ (bài §5.1.1, §5.1.2) và vai trò của shuffle.

k máy (rank), mỗi máy giữ một phần dữ liệu CIFAR-10 cố định, huấn luyện ResNet-18.
Thuật toán (tham số --algs):
  sync          all-reduce gradient mỗi bước, batch tổng k·b (ADMM.SGD của bài: lấy tham số,
                tính gradient, đẩy gradient mỗi vòng, tương đương SGD đồng bộ)
  avg<τ>        mỗi máy chạy SGD cục bộ τ bước rồi lấy trung bình tham số (ParallelSGD với chu kỳ τ);
                avg_epoch = τ bằng số bước của một epoch: đúng "một vòng SGD = một epoch" của bài
  oneshot       chạy cục bộ cả quá trình, chỉ lấy trung bình ở cuối (Zinkevich, thuật toán 1 gốc)
Bố cục dữ liệu (--layouts), tương ứng giả định của bài:
  shuffled      xáo trộn toàn cục rồi chia đều (kết quả của bước ShuffleSGD, Algorithm 2)
  sorted        sắp theo nhãn rồi chia từng khối liên tiếp: mỗi máy chỉ thấy 2–3 lớp ("không lưu
                ngẫu nhiên, đã sắp theo nhãn" như bài giả định khi chưa shuffle)
Đo: độ chính xác test của mô hình đã lấy trung bình sau mỗi epoch, thời gian thực, số lần và số
byte giao tiếp. Mọi thuật toán có cùng số epoch (mỗi máy duyệt hết phần dữ liệu của mình).
"""
import argparse
import math
import os
import time
from datetime import datetime, timezone

import torch
import torch.distributed as dist
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from common.data import CIFAR10Npz
from common.dist import barrier, cleanup, init_dist, sync
from common.models import build, n_params
from common.timing import CsvLog, dump_dir_to_log, env_info, write_json

FIELDS = ["model", "layout", "alg", "seed", "epoch", "steps", "test_acc", "train_loss", "wall_s",
          "comm_calls", "comm_MB"]


def float_state(model):
    """Tham số và buffer dấu phẩy động (kể cả running_mean/var của BatchNorm)."""
    return [t for t in list(model.parameters()) + list(model.buffers()) if t.is_floating_point()]


class Comm:
    def __init__(self, k):
        self.k, self.calls, self.bytes = k, 0, 0

    def average_(self, tensors):
        if self.k == 1 or not tensors:      # GroupNorm không có buffer: danh sách rỗng
            return
        flat = _flatten_dense_tensors([t.data for t in tensors])
        dist.all_reduce(flat)
        flat.div_(self.k)
        for t, s in zip(tensors, _unflatten_dense_tensors(flat, [t.data for t in tensors])):
            t.data.copy_(s)
        self.calls += 1
        self.bytes += flat.numel() * 4


def make_shard(layout, y_all, rank, k, seed):
    n = len(y_all)
    if layout == "sorted":
        order = torch.argsort(y_all, stable=True)
        return order[rank * (n // k):(rank + 1) * (n // k)]
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=g)
    return order[rank::k][: n // k]


@torch.no_grad()
def evaluate(model, x, y, bs=1000):
    model.eval()
    correct = 0
    for i in range(0, len(y), bs):
        correct += (model(x[i:i + bs]).argmax(1) == y[i:i + bs]).sum().item()
    model.train()
    return correct / len(y)


def run(layout, alg, seed, a, rank, k, device, data, log):
    x_tr, y_tr, x_te, y_te = data
    torch.manual_seed(seed)                     # cùng khởi tạo trên mọi rank
    model, _, _ = build(a.model, device)
    eval_model, _, _ = build(a.model, device)
    shard = make_shard(layout, y_tr.cpu(), rank, k, seed).to(device)
    n_local = len(shard)
    steps_per_epoch = n_local // a.batch
    total_steps = steps_per_epoch * a.epochs
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=5e-4, foreach=True)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / total_steps)))
    loss_fn = nn.CrossEntropyLoss()
    comm = Comm(k)
    gen = torch.Generator(device="cpu").manual_seed(seed * 1000 + rank)

    if alg == "sync":
        period = 0
    elif alg == "oneshot":
        period = None
    elif alg == "avg_epoch":
        period = steps_per_epoch
    else:
        period = int(alg.replace("avg", ""))

    # Đồng bộ trạng thái ban đầu (cùng seed nên đã giống; all-reduce lấy trung bình để chắc chắn).
    comm.average_(float_state(model))
    comm.calls, comm.bytes = 0, 0

    sync(device)
    barrier()
    t_start = time.perf_counter()
    step = 0
    for epoch in range(1, a.epochs + 1):
        perm = shard[torch.randperm(n_local, generator=gen).to(device)]
        loss_sum = torch.zeros((), device=device)
        for i in range(steps_per_epoch):
            idx = perm[i * a.batch:(i + 1) * a.batch]
            xb, yb = x_tr[idx], y_tr[idx]
            flip = torch.rand(len(idx), device=device) < 0.5
            xb = torch.where(flip.view(-1, 1, 1, 1), xb.flip(3), xb)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            if alg == "sync":
                grads = [p.grad for p in model.parameters()]
                comm.average_(grads)
            opt.step()
            sched.step()
            step += 1
            loss_sum += loss.detach()
            if period and step % period == 0:
                comm.average_(float_state(model))
        # Mô hình để đánh giá: bản trung bình của các máy (không đụng vào trạng thái cục bộ).
        train_comm = (comm.calls, comm.bytes)
        if alg == "sync":
            # Gradient đã đồng bộ nên tham số giống nhau; chỉ cần gộp thống kê BatchNorm.
            comm.average_([b for b in model.buffers() if b.is_floating_point()])
            em = model
        else:
            eval_model.load_state_dict(model.state_dict())
            comm.average_(float_state(eval_model))
            comm.calls, comm.bytes = train_comm      # lần lấy trung bình để đánh giá không tính vào chi phí
            em = eval_model
        acc = evaluate(em, x_te, y_te) if rank == 0 else 0.0
        sync(device)
        if rank == 0:
            log.add(model=a.model, layout=layout, alg=alg, seed=seed, epoch=epoch, steps=step, test_acc=acc,
                    train_loss=(loss_sum / steps_per_epoch).item(), wall_s=time.perf_counter() - t_start,
                    comm_calls=comm.calls, comm_MB=comm.bytes / 1e6)
    del model, eval_model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("OUT_DIR", "/data/results") + "/e5")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128, help="batch mỗi máy")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--layouts", default="shuffled,sorted")
    ap.add_argument("--algs", default="sync,avg_epoch,avg50,oneshot")
    ap.add_argument("--model", default="resnet18_cifar", choices=["resnet18_cifar", "resnet18_gn_cifar"])
    a = ap.parse_args()
    if a.quick:
        a.epochs, a.batch, a.seeds, a.algs = 1, 32, "0", "sync,avg_epoch"

    rank, k, _, device = init_dist()
    run_id = os.environ.get("RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = os.path.join(a.out, f"{run_id}_k{k}{'_quick' if a.quick else ''}")
    if rank == 0:
        os.makedirs(out, exist_ok=True)
        write_json(os.path.join(out, "meta.json"), {"experiment": "E5", "k": k, "env": env_info(device),
                                                    "args": vars(a), "params": n_params(build(a.model, "cpu")[0])})
    tr, te = CIFAR10Npz("train"), CIFAR10Npz("test")
    if a.quick:
        tr.x, tr.y = tr.x[:2000], tr.y[:2000]
        te.x, te.y = te.x[:500], te.y[:500]
    data = (tr.x.to(device), tr.y.to(device), te.x.to(device), te.y.to(device))
    log = CsvLog(os.path.join(out, "convergence.csv"), FIELDS) if rank == 0 else None
    for seed in [int(s) for s in a.seeds.split(",")]:
        for layout in a.layouts.split(","):
            for alg in a.algs.split(","):
                if rank == 0:
                    print(f"=== layout={layout} alg={alg} seed={seed}", flush=True)
                run(layout, alg, seed, a, rank, k, device, data, log)
    if rank == 0:
        dump_dir_to_log(out)
    barrier()
    cleanup()


if __name__ == "__main__":
    main()
