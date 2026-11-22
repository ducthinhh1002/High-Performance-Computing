"""E6: cập nhật bất đồng bộ kiểu Downpour SGD (bài §5.2.1, Algorithm 4).

Một process parameter server giữ vector tham số θ trong bộ nhớ chia sẻ (CPU). W worker, mỗi
worker một GPU (hoặc dùng chung GPU khi W > số GPU), lặp: lấy θ → tính gradient trên mini-batch
→ đẩy gradient lên server → server áp ngay θ ← θ − lr·g mà không chờ worker khác.

Đối chiếu với bài (§5.2.1): T = thời gian worker tính một gradient, τ = thời gian server nhận và áp
một gradient. Bài suy ra: tốc độ tối đa là 1/τ cập nhật mỗi giây, và thêm worker không còn tác dụng
khi W ≥ P = T/τ. Script đo T, τ, số cập nhật/giây theo W, và độ trễ (staleness) = số cập nhật mà
server đã áp từ lúc worker lấy θ đến lúc gradient của nó được áp. Độ chính xác sau cùng một số
cập nhật tổng cho thấy ảnh hưởng của staleness lên hội tụ.

Lưu ý thiết kế: gradient truyền qua bộ đệm chia sẻ riêng của từng worker và worker chờ server áp
xong mới tính tiếp, nên τ gồm cả thời gian xếp hàng. Đọc θ không khoá (kiểu Hogwild) nên có thể đọc
lẫn hai phiên bản; đây là bất đồng bộ thật chứ không phải mô phỏng. BatchNorm: worker giữ thống kê
cục bộ, khi đánh giá hiệu chỉnh lại bằng vài batch train (không dùng nhãn).
"""
import argparse
import os
import statistics
import time
from datetime import datetime, timezone

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from common.data import CIFAR10Npz
from common.models import build
from common.timing import CsvLog, dump_dir_to_log, env_info, write_json

FIELDS = ["model", "extra_ms", "workers", "gpus", "total_updates", "wall_s", "updates_per_s", "T_ms_median",
          "tau_ms_mean", "P_pred", "stale_mean", "stale_p50", "stale_p95", "stale_max", "test_acc"]


def pick_device(wid, n_gpu):
    return torch.device("cuda", wid % n_gpu) if n_gpu > 0 else torch.device("cpu")


def server(theta, grad_bufs, pushq, done, version, total, lr, n_threads, resultq, extra_s=0.0):
    torch.set_num_threads(n_threads)
    taus, stale = [], []
    for _ in range(total):
        wid, fetched = pushq.get()
        t0 = time.perf_counter()
        theta.add_(grad_bufs[wid], alpha=-lr)
        if extra_s:
            time.sleep(extra_s)             # làm chậm server có kiểm soát: cộng thẳng vào τ
        stale.append(version.value - fetched)
        with version.get_lock():
            version.value += 1
        done[wid].set()
        taus.append(time.perf_counter() - t0)
    resultq.put((taus, stale))


def worker(wid, model_name, n_gpu, theta, grad_bufs, pushq, done, version, steps, batch, ready, start,
           resultq, data_dir):
    torch.manual_seed(1000 + wid)
    dev = pick_device(wid, n_gpu)
    torch.backends.cudnn.allow_tf32 = False
    model, shape, ncls = build(model_name, dev)
    loss_fn = nn.CrossEntropyLoss()
    ds = CIFAR10Npz("train", root=data_dir) if data_dir else None
    if ds is not None and model_name == "resnet18_cifar":
        x_all, y_all = ds.x.to(dev), ds.y.to(dev)
    else:  # mô hình không dùng CIFAR (VGG ở đây cũng dùng CIFAR; AlexNet dùng dữ liệu tổng hợp)
        x_all = torch.randn(max(4096, batch), *shape, device=dev)
        y_all = torch.randint(0, ncls, (len(x_all),), device=dev)
        if ds is not None and shape == (3, 32, 32):
            x_all, y_all = ds.x.to(dev), ds.y.to(dev)
    params = list(model.parameters())
    gbuf = grad_bufs[wid]
    Ts = []
    ready.wait()
    start.wait()
    for _ in range(steps):
        fetched = version.value
        vector_to_parameters(theta.to(dev, non_blocking=False), params)
        idx = torch.randint(0, len(x_all), (batch,), device=dev)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        for p in params:
            p.grad = None
        loss_fn(model(x_all[idx]), y_all[idx]).backward()
        g = parameters_to_vector([p.grad for p in params])
        gbuf.copy_(g.cpu())
        Ts.append(time.perf_counter() - t0)
        done[wid].clear()
        pushq.put((wid, fetched))
        done[wid].wait()
    resultq.put((wid, Ts))


@torch.no_grad()
def eval_theta(model_name, theta, n_gpu, data_dir, calib_batches=20):
    """Nạp θ vào mô hình, hiệu chỉnh BatchNorm bằng batch train rồi đo độ chính xác test."""
    dev = pick_device(0, n_gpu)
    model, _, _ = build(model_name, dev)
    vector_to_parameters(theta.to(dev), list(model.parameters()))
    if data_dir is None or model_name == "alexnet_gray":
        return float("nan")
    tr, te = CIFAR10Npz("train", root=data_dir), CIFAR10Npz("test", root=data_dir)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None                       # trung bình tích luỹ
    model.train()
    g = torch.Generator().manual_seed(0)
    for _ in range(calib_batches):
        idx = torch.randint(0, len(tr), (256,), generator=g)
        model(tr.x[idx].to(dev))
    model.eval()
    correct = 0
    for i in range(0, len(te), 1000):
        correct += (model(te.x[i:i + 1000].to(dev)).argmax(1).cpu() == te.y[i:i + 1000]).sum().item()
    return correct / len(te)


def run_once(model_name, W, n_gpu, total_updates, batch, lr, server_threads, data_dir, extra_ms=0.0):
    ctx = mp.get_context("spawn")
    dev0 = pick_device(0, 0)
    torch.manual_seed(0)
    model, _, _ = build(model_name, dev0)
    theta = parameters_to_vector(model.parameters()).detach().clone().share_memory_()
    grad_bufs = [torch.zeros_like(theta).share_memory_() for _ in range(W)]
    pushq = ctx.Queue()
    done = [ctx.Event() for _ in range(W)]
    version = ctx.Value("l", 0)
    ready, start = ctx.Event(), ctx.Event()
    resultq, sresultq = ctx.Queue(), ctx.Queue()
    steps = total_updates // W
    total = steps * W
    srv = ctx.Process(target=server, args=(theta, grad_bufs, pushq, done, version, total, lr,
                                           server_threads, sresultq, extra_ms / 1e3))
    ws = [ctx.Process(target=worker, args=(i, model_name, n_gpu, theta, grad_bufs, pushq, done, version,
                                           steps, batch, ready, start, resultq, data_dir)) for i in range(W)]
    srv.start()
    for w in ws:
        w.start()
    time.sleep(1.0)
    ready.set()
    time.sleep(2.0 + 0.5 * W)                       # worker dựng mô hình xong trước khi tính giờ
    t0 = time.perf_counter()
    start.set()
    taus, stale = sresultq.get(timeout=3600)
    wall = time.perf_counter() - t0
    Ts = [t for _ in range(W) for t in resultq.get(timeout=600)[1]]
    for p in ws + [srv]:
        p.join(timeout=60)
    acc = eval_theta(model_name, theta, n_gpu, data_dir)
    T = statistics.median(Ts)
    tau = statistics.mean(taus)
    srt = sorted(stale)
    return dict(model=model_name, extra_ms=extra_ms, workers=W, gpus=n_gpu, total_updates=total, wall_s=wall,
                updates_per_s=total / wall, T_ms_median=T * 1e3, tau_ms_mean=tau * 1e3,
                P_pred=T / tau, stale_mean=statistics.mean(stale), stale_p50=srt[len(srt) // 2],
                stale_p95=srt[int(len(srt) * 0.95)], stale_max=srt[-1], test_acc=acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("OUT_DIR", "/data/results") + "/e6")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--models", default="resnet18_cifar,vgg16_cifar")
    ap.add_argument("--workers", default="1,2,4,8")
    ap.add_argument("--updates", type=int, default=1600, help="tổng số cập nhật, chia đều cho các worker")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--server-threads", type=int, default=4)
    ap.add_argument("--tau-extra-ms", type=float, default=0.0, help="thêm độ trễ vào mỗi lần server áp cập nhật")
    a = ap.parse_args()
    if a.quick:
        a.models, a.workers, a.updates, a.batch = "resnet18_cifar", "1,2", 8, 8
    n_gpu = torch.cuda.device_count()
    data_dir = os.environ.get("DATA_DIR", "/data/datasets")
    if not os.path.exists(os.path.join(data_dir, "cifar10", ".complete")):
        data_dir = None
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = os.path.join(a.out, f"{run_id}{'_quick' if a.quick else ''}")
    os.makedirs(out, exist_ok=True)
    write_json(os.path.join(out, "meta.json"), {"experiment": "E6", "env": env_info("cuda" if n_gpu else "cpu"),
                                                "n_gpu": n_gpu, "args": vars(a), "data_dir": data_dir})
    log = CsvLog(os.path.join(out, "downpour.csv"), FIELDS)
    for model_name in a.models.split(","):
        for W in [int(w) for w in a.workers.split(",")]:
            print(f"=== {model_name} W={W}", flush=True)
            try:
                log.add(**run_once(model_name, W, n_gpu, a.updates, a.batch, a.lr, a.server_threads, data_dir, a.tau_extra_ms))
            except Exception as e:  # noqa: BLE001
                print(f"  FAILED {model_name} W={W}: {e!r}", flush=True)
    dump_dir_to_log(out)


if __name__ == "__main__":
    main()
