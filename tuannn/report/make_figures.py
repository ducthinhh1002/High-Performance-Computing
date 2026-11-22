#!/usr/bin/env python3
"""Hình cho báo cáo và slide, vẽ lại từ dữ liệu gốc trong ../experiments/results (đợt 1).

    python report/make_figures.py          # chạy từ thư mục tuannn/
Ghi PNG (200 dpi) vào report/figures/. Chỉ cần numpy, pandas, matplotlib.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results"
OUT = ROOT / "report" / "figures"

# Bảng màu phân loại (thứ tự cố định, đã kiểm tra cho người mù màu) và màu chữ/lưới.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 10, "axes.titlesize": 11,
        "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 6,
    })


def latest(exp: str) -> Path:
    runs = sorted(p for p in (RES / exp).iterdir() if p.is_dir() and p.name[:1].isdigit())
    return runs[-1]


def records(exp: str) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(latest(exp) / "raw" / "*.json"))):
        rows += json.load(open(f))["records"]
    return pd.DataFrame(rows)


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def fig_conv() -> None:
    df = records("dot1_e2_paper_cpu")
    conv = df[df.op == "conv"]
    g = conv[conv.impl == "cpu-blocked"].sort_values("batch")
    d = conv[conv.impl == "cpu-direct"].sort_values("batch")
    b, t = g.batch.values.astype(float), g.time_mean_s.values * 1e3
    slope, icpt = np.polyfit(b, t, 1)
    r2 = 1 - ((t - (slope * b + icpt)) ** 2).sum() / ((t - t.mean()) ** 2).sum()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    a1.plot(b, t, "o", color=C[0], label="đo (im2col + GEMM)")
    xs = np.linspace(0, b.max(), 50)
    a1.plot(xs, slope * xs + icpt, "--", color=INK2, lw=1.2, label=f"khớp tuyến tính: {slope:.2f} ms/ảnh, R² = {r2:.4f}")
    a1.set(title="Conv 96×11×11/4 trên ảnh 224×224 (CPU, 1 luồng)", xlabel="batch size", ylabel="thời gian forward [ms]")
    a1.legend(loc="upper left")
    ratio = d.time_mean_s.values / g.time_mean_s.values
    a2.bar([str(int(x)) for x in b], ratio, color=C[1], width=0.6)
    for i, r in enumerate(ratio):
        a2.text(i, r + 0.3, f"{r:.1f}×", ha="center", color=INK, fontsize=9)
    a2.set(title="Conv trực tiếp chậm hơn im2col + GEMM", xlabel="batch size", ylabel="lần chậm hơn")
    a2.grid(axis="x", visible=False)
    save(fig, "f1_conv_batch")


def fig_gemm() -> None:
    df = records("dot1_e1_gemm")
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for i, (impl, lab) in enumerate((("cpu-blocked", "blocked (chia khối theo cache)"), ("cpu-naive", "naive (3 vòng lặp)"))):
        g = df[df.impl == impl].sort_values("n")
        ax.plot(g.n, g.gflops, "o-", color=C[i], label=lab)
    ax.set(title="GEMM N×N×N, CPU 1 luồng", xlabel="N", ylabel="GFLOP/s")
    ax.set_xscale("log", base=2)
    ax.legend()
    save(fig, "f2_gemm")


def fig_comm() -> None:
    rows, pp = [], []
    for f in sorted(glob.glob(str(latest("dot1_e3_comm") / "raw" / "*.json"))):
        d = json.load(open(f))
        for a in d["allreduce"]:
            rows.append(dict(k=d["world_size"], algo=a["algo"], bytes=a["bytes"], t=a["time_mean_s"]))
    df = pd.DataFrame(rows)
    algos = ["mpi", "ring", "rd", "tree", "naive", "reduce_bcast"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4))
    k4 = df[df.k == 4]
    for i, al in enumerate(algos):
        g = k4[k4.algo == al].sort_values("bytes")
        a1.plot(g.bytes, g.t * 1e6, "o-", ms=4, lw=1.6, color=C[i], label=al)
    a1.set(title="Allreduce, k = 4 tiến trình", xlabel="kích thước message [byte]", ylabel="thời gian [µs]")
    a1.set_xscale("log", base=2)
    a1.set_yscale("log")
    a1.legend(ncol=2, fontsize=9)
    big = df[df.bytes == df.bytes.max()]
    ks = sorted(big.k.unique())
    w = 0.8 / len(algos)
    for i, al in enumerate(algos):
        vals = [big[(big.k == k) & (big.algo == al)].t.values[0] * 1e3 for k in ks]
        a2.bar(np.arange(len(ks)) + (i - 2.5) * w, vals, width=w * 0.9, color=C[i], label=al)
    a2.set_xticks(range(len(ks)), [f"k = {k}" for k in ks])
    a2.set(title="Allreduce 16 MiB theo số tiến trình", ylabel="thời gian [ms]")
    a2.set_yscale("log")
    a2.grid(axis="x", visible=False)
    a2.legend(ncol=2, fontsize=9)
    save(fig, "f3_allreduce")


def fig_scaling() -> None:
    run = latest("dot1_e4_strong_scaling")
    s = pd.read_csv(run / "analysis" / "train_summary.csv").sort_values("np")
    pm = pd.read_csv(run / "analysis" / "perfmodel_train.csv").sort_values("np")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.9))
    k = s.np.values
    a1.plot(k, k, "--", color=INK2, lw=1.2, label="lý tưởng")
    a1.plot(pm.np, pm.predicted_speedup, "s-", color=C[1], ms=5, lw=1.6, label="mô hình α-β")
    err = s.speedup * s.train_time_s_std / s.train_time_s_mean
    a1.errorbar(k, s.speedup, yerr=err, fmt="o-", color=C[0], capsize=3, label="đo (TB 3 lần)")
    for x, y in zip(k, s.speedup):
        a1.annotate(f"{y:.2f}×", (x, y), textcoords="offset points", xytext=(6, -12), fontsize=9, color=INK)
    a1.axvspan(4.2, 8.4, color=GRID, alpha=0.5, lw=0)
    a1.text(6.2, 1.2, "hyper-thread\n(chỉ 4 nhân thật)", ha="center", fontsize=8.5, color=INK2)
    a1.set(title="Strong scaling sync SGD (LeNet/MNIST, batch tổng 256)", xlabel="số tiến trình k", ylabel="speedup")
    a1.set_xticks(k)
    a1.legend(loc="upper left")
    comp = sum(s[f"phase_mean_{p}_mean"] for p in ("data", "h2d", "forward", "backward", "update"))
    wait, comm = s["phase_mean_sync_wait_mean"], s["phase_mean_comm_mean"]
    tot = comp + wait + comm
    labels = [str(x) for x in k]
    a2.bar(labels, comp / tot * 100, color=C[0], width=0.6, label="tính toán")
    a2.bar(labels, wait / tot * 100, bottom=comp / tot * 100, color=C[3], width=0.6, label="chờ đồng bộ")
    a2.bar(labels, comm / tot * 100, bottom=(comp + wait) / tot * 100, color=C[1], width=0.6, label="giao tiếp (allreduce)")
    for i, (w_, c_) in enumerate(zip(wait / tot * 100, comm / tot * 100)):
        a2.text(i, 101, f"{w_ + c_:.1f}%", ha="center", fontsize=9, color=INK)
    a2.set(title="Tỉ lệ thời gian mỗi pha", xlabel="số tiến trình k", ylabel="% thời gian huấn luyện", ylim=(0, 108))
    a2.grid(axis="x", visible=False)
    a2.legend(loc="lower left", fontsize=9)
    save(fig, "f4_strong_scaling")


NAMES = {
    ("sequential", "global", 0, 0): "Tuần tự",
    ("sync", "global", 0, 0): "Sync SGD",
    ("psgd", "iid", 0, 0): "PSGD (TB mỗi epoch)",
    ("psgd", "iid", 10, 0): "PSGD (TB mỗi 10 bước)",
    ("psgd", "sorted", 0, 0): "PSGD (TB mỗi epoch)",
    ("psgd", "sorted", 0, 1): "PSGD + ShuffleSGD",
    ("psgd", "sorted", 10, 0): "PSGD (TB mỗi 10 bước)",
    ("admm", "iid", 0, 0): "ADMM",
    ("admm", "sorted", 0, 0): "ADMM",
    ("downpour", "iid", 0, 0): "Downpour",
    ("downpour", "sorted", 0, 0): "Downpour",
}


def fig_strategies() -> None:
    s = pd.read_csv(latest("dot1_e5_strategies") / "analysis" / "train_summary.csv")
    s["name"] = [NAMES[(r.strategy, r.partition, int(r.avg_period), int(r.reshuffle))] for r in s.itertuples()]
    order = ["Tuần tự", "Sync SGD", "PSGD (TB mỗi epoch)", "PSGD (TB mỗi 10 bước)", "PSGD + ShuffleSGD", "ADMM", "Downpour"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.4, 1]})
    y = np.arange(len(order))
    h = 0.38
    for j, (part, lab, col) in enumerate((("iid", "IID / toàn bộ dữ liệu", C[0]), ("sorted", "không IID (shard theo nhãn)", C[1]))):
        vals, errs = [], []
        for n in order:
            r = s[(s.name == n) & ((s.partition == part) | ((s.partition == "global") & (part == "iid")))]
            vals.append(r.final_test_acc_mean.values[0] * 100 if len(r) else np.nan)
            errs.append(r.final_test_acc_std.values[0] * 100 if len(r) else 0)
        a1.barh(y + (j - 0.5) * h, vals, height=h * 0.92, color=col, xerr=errs, error_kw=dict(ecolor=INK2, lw=1), label=lab)
        for yi, v in zip(y, vals):
            if not np.isnan(v):
                a1.text(101.5, yi + (j - 0.5) * h, f"{v:.1f}", va="center", ha="right", fontsize=8.5, color=INK)
    a1.set_yticks(y, order)
    a1.invert_yaxis()
    a1.set(title="Độ chính xác test sau 3 epoch (TB 2 seed, thanh lỗi = độ lệch chuẩn)", xlabel="test accuracy [%]",
           xlim=(70, 102))
    a1.grid(axis="y", visible=False)
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=9)
    iid = s[s.partition.isin(["iid", "global"])].copy()
    iid = iid[iid.name.isin(["Tuần tự", "Sync SGD", "PSGD (TB mỗi epoch)", "ADMM", "Downpour"])]
    names = ["Tuần tự", "Sync SGD", "PSGD (TB mỗi epoch)", "ADMM", "Downpour"]
    t = [iid[iid.name == n].train_time_s_mean.values[0] for n in names]
    cf = [iid[iid.name == n].comm_fraction_mean.values[0] * 100 for n in names]
    a2.bar(range(len(names)), t, color=C[0], width=0.6)
    for i, (ti, ci) in enumerate(zip(t, cf)):
        a2.text(i, ti + 1.0, f"{ti:.1f} s\n{ci:.1f}% comm", ha="center", fontsize=8, color=INK)
    a2.set_xticks(range(len(names)), ["Tuần tự\n(1 tiến trình)", "Sync", "PSGD", "ADMM", "Downpour"], fontsize=9)
    a2.set(title="Thời gian 3 epoch (IID; song song: 4 tiến trình)", ylabel="giây", ylim=(0, max(t) * 1.25))
    a2.grid(axis="x", visible=False)
    save(fig, "f5_strategies")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    fig_conv()
    fig_gemm()
    fig_comm()
    fig_scaling()
    fig_strategies()


if __name__ == "__main__":
    main()
