"""Plot one benchmark run as the paper's Fig 1-4 (plus extensions) and write a summary.

    python plot.py ../results/laptop/run_20261121_101500
    python plot.py <run folder> --baseline-threads 1 --figdir ../figures/laptop

Figures (PNG, in <run folder>/figures unless --figdir is given):
  fig1_conv_time      conv forward time vs batch size, Caffe-style impl, linear fit   (paper Fig 1)
  fig2_conv_speedup   speedup over the CPU baseline vs batch size                     (paper Fig 2)
  fig3_fc_time        matrix-vector time vs n, log-log, fitted exponent               (paper Fig 3)
  fig4_fc_speedup     matrix-vector speedup over the CPU baseline vs n                (paper Fig 4)
  fig5_conv_impls     time per image of caffe / batched / conv2d, one panel per device (extension)
  fig6_gemm_time      matrix-matrix time vs n, log-log                                (extension)
Summary: summary.md (readable) and summary.csv (one row per metric), the table view of the charts.
"""
import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator  # noqa: E402

# Paper section 3.2.1: slopes of the CPU and GPU lines in its Fig 1 (unit not stated,
# read here as ms per image), Xeon E5-2670 vs NVIDIA GRID K520 with Caffe.
PAPER_CPU_SLOPE = 6.45
PAPER_GPU_SLOPE = 0.16

# Validated categorical palette (light surface); color follows the entity:
# slot 1 = CPU baseline, slot 2 = GPU, slot 3+ = other CPU thread counts.
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
IMPL_LABEL = {"caffe": "kiểu Caffe (im2col + GEMM từng ảnh)",
              "batched": "im2col + GEMM cả batch",
              "conv2d": "thư viện (oneDNN / cuDNN)"}
IMPL_ORDER = ["caffe", "batched", "conv2d"]


def apply_style():
    names = {f.name for f in font_manager.fontManager.ttflist}
    font = "Segoe UI" if "Segoe UI" in names else "DejaVu Sans"
    plt.rcParams.update({
        "font.family": font,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.axisbelow": True,
        "axes.labelcolor": INK2, "axes.labelsize": 10,
        "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlepad": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": INK2,
        "lines.linewidth": 2, "lines.markersize": 6,
    })


# ---------------------------------------------------------------- data

def load_run(run_dir):
    env_path = run_dir / "env.json"
    env = json.loads(env_path.read_text(encoding="utf-8")) if env_path.exists() else {}
    frames = {}
    for name in ("conv", "fc"):
        path = run_dir / f"{name}.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["config"] = np.where(df["device"] == "cuda", "cuda", "cpu-" + df["threads"].astype(str))
            frames[name] = df
    return env, frames


def series_meta(keys, env, baseline):
    """key -> (label, color, sort order). Fixed slots so colors match across machines."""
    gpu = env.get("gpu", "GPU")
    meta = {}
    others = sorted((k for k in keys if k.startswith("cpu-") and k != baseline), key=lambda k: int(k[4:]))
    if baseline in keys:
        meta[baseline] = (f"CPU · {baseline[4:]} luồng", SLOTS[0], 0)
    if "cuda" in keys:
        meta["cuda"] = (f"GPU · {gpu}", SLOTS[1], 1)
    for i, key in enumerate(others):
        meta[key] = (f"CPU · {key[4:]} luồng", SLOTS[min(2 + i, len(SLOTS) - 1)], 2 + i)
    return dict(sorted(meta.items(), key=lambda kv: kv[1][2]))


def linfit(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - float(np.sum((y - pred) ** 2)) / ss_tot if ss_tot > 0 else 1.0
    return float(slope), float(intercept), r2


def powfit(n, t, min_n=1024):
    """Exponent k of t ~ c * n^k, fitted on the larger sizes (small ones are overhead-bound)."""
    mask = n >= min_n
    if mask.sum() < 3:
        mask = np.ones_like(n, dtype=bool)
    k, _ = np.polyfit(np.log(n[mask]), np.log(t[mask]), 1)
    return float(k)


def speedup_table(df, baseline):
    """Rows: size; columns: config; value = baseline time / config time."""
    pivot = df.pivot_table(index="size", columns="config", values="mean_ms")
    if baseline not in pivot:
        return None
    return pivot.rdiv(pivot[baseline], axis=0).drop(columns=baseline)


# ---------------------------------------------------------------- drawing helpers

def end_labels(ax, items, min_gap=0.07):
    """Direct labels right of the plot: items = [(x, y, text, color)].

    Text stays in ink; a thin leader in the series color ties it to the line end.
    Labels are pushed apart vertically so they never overlap.
    """
    if not items:
        return
    ax.get_xlim(), ax.get_ylim()
    to_axes = ax.transAxes.inverted()
    placed = []
    for x, y, text, color in items:
        frac = to_axes.transform(ax.transData.transform((x, y)))[1]
        placed.append([frac, x, y, text, color])
    placed.sort(key=lambda p: p[0])
    for i in range(1, len(placed)):
        placed[i][0] = max(placed[i][0], placed[i - 1][0] + min_gap)
    if placed[-1][0] > 0.98:                  # pushed above the plot: slide the stack back down
        placed[-1][0] = 0.98
        for i in range(len(placed) - 2, -1, -1):
            placed[i][0] = min(placed[i][0], placed[i + 1][0] - min_gap)
    for frac, x, y, text, color in placed:
        ax.annotate(text, xy=(x, y), xycoords="data", xytext=(1.03, frac), textcoords="axes fraction",
                    va="center", ha="left", fontsize=9, color=INK2, annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=color, lw=1, shrinkA=0, shrinkB=4))


def top_legend(ax):
    """One-row legend between the title and the plot, so it never covers data."""
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=4, borderaxespad=0.3,
              handletextpad=0.3, columnspacing=1.6)


def reps(df):
    return int(df["repeats"].max())


def footnote(fig, text, width=135):
    fig.text(0.0, -0.02, textwrap.fill(text, width), ha="left", va="top", fontsize=8, color=MUTED)


def int_ticks(ax, values, log=False):
    """Ticks exactly at the measured sizes, as plain integers (no thousands separator)."""
    if log:
        ax.set_xscale("log", base=2)
    ax.set_xticks(values)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.xaxis.set_minor_locator(NullLocator())


def pow2(values):
    """Only the power-of-two sizes get a tick label (12288 next to 16384 would collide)."""
    ticks = sorted(int(v) for v in values if int(v) > 0 and int(v) & (int(v) - 1) == 0)
    return ticks or sorted(int(v) for v in values)


def log_y(ax):
    """Log y axis with readable plain-number ticks at 1-2-5 steps."""
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(NullLocator())


def save(fig, figdir, name):
    path = figdir / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("saved", path)


def scatter_line(ax, x, y, color, label, line=True):
    if line:
        ax.plot(x, y, color=color, lw=2, zorder=2)
    ax.plot(x, y, "o", color=color, ms=6, mec=SURFACE, mew=1.5, label=label, zorder=3)


# ---------------------------------------------------------------- figures

def fig_conv_time(conv, meta, figdir, summary):
    df = conv[conv["impl"] == "caffe"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    labels = []
    for key, (label, color, _) in meta.items():
        d = df[df["config"] == key].sort_values("size")
        if d.empty:
            continue
        x, y = d["size"].to_numpy(float), d["mean_ms"].to_numpy(float)
        slope, intercept, r2 = linfit(x, y)
        summary.append(("conv_caffe", key, "slope_ms_per_image", slope))
        summary.append(("conv_caffe", key, "intercept_ms", intercept))
        summary.append(("conv_caffe", key, "r2", r2))
        xs = np.array([0, x.max()])
        ax.plot(xs, slope * xs + intercept, color=color, lw=2, zorder=2)
        scatter_line(ax, x, y, color, label, line=False)
        labels.append((x.max(), slope * x.max() + intercept, f"{label}\n{slope:.3g} ms/ảnh · R² {r2:.3f}", color))
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Batch size (số ảnh)")
    ax.set_ylabel("Thời gian forward (ms)")
    ax.set_title("Hình 1 · Lớp convolution: thời gian forward theo batch size", pad=30)
    top_legend(ax)
    end_labels(ax, labels, min_gap=0.12)
    footnote(fig, "Lớp conv của paper: ảnh xám 224×224, 96 filter 11×11, stride 4. Cài đặt kiểu Caffe "
                  f"(im2col + GEMM từng ảnh), trung bình {reps(df)} lần đo. Đường thẳng: hồi quy tuyến tính, "
                  f"độ dốc = ms/ảnh. Paper: CPU {PAPER_CPU_SLOPE}, GPU {PAPER_GPU_SLOPE}.")
    save(fig, figdir, "fig1_conv_time")


def fig_speedup(df, impl, baseline, meta, figdir, name, title, xlabel, log_x, note):
    table = speedup_table(df[df["impl"] == impl], baseline)
    if table is None or table.empty:
        print(f"skip {name}: need the baseline and at least one other device/thread config")
        return None
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    labels = []
    for key, (label, color, _) in meta.items():
        if key not in table:
            continue
        s = table[key].dropna()
        scatter_line(ax, s.index.to_numpy(float), s.to_numpy(float), color, label)
        labels.append((s.index[-1], s.iloc[-1], f"{label}\n{s.iloc[-1]:.1f}× ở {int(s.index[-1])}", color))
    ax.axhline(1, color=AXIS, lw=0.8, zorder=1)
    if log_x:
        int_ticks(ax, pow2(table.index), log=True)
    else:
        ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"Speedup so với {meta[baseline][0]} (lần)")
    ax.set_title(title, pad=30)
    top_legend(ax)
    end_labels(ax, labels, min_gap=0.12)
    footnote(fig, note)
    save(fig, figdir, name)
    return table


def fig_fc_time(fc, op, meta, figdir, name, title, summary, expected, min_n):
    df = fc[fc["impl"] == op]
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    labels = []
    for key, (label, color, _) in meta.items():
        d = df[df["config"] == key].sort_values("size")
        if d.empty:
            continue
        n, t = d["size"].to_numpy(float), d["mean_ms"].to_numpy(float)
        k = powfit(n, t, min_n)
        summary.append((op, key, "exponent", k))
        summary.append((op, key, "time_ms_at_max_n", t[-1]))
        summary.append((op, key, "gflops_at_max_n", float(d["gflops"].iloc[-1])))
        summary.append((op, key, "gbps_at_max_n", float(d["gbps"].iloc[-1])))
        scatter_line(ax, n, t, color, label)
        labels.append((n[-1], t[-1], f"{label}\nt $\\propto$ n$^{{{k:.2f}}}$", color))
    log_y(ax)
    int_ticks(ax, pow2(df["size"].unique()), log=True)
    ax.set_xlabel("Kích thước n (ma trận n×n)")
    ax.set_ylabel("Thời gian (ms, thang log)")
    ax.set_title(title, pad=30)
    top_legend(ax)
    end_labels(ax, labels, min_gap=0.12)
    footnote(fig, f"Trục log–log: độ dốc của đường là số mũ k trong t ~ n^k (lý thuyết: k = {expected}), "
                  f"khớp trên các n ≥ {min_n}. Trung bình {reps(df)} lần đo.")
    save(fig, figdir, name)


def fig_conv_impls(conv, meta, figdir):
    keys = [k for k in meta if k in set(conv["config"])]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 4.0), squeeze=False)
    for ax, key in zip(axes[0], keys):
        d = conv[conv["config"] == key]
        for i, impl in enumerate(IMPL_ORDER):
            s = d[d["impl"] == impl].sort_values("size")
            if s.empty:
                continue
            per_img = s["mean_ms"].to_numpy(float) / s["size"].to_numpy(float)
            scatter_line(ax, s["size"].to_numpy(float), per_img, SLOTS[i], IMPL_LABEL[impl])
        log_y(ax)
        ax.set_xlim(left=0)
        ax.set_title(meta[key][0], fontsize=11)
        ax.set_xlabel("Batch size")
    axes[0][0].set_ylabel("Thời gian mỗi ảnh (ms, thang log)")
    fig.subplots_adjust(top=0.74, wspace=0.28)
    fig.suptitle("Hình 5 · So sánh ba cách cài đặt lớp convolution (mở rộng)", x=0.06, y=0.99,
                 ha="left", fontsize=12, fontweight="bold", color=INK)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.06, 0.93), ncol=3)
    footnote(fig, "Mỗi khung một thiết bị, trục y riêng cho từng khung. Đường càng thấp càng nhanh; "
                  "đường giảm dần theo batch size nghĩa là gộp batch giúp tận dụng phần cứng tốt hơn.",
             width=60 * len(keys))
    save(fig, figdir, "fig5_conv_impls")


# ---------------------------------------------------------------- summary

def write_summary(run_dir, figdir, env, meta, summary, conv_speed, fc_speed, baseline):
    rows = pd.DataFrame(summary, columns=["metric_group", "config", "metric", "value"])
    rows["label"] = rows["config"].map(lambda k: meta.get(k, (k,))[0])
    rows.to_csv(figdir / "summary.csv", index=False)

    def get(group, key, metric):
        hit = rows[(rows.metric_group == group) & (rows.config == key) & (rows.metric == metric)]
        return float(hit["value"].iloc[0]) if not hit.empty else None

    lines = [f"# Tóm tắt kết quả — {env.get('machine', run_dir.name)} / {run_dir.name}", "",
             f"- CPU: {env.get('cpu', '?')} (torch dùng tối đa {env.get('torch_default_threads', '?')} luồng)",
             f"- GPU: {env.get('gpu', 'không có')}"
             + (f" · CUDA {env.get('cuda')} · cuDNN {env.get('cudnn')}" if env.get("cuda_available") else ""),
             f"- PyTorch {env.get('torch', '?')} · Python {env.get('python', '?')} · {env.get('os', '')}", "",
             "## Thí nghiệm 1 — lớp convolution, cài đặt kiểu Caffe (so với Hình 1–2 của paper)", "",
             "| Cấu hình | Độ dốc (ms/ảnh) | R² | Tỷ lệ độ dốc so với baseline |",
             "|---|---|---|---|",
             f"| *Paper: Xeon E5-2670 (CPU)* | *{PAPER_CPU_SLOPE}* | | *1×* |",
             f"| *Paper: GRID K520 (GPU)* | *{PAPER_GPU_SLOPE}* | | *{PAPER_CPU_SLOPE / PAPER_GPU_SLOPE:.1f}×* |"]
    base_slope = get("conv_caffe", baseline, "slope_ms_per_image")
    for key, (label, _, _) in meta.items():
        slope = get("conv_caffe", key, "slope_ms_per_image")
        if slope is None:
            continue
        ratio = f"{base_slope / slope:.1f}×" if base_slope else ""
        lines.append(f"| {label} | {slope:.4g} | {get('conv_caffe', key, 'r2'):.4f} | {ratio} |")
    if conv_speed is not None:
        lines += ["", "Speedup theo batch size (thời gian baseline / thời gian cấu hình):", "",
                  "| Batch | " + " | ".join(meta[k][0] for k in conv_speed.columns) + " |",
                  "|---" * (len(conv_speed.columns) + 1) + "|"]
        for size, row in conv_speed.iterrows():
            lines.append(f"| {size} | " + " | ".join(f"{v:.2f}×" for v in row) + " |")

    lines += ["", "## Thí nghiệm 2 — lớp fully-connected (so với Hình 3–4 của paper)", "",
              "| Cấu hình | Phép | Số mũ k (t ∝ nᵏ) | Thời gian ở n lớn nhất (ms) | GFLOP/s | GB/s |",
              "|---|---|---|---|---|---|"]
    for op, expected in (("matvec", 2), ("gemm", 3)):
        for key, (label, _, _) in meta.items():
            k = get(op, key, "exponent")
            if k is None:
                continue
            lines.append(f"| {label} | {op} (lý thuyết k={expected}) | {k:.2f} | "
                         f"{get(op, key, 'time_ms_at_max_n'):.3f} | {get(op, key, 'gflops_at_max_n'):.1f} | "
                         f"{get(op, key, 'gbps_at_max_n'):.1f} |")
    if fc_speed is not None:
        lines += ["", "Speedup matvec theo n:", "",
                  "| n | " + " | ".join(meta[k][0] for k in fc_speed.columns) + " |",
                  "|---" * (len(fc_speed.columns) + 1) + "|"]
        for size, row in fc_speed.iterrows():
            lines.append(f"| {size} | " + " | ".join(f"{v:.2f}×" for v in row) + " |")
    lines += ["", "Ghi chú: paper không ghi đơn vị độ dốc; ở đây hiểu là ms/ảnh. "
              "Con số tuyệt đối khác paper vì phần cứng khác đời; điều cần so là xu hướng."]
    path = figdir / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("saved", path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--baseline-threads", type=int, default=1,
                        help="CPU thread count used as the baseline (paper: single core)")
    parser.add_argument("--figdir", type=Path, default=None)
    # Matvec is memory-bound: below ~4096 the matrix (<= 16 MiB at n=2048) still fits in
    # the last-level cache, so the fit starts where the data streams from DRAM.
    parser.add_argument("--fit-min-n-matvec", type=int, default=4096)
    parser.add_argument("--fit-min-n-gemm", type=int, default=1024)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    figdir = (args.figdir or run_dir / "figures").resolve()
    figdir.mkdir(parents=True, exist_ok=True)
    env, frames = load_run(run_dir)
    if not frames:
        raise SystemExit(f"no conv.csv / fc.csv in {run_dir}")
    apply_style()

    keys = set().union(*(set(df["config"]) for df in frames.values()))
    baseline = f"cpu-{args.baseline_threads}"
    if baseline not in keys:
        cpu = sorted((k for k in keys if k.startswith("cpu-")), key=lambda k: int(k[4:]))
        baseline = cpu[0] if cpu else sorted(keys)[0]
    meta = series_meta(keys, env, baseline)

    summary, conv_speed, fc_speed = [], None, None
    if "conv" in frames:
        conv = frames["conv"]
        fig_conv_time(conv, meta, figdir, summary)
        conv_speed = fig_speedup(conv, "caffe", baseline, meta, figdir, "fig2_conv_speedup",
                                 "Hình 2 · Lớp convolution: speedup theo batch size",
                                 "Batch size (số ảnh)", False,
                                 "Speedup = thời gian của baseline / thời gian của cấu hình, cùng batch size, "
                                 "cài đặt kiểu Caffe. Đường ngang: 1× (bằng baseline).")
        fig_conv_impls(conv, meta, figdir)
    if "fc" in frames:
        fc = frames["fc"]
        fig_fc_time(fc, "matvec", meta, figdir, "fig3_fc_time",
                    "Hình 3 · Lớp fully-connected (nhân ma trận–vector): thời gian theo n", summary, 2,
                    args.fit_min_n_matvec)
        fc_speed = fig_speedup(fc, "matvec", baseline, meta, figdir, "fig4_fc_speedup",
                               "Hình 4 · Nhân ma trận–vector: speedup theo n",
                               "Kích thước n (ma trận n×n)", True,
                               "Speedup = thời gian của baseline / thời gian của cấu hình, cùng n. "
                               "Đường ngang: 1× (bằng baseline).")
        fig_fc_time(fc, "gemm", meta, figdir, "fig6_gemm_time",
                    "Hình 6 · Nhân ma trận–ma trận: thời gian theo n (mở rộng)", summary, 3,
                    args.fit_min_n_gemm)
    write_summary(run_dir, figdir, env, meta, summary, conv_speed, fc_speed, baseline)


if __name__ == "__main__":
    main()
