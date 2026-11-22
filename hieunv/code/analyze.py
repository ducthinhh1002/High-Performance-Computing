"""Stage 5: compare the official runs with the paper (its figures were digitized) and test explanations.

    python analyze.py                    # reads ../results, writes ../results/analysis

Inputs : results/kaggle/{run1,run2,run3,agg}_*, results/laptop/{run1..3,agg}_*, results/paper_digitized.csv
Output : tables.md (every number used in ANALYSIS.md), and
  fig7_speedup_vs_batch_paper    speedup normalized to batch 256, ours vs the paper's Fig 2
  fig8_matvec_speedup_paper      matvec speedup vs n, ours vs the paper's Fig 4
  fig9_matvec_bandwidth          matvec bytes/second vs n, CPU and GPU
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import plot as pl
from plot import INK2, SLOTS, apply_style, log_y, save, scatter_line, top_legend

HERE = Path(__file__).resolve().parent
KAGGLE, LAPTOP = "20261122_074407", "20261122_152118"
FLOPS_PER_IMAGE = 2 * 96 * 121 * 2916          # multiply-adds of one image of the paper's conv layer
PAPER = dict(cpu_slope=6.45, gpu_slope=0.16)
# Peak numbers are nominal (spec sheets / own arithmetic), only for orientation, not measured:
PEAK = {"K520 1 GPU (danh định)": 2289.0, "T4 (danh định)": 8100.0,   # GFLOP/s FP32
        "T4 bandwidth GB/s": 320.0, "K520 bandwidth GB/s": 160.0}
FP32_PER_CYCLE_T4 = 2 * 2560                   # FMA on 2560 FP32 cores


# ---------------------------------------------------------------- data

def cpu_label(thr):
    return f"CPU {thr} luồng"


def read_runs(base, tag):
    runs = []
    for i in (1, 2, 3):
        d = base / f"run{i}_{tag}"
        if not d.exists():
            raise SystemExit(f"missing {d}")
        runs.append((f"run{i}", d))
    return runs


def series(df, impl, device, thr=None):
    d = df[(df["impl"] == impl) & (df["device"] == device)]
    if thr is not None:
        d = d[d["threads"] == thr]
    return d.set_index("size").sort_index()


def md_table(df, floatfmt="{:.3g}"):
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(v)
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


# ---------------------------------------------------------------- conv

def conv_ratios(conv):
    """Per run: CPU-1T caffe time / GPU caffe time, by batch."""
    return pd.DataFrame({r: series(d, "caffe", "cpu", 1)["mean_ms"] / series(d, "caffe", "cuda")["mean_ms"]
                         for r, d in conv.items()})


def rel_fit(x, t):
    """t = a + b x, least squares on the relative error (so small sizes count too)."""
    w = 1 / t
    A = np.vstack([np.ones_like(x), x]).T * w[:, None]
    (a, b), *_ = np.linalg.lstsq(A, np.ones(len(x)), rcond=None)
    return a, b


def conv_model(conv):
    rows = []
    for r, d in conv.items():
        g, c = series(d, "caffe", "cuda")["mean_ms"], series(d, "caffe", "cpu", 1)["mean_ms"]
        x = g.index.to_numpy(float)
        a, b = rel_fit(x, g.to_numpy())
        ac, bc = rel_fit(x, c.reindex(g.index).to_numpy())
        pred = (ac + bc * x) / (a + b * x)
        meas = (c.reindex(g.index) / g).to_numpy()
        ok = np.array([not (r == "run3" and int(s) == 8) for s in x])      # run 3 batch 8: throttled clock
        rows.append(dict(run=r, gpu_fixed_us=a * 1e3, gpu_us_per_img=b * 1e3, cpu_ms_per_img=bc,
                         max_err_pct=100 * np.max(np.abs(pred / meas - 1)[ok])))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- clocks

def load_clock_log(run_dir):
    path = run_dir / "gpu_clocks.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, on_bad_lines="skip").apply(pd.to_numeric, errors="coerce").dropna()


def window_clock(log, t0, t1):
    hit = log[(log["t"] >= t0) & (log["t"] <= t1)]
    if hit.empty:
        hit = log.iloc[[(log["t"] - t0).abs().argmin()]]
    return hit["sm_mhz"].mean()


def clock_tests(runs):
    """Cycles per image for the conv implementations; GEMM efficiency; matvec bandwidth vs clock."""
    conv_rows, gemm_rows, mv_rows = [], [], []
    for name, d in runs:
        log, conv, fc = load_clock_log(d), pd.read_csv(d / "conv.csv"), pd.read_csv(d / "fc.csv")
        for impl in ("caffe", "batched", "conv2d"):
            s = conv[(conv["device"] == "cuda") & (conv["impl"] == impl) & (conv["size"] >= 64)]
            k = [x.mean_ms / x["size"] * window_clock(log, x.t_start, x.t_end) for _, x in s.iterrows()]
            conv_rows.append(dict(run=name, impl=impl, us_per_img=float((s["mean_ms"] / s["size"]).median() * 1e3),
                                  mhz=float(np.median([window_clock(log, x.t_start, x.t_end) for _, x in s.iterrows()])),
                                  kcycles_per_img=float(np.median(k))))
            us, mhz = conv_rows[-1]["us_per_img"], conv_rows[-1]["mhz"]
            conv_rows[-1]["gflops"] = FLOPS_PER_IMAGE / us / 1e3
            conv_rows[-1]["peak_pct"] = 100 * conv_rows[-1]["gflops"] / (FP32_PER_CYCLE_T4 / 1000 * mhz)
        for op, rows_, col, sizes in (("gemm", gemm_rows, "gflops", (2048, 3072, 4096)),
                                      ("matvec", mv_rows, "gbps", (8192, 12288, 16384))):
            s = fc[(fc["device"] == "cuda") & (fc["impl"] == op) & (fc["size"].isin(sizes))]
            for _, x in s.sort_values("size").iterrows():
                mhz = window_clock(log, x.t_start, x.t_end)
                rows_.append(dict(run=name, n=int(x["size"]), mhz=mhz, value=float(x[col])))
    return pd.DataFrame(conv_rows), pd.DataFrame(gemm_rows), pd.DataFrame(mv_rows)


# ---------------------------------------------------------------- parallel scaling

def scaling(conv, fc, p):
    """Speedup S, efficiency E, and the serial fraction f solving Amdahl's law S = 1/(f+(1-f)/p)."""
    rows = []
    for what, df, impl, size in (("conv kiểu Caffe", conv, "caffe", None), ("conv batched", conv, "batched", None),
                                 ("conv2d (oneDNN)", conv, "conv2d", None), ("GEMM n=4096", fc, "gemm", 4096),
                                 ("matvec n=16384", fc, "matvec", 16384)):
        t1, tp = series(df, impl, "cpu", 1)["mean_ms"], series(df, impl, "cpu", p)["mean_ms"]
        if size is None:                                # slope of the straight line over the batches
            a, b1 = np.polyfit(t1.index, t1.to_numpy(), 1)[::-1]
            _, bp = np.polyfit(tp.index, tp.to_numpy(), 1)[::-1]
            s1, sp = b1, bp
        else:
            s1, sp = t1.loc[size], tp.loc[size]
        S = s1 / sp
        rows.append({"phép tính": what, "S": S, "E = S/p": S / p, "f (Amdahl)": (1 / S - 1 / p) / (1 - 1 / p)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=HERE.parent / "results")
    args = ap.parse_args()
    res = args.results
    out = res / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    kr, lr = read_runs(res / "kaggle", KAGGLE), read_runs(res / "laptop", LAPTOP)
    kconv = {n: pd.read_csv(d / "conv.csv") for n, d in kr}
    kfc = {n: pd.read_csv(d / "fc.csv") for n, d in kr}
    k_agg_fc = pd.read_csv(res / "kaggle" / f"agg_{KAGGLE}" / "fc.csv")
    k_agg_conv = pd.read_csv(res / "kaggle" / f"agg_{KAGGLE}" / "conv.csv")
    l_agg_conv = pd.read_csv(res / "laptop" / f"agg_{LAPTOP}" / "conv.csv")
    l_agg_fc = pd.read_csv(res / "laptop" / f"agg_{LAPTOP}" / "fc.csv")
    paper = pd.read_csv(res / "paper_digitized.csv", comment="#")
    lines = ["# Bảng số cho phân tích giai đoạn 5 (sinh bởi code/analyze.py)", ""]

    # 1. Speedup vs batch, normalized to batch 256 -------------------------------------------
    ratios = conv_ratios(kconv)                                   # rows: batch, columns: run
    norm = ratios / ratios.loc[256]
    p2 = paper[(paper["figure"] == "fig2")].set_index("x")["value"]
    t = pd.DataFrame({"batch": ratios.index,
                      "tỷ lệ đo lại (trung vị 3 lần)": ratios.median(axis=1).to_numpy(),
                      "nhỏ nhất–lớn nhất": [f"{a:.1f}–{b:.1f}" for a, b in zip(ratios.min(axis=1), ratios.max(axis=1))],
                      "đo lại / batch 256": norm.median(axis=1).to_numpy(),
                      "bài báo (Hình 2)": [p2.get(b, np.nan) for b in ratios.index],
                      "bài báo / batch 256": [p2.get(b, np.nan) / p2.loc[256] for b in ratios.index]})
    lines += ["## 1. Speedup CPU-1 luồng / GPU theo batch (conv kiểu Caffe, Kaggle) và bài báo", "", md_table(t, "{:.2f}"), ""]

    # 2. Linearity and slope ratio ------------------------------------------------------------
    rows = []
    for r, d in kconv.items():
        g, c = series(d, "caffe", "cuda")["mean_ms"], series(d, "caffe", "cpu", 1)["mean_ms"]
        sg, sc = np.polyfit(g.index, g, 1)[0], np.polyfit(c.index, c, 1)[0]
        r2 = lambda s: 1 - ((s - np.polyval(np.polyfit(s.index, s, 1), s.index)) ** 2).sum() / ((s - s.mean()) ** 2).sum()  # noqa: E731
        rows.append(dict(run=r, cpu_slope=sc, gpu_slope=sg, ratio=sc / sg, r2_cpu=r2(c), r2_gpu=r2(g)))
    t = pd.DataFrame(rows)
    lines += ["## 2. Độ dốc conv theo từng lần chạy (Kaggle)", "", md_table(t, "{:.4f}"), "",
              f"Bài báo: {PAPER['cpu_slope']} và {PAPER['gpu_slope']} ms/ảnh, tỷ lệ {PAPER['cpu_slope'] / PAPER['gpu_slope']:.1f}. "
              f"Tốc độ FLOP của conv: CPU bài báo {FLOPS_PER_IMAGE / (PAPER['cpu_slope'] * 1e6):.1f}, "
              f"GPU bài báo {FLOPS_PER_IMAGE / (PAPER['gpu_slope'] * 1e6):.0f} GFLOP/s; "
              f"CPU-1 luồng đo lại {FLOPS_PER_IMAGE / (series(k_agg_conv, 'caffe', 'cpu', 1)['mean_ms'].iloc[-1] / 256 * 1e6):.1f}, "
              f"GPU đo lại {FLOPS_PER_IMAGE / (np.polyfit(series(k_agg_conv, 'caffe', 'cuda').index, series(k_agg_conv, 'caffe', 'cuda')['mean_ms'], 1)[0] * 1e6):.0f} GFLOP/s.", ""]

    # 3. Two-parameter model of the GPU time ---------------------------------------------------
    m = conv_model(kconv)
    lines += ["## 3. Mô hình t_GPU(B) = a + b·B cho conv kiểu Caffe (khớp theo sai số tương đối)", "", md_table(m, "{:.3g}"), "",
              "max_err_pct: sai lệch lớn nhất giữa speedup do mô hình dự đoán và đo được (không tính lần 3 batch 8 bị hạ xung).", ""]

    # 4. Matvec speedup vs n with the paper ---------------------------------------------------
    fc_ratio = pd.DataFrame({r: series(d, "matvec", "cpu", 1)["mean_ms"] / series(d, "matvec", "cuda")["mean_ms"]
                             for r, d in kfc.items()})
    p4 = paper[(paper["figure"] == "fig4")].set_index("x")["value"]
    t = pd.DataFrame({"n": fc_ratio.index, "đo lại (trung vị 3 lần)": fc_ratio.median(axis=1).to_numpy(),
                      "nhỏ nhất–lớn nhất": [f"{a:.2f}–{b:.2f}" for a, b in zip(fc_ratio.min(axis=1), fc_ratio.max(axis=1))],
                      "bài báo (Hình 4)": [p4.get(n, np.nan) for n in fc_ratio.index]})
    lines += ["## 4. Speedup matvec CPU-1 luồng / GPU theo n (Kaggle) và bài báo", "", md_table(t, "{:.2f}"), ""]

    # 5. GB/s and ns per element --------------------------------------------------------------
    cpu1, gpu = series(k_agg_fc, "matvec", "cpu", 1), series(k_agg_fc, "matvec", "cuda")
    lcpu1 = series(l_agg_fc, "matvec", "cpu", 1)
    p3 = paper[paper["figure"] == "fig3"].set_index("x")["value"]
    rows = []
    for n in (2048, 4096, 8192):
        pcpu = p3.get(n, np.nan)
        pgpu = pcpu / p4.get(n)
        rows.append({"n": n, "bài báo CPU (ms)": pcpu, "bài báo CPU ns/phần tử": pcpu * 1e6 / n ** 2,
                     "bài báo CPU GB/s": 4e-6 * n ** 2 / pcpu, "bài báo GPU GB/s (suy ra)": 4e-6 * n ** 2 / pgpu,
                     "đo lại CPU-1 luồng ms": cpu1.loc[n, "mean_ms"], "đo lại CPU GB/s": cpu1.loc[n, "gbps"],
                     "đo lại GPU GB/s": gpu.loc[n, "gbps"], "laptop CPU GB/s": lcpu1.loc[n, "gbps"]})
    lines += ["## 5. Matvec: số tuyệt đối so với bài báo (GB/s tính theo 4 byte mỗi phần tử ma trận)", "", md_table(pd.DataFrame(rows), "{:.3g}"), ""]
    # 5b. Hypothesis: the paper's "FC" is (n x n) * (n x B) as its caption says ("matrix-matrix").
    rows = []
    for n in (2048, 4096, 8192):
        tc = p3.get(n, np.nan)
        tg = tc / p4.get(n)
        rows.append({"n": n, "CPU GFLOP/s nếu B = 1": 2e-9 * n * n / (tc * 1e-3), "B để CPU bằng conv (10,5 GFLOP/s)": 10.5 * tc * 1e-3 / (2e-9 * n * n),
                     "CPU GFLOP/s nếu B = 64": 2e-9 * n * n * 64 / (tc * 1e-3), "GPU GFLOP/s nếu B = 64": 2e-9 * n * n * 64 / (tg * 1e-3),
                     "CPU GFLOP/s nếu B = 76": 2e-9 * n * n * 76 / (tc * 1e-3), "GPU GFLOP/s nếu B = 76": 2e-9 * n * n * 76 / (tg * 1e-3)})
    lines += ["## 5b. Giả thuyết: FC của bài báo là (n×n)·(n×B) (\"matrix-matrix\"), với B vector", "",
              md_table(pd.DataFrame(rows), "{:.3g}"), "",
              f"Trần danh định của K520 (1 GPU) là {PEAK['K520 1 GPU (danh định)']:.0f} GFLOP/s FP32.", ""]

    t = pd.DataFrame({"n": cpu1.index, "Kaggle CPU-1 luồng GB/s": cpu1["gbps"].to_numpy(),
                      "Kaggle CPU-2 luồng GB/s": series(k_agg_fc, "matvec", "cpu", 2)["gbps"].reindex(cpu1.index).to_numpy(),
                      "Kaggle GPU GB/s": gpu["gbps"].to_numpy(), "laptop CPU-1 luồng GB/s": lcpu1["gbps"].reindex(cpu1.index).to_numpy(),
                      "laptop CPU-8 luồng GB/s": series(l_agg_fc, "matvec", "cpu", 8)["gbps"].reindex(cpu1.index).to_numpy()})
    lines += ["## 6. Băng thông matvec theo n", "", md_table(t, "{:.1f}"), ""]

    # 7. Clock tests --------------------------------------------------------------------------
    conv_c, gemm_c, mv_c = clock_tests(kr)
    t = conv_c.groupby("impl")[["us_per_img", "mhz", "kcycles_per_img", "gflops", "peak_pct"]].median().reindex(["caffe", "batched", "conv2d"]).reset_index()
    lines += ["## 7. Chu kỳ GPU mỗi ảnh theo cách cài đặt (batch ≥ 64, trung vị 3 lần; xung nhịp trong đúng khoảng đo)", "", md_table(t, "{:.1f}"), ""]
    lines += ["### Từng lần", "", md_table(conv_c.round(1), "{:.1f}"), ""]
    gemm_c["peak_pct"] = 100 * gemm_c["value"] / (FP32_PER_CYCLE_T4 / 1000 * gemm_c["mhz"])
    lines += ["## 8. GEMM trên GPU: tỷ lệ so với trần FP32 ở đúng xung nhịp đo được", "", md_table(gemm_c.round(1), "{:.1f}"), ""]
    lines += ["## 9. Matvec GPU: GB/s ở các xung nhịp SM khác nhau (xung nhịp bộ nhớ trong các khoảng đo đều 5000 MHz)", "", md_table(mv_c.round(1), "{:.1f}"), ""]

    # 10. Parallel scaling -------------------------------------------------------------------
    lines += ["## 10. Tăng tốc CPU: Kaggle 2 luồng (p=2)", "", md_table(scaling(k_agg_conv, k_agg_fc, 2), "{:.3g}"), "",
              "## 11. Tăng tốc CPU: laptop 8 luồng (p=8)", "", md_table(scaling(l_agg_conv, l_agg_fc, 8), "{:.3g}"), ""]

    (out / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # ------------------------------------------------------------ figures
    apply_style()
    plot_speedup_batch(ratios, norm, p2, out)
    plot_matvec(fc_ratio, p4, out)
    plot_bandwidth(cpu1, gpu, lcpu1, out)


def plot_speedup_batch(ratios, norm, p2, out):
    fig, ax = pl.plt.subplots(figsize=(7.2, 4.4))
    med, lo, hi = norm.median(axis=1), norm.min(axis=1), norm.max(axis=1)
    x = norm.index.to_numpy(float)
    ax.errorbar(x, med, yerr=[med - lo, hi - med], fmt="none", ecolor=SLOTS[0], elinewidth=1, capsize=3, zorder=2)
    scatter_line(ax, x, med.to_numpy(), SLOTS[0], "Đo lại trên Kaggle (trung vị 3 lần, thanh = nhỏ nhất–lớn nhất)")
    ax.plot(p2.index, (p2 / p2.loc[256]).to_numpy(), "s--", color=SLOTS[2], ms=7, mec=pl.SURFACE, mew=1.5, lw=2,
            label="Bài báo, Hình 2 (đọc từ hình)", zorder=3)
    ax.annotate("lần 3, batch 8:\nGPU hạ xung", xy=(8, lo.loc[8]), xytext=(13, lo.loc[8]), fontsize=8.5, color=INK2,
                va="center", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    pl.int_ticks(ax, [1, 8, 16, 32, 64, 128, 256], log=True)
    ax.set_xlim(0.8, 330)
    ax.set_ylim(0, 1.15)
    ax.axhline(1, color=pl.AXIS, lw=0.8, zorder=0)
    ax.set_xlabel("Batch size (thang log)")
    ax.set_ylabel("Speedup GPU/CPU so với ở batch 256")
    ax.set_title("Hình 7 · Speedup convolution theo batch: đo lại so với bài báo", pad=30)
    top_legend(ax)
    pl.footnote(fig, "Chuẩn hóa về batch 256 vì hai bên khác giá trị tuyệt đối (bài báo 40,8×; đo lại 27,4×). Bài báo chỉ đo batch ≥ 16.")
    save(fig, out, "fig7_speedup_vs_batch_paper")


def plot_matvec(fc_ratio, p4, out):
    fig, ax = pl.plt.subplots(figsize=(7.2, 4.4))
    med, lo, hi = fc_ratio.median(axis=1), fc_ratio.min(axis=1), fc_ratio.max(axis=1)
    x = fc_ratio.index.to_numpy(float)
    ax.errorbar(x, med, yerr=[med - lo, hi - med], fmt="none", ecolor=SLOTS[0], elinewidth=1, capsize=3, zorder=2)
    scatter_line(ax, x, med.to_numpy(), SLOTS[0], "Đo lại trên Kaggle (trung vị 3 lần, thanh = nhỏ nhất–lớn nhất)")
    ax.annotate("Đo lại", xy=(x[-1], med.iloc[-1]), xytext=(6, -10), textcoords="offset points", fontsize=9, color=INK2)
    ax.annotate("Bài báo", xy=(8192, p4.loc[8192]), xytext=(6, 2), textcoords="offset points", fontsize=9, color=INK2)
    ax.plot(p4.index, p4.to_numpy(), "s--", color=SLOTS[2], ms=7, mec=pl.SURFACE, mew=1.5, lw=2,
            label="Bài báo, Hình 4 (đọc từ hình)", zorder=3)
    log_y(ax)
    pl.int_ticks(ax, [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384], log=True)
    ax.axhline(1, color=pl.AXIS, lw=0.8, zorder=0)
    ax.set_xlabel("Kích thước n (thang log)")
    ax.set_ylabel("CPU-1 luồng / GPU (lần, thang log)")
    ax.set_title("Hình 8 · Speedup matvec theo n: đo lại so với bài báo", pad=30)
    top_legend(ax)
    pl.footnote(fig, "Trên 1 = GPU nhanh hơn CPU. Bài báo: đường đứt; không đơn điệu ở n = 1024–2048 và ở n nhỏ.")
    save(fig, out, "fig8_matvec_speedup_paper")


def plot_bandwidth(cpu1, gpu, lcpu1, out):
    fig, ax = pl.plt.subplots(figsize=(7.2, 4.4))
    for s, color, label, mk in ((cpu1, SLOTS[0], "Kaggle, CPU 1 luồng", "o"), (gpu, SLOTS[1], "Kaggle, GPU T4", "o"),
                                (lcpu1, SLOTS[2], "Laptop, CPU 1 luồng", "o")):
        scatter_line(ax, s.index.to_numpy(float), s["gbps"].to_numpy(), color, label)
    ax.axhline(PEAK["T4 bandwidth GB/s"], color=INK2, lw=0.8, ls=":", zorder=1)
    ax.annotate("băng thông danh định của T4: 320 GB/s", xy=(256, 320), xytext=(0, 4), textcoords="offset points", fontsize=8.5, color=INK2)
    log_y(ax)
    pl.int_ticks(ax, [256, 512, 1024, 2048, 4096, 8192, 16384], log=True)
    ax.set_ylim(8, 600)
    ax.set_xlabel("Kích thước n (thang log)")
    ax.set_ylabel("Băng thông (GB/s, thang log)")
    ax.set_title("Hình 9 · Băng thông hiệu dụng của matvec theo n", pad=30)
    top_legend(ax)
    pl.end_labels(ax, [(s.index[-1], s["gbps"].iloc[-1], lab, col) for s, col, lab in
                       ((cpu1, SLOTS[0], "CPU Kaggle"), (gpu, SLOTS[1], "GPU T4"), (lcpu1, SLOTS[2], "CPU laptop"))])
    pl.footnote(fig, "Băng thông = 4 byte × n² / thời gian, tức tính từ chính thời gian đo (không phải phép đo độc lập). "
                     "CPU giảm khi ma trận lớn (có thể do vượt bộ đệm); GPU tăng vì chi phí cố định được chia nhỏ.")
    save(fig, out, "fig9_matvec_bandwidth")


if __name__ == "__main__":
    main()
