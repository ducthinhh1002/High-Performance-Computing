"""Analyze the bandwidth checks (PREDICTIONS.md) against the official runs.

    python analyze_mem.py                # reads ../results, writes ../results/analysis/mem_tables.md + fig10

Runs: results/kaggle/bw{1,2,3}_<stamp>. Tables: bandwidth kernels (D1-D3), FC with B vectors (D4), conv steps (D5),
and the ratio of measured bandwidths vs the matvec speedup.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import plot as pl
from plot import INK2, MUTED, SLOTS, apply_style, save

HERE = Path(__file__).resolve().parent
STAMP = "20261122_095300"
FLOPS_PER_IMAGE = 2 * 96 * 121 * 2916
PEAK_PER_GHZ = 5.12e3                  # FP32 GFLOP/s per GHz of SM clock on a T4 (2560 cores x 2)


def md_table(df, fmt="{:.3g}"):
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else fmt.format(v)
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    return "\n".join([head, sep] + ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)])


def load(res):
    runs = [res / "kaggle" / f"bw{i}_{STAMP}" for i in (1, 2, 3)]
    out = {}
    for name in ("bw", "fcb", "step"):
        out[name] = pd.concat([pd.read_csv(r / f"{name}.csv").assign(run=r.name[:3], runpath=str(r)) for r in runs],
                              ignore_index=True)
    return out


def spread(g):
    return f"{g.min():.1f}–{g.max():.1f}"


def clock_in(run_dir, t0, t1):
    log = pd.read_csv(Path(run_dir) / "gpu_clocks.csv", on_bad_lines="skip").apply(pd.to_numeric, errors="coerce").dropna()
    h = log[(log["t"] >= t0) & (log["t"] <= t1)]
    if h.empty:
        h = log.iloc[[(log["t"] - t0).abs().argmin()]]
    return h["sm_mhz"].mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=HERE.parent / "results")
    args = ap.parse_args()
    res = args.results
    out = res / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    d = load(res)
    bw, fcb, step = d["bw"], d["fcb"], d["step"]
    lines = ["# Bảng số của lần kiểm chứng băng thông (Kaggle, 3 lần chạy, sinh bởi code/analyze_mem.py)", "",
             "Băng thông tính theo byte của từng kernel: sum 4, dot 8, copy 8, triad 12 (STREAM), matvec 4 byte mỗi phần tử; mảng N = n².", ""]

    # 1. all kernels
    t = bw.pivot_table(index=["device", "threads", "impl"], columns="size", values="gbps", aggfunc="mean").round(1).reset_index()
    t.columns = [str(c) for c in t.columns]
    lines += ["## 1. Băng thông trung bình (GB/s)", "", md_table(t, "{:.1f}"), ""]

    # 2. best independent read kernel vs matvec
    rows = []
    for dev, thr in (("cpu", 1), ("cpu", 2), ("cuda", 2)):
        for n in (1024, 4096, 8192, 16384):
            s = bw[(bw.device == dev) & (bw.threads == thr) & (bw["size"] == n)]
            m = s.pivot_table(index="run", columns="impl", values="gbps")
            best = m[["sum", "dot"]].max(axis=1)
            rows.append({"thiết bị": f"CPU {thr} luồng" if dev == "cpu" else "GPU", "n": n,
                         "đọc tốt nhất (max của sum, dot)": best.mean(), "từng lần": spread(best),
                         "matvec": m["matvec"].mean(), "matvec / đọc tốt nhất": (m["matvec"] / best).mean(),
                         "triad": m["triad"].mean()})
    lines += ["## 2. Matvec so với băng thông đọc tốt nhất (độc lập)", "", md_table(pd.DataFrame(rows), "{:.2f}"), ""]

    # 3. bandwidth ratio vs matvec speedup (GPU / CPU-1T)
    rows = []
    for n in (4096, 8192, 16384):
        g = bw[(bw.device == "cuda") & (bw["size"] == n)].pivot_table(index="run", columns="impl", values="gbps")
        c = bw[(bw.device == "cpu") & (bw.threads == 1) & (bw["size"] == n)].pivot_table(index="run", columns="impl", values="gbps")
        rb = g[["sum", "dot"]].max(axis=1) / c[["sum", "dot"]].max(axis=1)
        rm = g["matvec"] / c["matvec"]
        rows.append({"n": n, "tỷ lệ băng thông GPU/CPU-1T": rb.mean(), "từng lần (băng thông)": spread(rb),
                     "tỷ lệ matvec cùng lần đo": rm.mean(), "từng lần (matvec)": spread(rm)})
    lines += ["## 3. Tỷ lệ băng thông (độc lập) so với tỷ lệ matvec", "", md_table(pd.DataFrame(rows), "{:.2f}"), "",
              "Các lần chạy chính thức (Kaggle, trung vị 3 lần): tỷ lệ matvec 18,3 (n=4096), 20,3 (n=8192), 21,7 (n=16384).", ""]

    # 4. FC with B vectors
    rows = []
    for n in (2048, 8192):
        for impl in ("mm1", "mm16", "mm64", "mm128"):
            c = fcb[(fcb.device == "cpu") & (fcb.threads == 1) & (fcb["size"] == n) & (fcb.impl == impl)].set_index("run")
            g = fcb[(fcb.device == "cuda") & (fcb["size"] == n) & (fcb.impl == impl)].set_index("run")
            r = c.mean_ms / g.mean_ms
            mhz = np.mean([clock_in(x.runpath, x.t_start, x.t_end) for _, x in g.reset_index().iterrows()])
            rows.append({"n": n, "B": int(impl[2:]), "tỷ lệ CPU-1T / GPU": r.mean(), "từng lần": " / ".join(f"{x:.1f}" for x in r),
                         "CPU ms": c.mean_ms.mean(), "GPU ms": g.mean_ms.mean(), "CPU GFLOP/s": c.gflops.mean(),
                         "GPU GFLOP/s": g.gflops.mean(), "GPU xung MHz": mhz})
    lines += ["## 4. FC với B vector: (n×n)·(n×B)", "", md_table(pd.DataFrame(rows), "{:.2f}"), ""]

    # 5. conv steps
    step["us"] = step["mean_ms"] / step["size"] * 1e3
    t = step.pivot_table(index=["device", "impl"], columns="size", values="us", aggfunc="mean").round(1).reset_index()
    t.columns = [str(c) for c in t.columns]
    lines += ["## 5. Tách conv kiểu Caffe (µs/ảnh)", "", md_table(t, "{:.1f}"), ""]
    rows = []
    for impl in ("im2col", "gemm", "caffe"):
        s = step[(step.device == "cuda") & (step.impl == impl) & (step["size"] == 256)]
        mhz = np.array([clock_in(x.runpath, x.t_start, x.t_end) for _, x in s.iterrows()])
        kcyc = s["us"].to_numpy() * mhz / 1e3
        gf = FLOPS_PER_IMAGE / (s["us"].to_numpy() * 1e-6) / 1e9
        rows.append({"bước (batch 256, GPU)": impl, "µs/ảnh": s["us"].mean(), "nghìn chu kỳ/ảnh": kcyc.mean(),
                     "từng lần (nghìn chu kỳ)": spread(pd.Series(kcyc)),
                     "% trần FP32 ở xung đo": np.nan if impl == "im2col" else float(np.mean(100 * gf / (PEAK_PER_GHZ * mhz / 1e3)))})
    lines += ["### Chu kỳ ở batch 256 trên GPU (im2col + GEMM so với caffe)", "", md_table(pd.DataFrame(rows), "{:.1f}"), ""]
    cpu = [step[(step.device == "cpu") & (step.impl == i) & (step["size"] == 256)]["us"].mean() for i in ("im2col", "gemm", "caffe")]
    lines += ["CPU 1 luồng, batch 256: im2col %.0f + GEMM %.0f so với caffe %.0f µs/ảnh." % tuple(cpu), ""]
    (out / "mem_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # ------------------------------------------------------------ figure
    apply_style()
    fig, axes = pl.plt.subplots(1, 2, figsize=(10.4, 4.6))
    for ax, (dev, thr, title) in zip(axes, (("cpu", 1, "CPU 1 luồng (Kaggle)"), ("cuda", 2, "GPU T4"))):
        s = bw[(bw.device == dev) & (bw.threads == thr)].pivot_table(index="size", columns="impl", values="gbps", aggfunc="mean")
        for kernel, color, lw, mk, z in (("copy", MUTED, 1.4, "v", 2), ("triad", MUTED, 1.4, "D", 2), ("sum", SLOTS[2], 1.8, "s", 3),
                                         ("dot", SLOTS[2], 1.8, "^", 3), ("matvec", SLOTS[0], 2.6, "o", 4)):
            ax.plot(s.index, s[kernel], color=color, lw=lw, marker=mk, ms=6, zorder=z, ls="-" if kernel == "matvec" else "--")
        ends = sorted(((s[k].iloc[-1], k) for k in s.columns), reverse=True)
        for rank, (v, k) in enumerate(ends):
            ax.annotate(k, xy=(s.index[-1], v), xytext=(8, (len(ends) / 2 - rank) * 9 - 4), textcoords="offset points",
                        fontsize=8.5, color=INK2, va="center")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1024, 4096, 8192, 16384])
        ax.xaxis.set_major_formatter(pl.FuncFormatter(lambda v, _: f"{int(v)}"))
        ax.xaxis.set_minor_locator(pl.NullLocator())
        ax.set_ylim(bottom=0)
        ax.set_xlim(900, 28000)
        ax.set_xlabel("n (mảng N = n² phần tử float32)")
        ax.set_title(title, fontsize=11)
    axes[0].set_ylabel("Băng thông (GB/s)")
    fig.subplots_adjust(top=0.80, wspace=0.22)
    fig.suptitle("Hình 10 · Băng thông bộ nhớ đo độc lập so với matvec", x=0.06, y=0.97, ha="left", fontsize=12, fontweight="bold", color=pl.INK)
    pl.footnote(fig, "Trung bình 3 lần chạy Kaggle (22/11/2026). Sum/dot chỉ đọc; copy và triad vừa đọc vừa ghi. "
                     "Matvec gần sát sum/dot nên phù hợp với giới hạn bởi bộ nhớ.", width=130)
    save(fig, out, "fig10_bandwidth_independent")


if __name__ == "__main__":
    main()
