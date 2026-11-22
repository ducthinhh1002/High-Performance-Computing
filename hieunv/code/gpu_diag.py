"""Why does the per-image GPU time change during a sweep? One table per run.

    python gpu_diag.py <run> [<run> ...] --out gpu_diag.md

For the Caffe-style conv on the GPU, in measurement order: per-image wall time, per-image
host issue time (launching kernels) and the mean SM clock NVML reported in that interval.
  wall changes together with the SM clock        -> the GPU clock is the cause
  wall ~ issue and both change, clock constant   -> the host (kernel launches) is the cause
"""
import argparse
from pathlib import Path

import pandas as pd


def clocks(run_dir):
    path = run_dir / "gpu_clocks.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, on_bad_lines="skip").apply(pd.to_numeric, errors="coerce").dropna()
    return None if df.empty else df


def table(run_dir):
    conv = pd.read_csv(run_dir / "conv.csv")
    d = conv[(conv["device"] == "cuda") & (conv["impl"] == "caffe")].sort_values("order")
    log = clocks(run_dir)
    lines = [f"## {run_dir.name}", "",
             "| Thứ tự đo | Batch | Wall (µs/ảnh) | Issue (µs/ảnh) | Issue/Wall | SM clock (MHz) | Số mẫu |",
             "|---|---|---|---|---|---|---|"]
    for _, r in d.iterrows():
        sm, count = "?", 0
        if log is not None:
            hit = log[(log["t"] >= r["t_start"]) & (log["t"] <= r["t_end"])]
            if hit.empty:                       # shorter than the 20 ms sampling: take the nearest
                hit = log.iloc[[(log["t"] - r["t_start"]).abs().argmin()]]
            else:
                count = len(hit)
            sm = f"{hit['sm_mhz'].mean():.0f}"
        wall, issue = r["mean_ms"] / r["size"] * 1000, r["issue_ms"] / r["size"] * 1000
        lines.append(f"| {int(r['order'])} | {int(r['size'])} | {wall:.1f} | {issue:.1f} | "
                     f"{issue / wall:.2f} | {sm} | {count} |")
    return lines + ["", "Số mẫu 0: khoảng đo ngắn hơn 20 ms, lấy mẫu xung nhịp gần nhất.", ""]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    text = "\n".join(["# Chẩn đoán GPU: conv kiểu Caffe", ""] + [l for r in args.runs for l in table(r)])
    args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
