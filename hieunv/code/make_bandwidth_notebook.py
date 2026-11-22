"""Build the Kaggle notebook for the bandwidth checks (PREDICTIONS.md): bench_mem.py, 3 runs.

    python make_bandwidth_notebook.py        # -> ../kaggle/hpc_bandwidth_kaggle.ipynb

Separate from hpc_replication_kaggle.ipynb (the replication of the paper), so neither changes the other.
"""
import json
from pathlib import Path

from make_kaggle_notebook import WORK, code, md

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "kaggle" / "hpc_bandwidth_kaggle.ipynb"
FILES = ["common.py", "gpu_sampler.py", "bench_conv.py", "bench_mem.py"]
RUNS, REPEAT = 3, 10


def build():
    cells = [
        md(f"""
# Memory bandwidth and extra checks for the paper replication (Kaggle, T4)

Independent checks of the analysis: STREAM-like bandwidth (not matvec), FC with B vectors, conv split
into im2col and GEMM. {RUNS} runs of `bench_mem.py`, about 10 minutes in total.

**Before running**
1. Settings → Accelerator → **GPU T4 x2**.
2. Run All. Predictions were written before any run (`PREDICTIONS.md` in the repository).
3. Download `results_bandwidth.zip` from the Output panel, then stop the session.

The code cells below are generated from the project's `code/` folder by `make_bandwidth_notebook.py`.
"""),
        md("## 1 · Machine check"),
        code(f"""
!nvidia-smi
!lscpu | grep -E 'Model name|^CPU\\(s\\)|Thread|Core|Socket|L2|L3'
import torch
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| GPU available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "No GPU: Settings -> Accelerator -> GPU T4 x2, then restart the session."
print("torch CPU threads (max):", torch.get_num_threads())
!mkdir -p {WORK}/code
"""),
        md("## 2 · Benchmark code (written to `/kaggle/working/code`)"),
    ]
    for name in FILES:
        cells.append(code(f"%%writefile {WORK}/code/{name}\n{(HERE / name).read_text(encoding='utf-8')}"))
    cells += [
        md("## 3 · Helper: run a command, stream its output, stop the notebook if it fails"),
        code(f"""
import os, shutil, subprocess, sys
from pathlib import Path
WORK = Path("{WORK}")
CODE, RES = WORK / "code", WORK / "results" / "kaggle"
ENV = dict(os.environ, MALLOC_MMAP_MAX_="0", MALLOC_TRIM_THRESHOLD_="68719476736", PYTHONUNBUFFERED="1")

def py(*args):
    cmd = [sys.executable, *map(str, args)]
    p = subprocess.Popen(cmd, cwd=CODE, env=ENV, text=True, encoding="utf-8",
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in p.stdout:
        print(line, end="")
    if p.wait():
        raise RuntimeError(f"exit code {{p.returncode}}: {{' '.join(cmd)}}")
"""),
        md("## 4 · Smoke test (a few seconds, kept out of the results)"),
        code("""
shutil.rmtree(WORK / "smoke", ignore_errors=True)
py("bench_mem.py", "--machine", "kaggle", "--quick", "--tag", "smoke", "--outdir", WORK / "smoke",
   "--devices", "cuda,cpu", "--warm-seconds", 1)
"""),
        md(f"## 5 · {RUNS} runs, GPU first, {REPEAT} repeats per point, shuffled sizes"),
        code("""
from datetime import datetime
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RES.mkdir(parents=True, exist_ok=True)
!lscpu > {RES}/lscpu_{STAMP}.txt
"""),
    ]
    for i in range(1, RUNS + 1):
        cells.append(code(f"""
RUN{i} = RES / f"bw{i}_{{STAMP}}"
py("bench_mem.py", "--machine", "kaggle", "--tag", RUN{i}.name, "--outdir", RES.parent,
   "--devices", "cuda,cpu", "--repeat", {REPEAT}, "--order-seed", {i})
"""))
    runs = ", ".join(f"RUN{i}" for i in range(1, RUNS + 1))
    cells += [
        md("## 6 · Quick look (means over the runs; the full analysis is done offline)"),
        code(f"""
import pandas as pd
bw = pd.concat([pd.read_csv(r / "bw.csv").assign(run=r.name) for r in ({runs})])
print(bw.pivot_table(index=["device", "threads", "impl"], columns="size", values="gbps", aggfunc="mean").round(1).to_string())
fcb = pd.concat([pd.read_csv(r / "fcb.csv").assign(run=r.name) for r in ({runs})])
print(fcb.pivot_table(index=["device", "threads", "impl"], columns="size", values="gflops", aggfunc="mean").round(1).to_string())
step = pd.concat([pd.read_csv(r / "step.csv").assign(run=r.name) for r in ({runs})])
step["us_per_img"] = step["mean_ms"] / step["size"] * 1e3
print(step.pivot_table(index=["device", "impl"], columns="size", values="us_per_img", aggfunc="mean").round(1).to_string())
"""),
        md("## 7 · Package the results for download"),
        code("""
print(shutil.make_archive(str(WORK / "results_bandwidth"), "zip", WORK, "results"))
"""),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": False,
                       "isGpuEnabled": True, "language": "python", "sourceType": "notebook"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
