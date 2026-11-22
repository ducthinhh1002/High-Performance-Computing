"""Build a self-contained Kaggle notebook from the benchmark code in this folder.

    python make_kaggle_notebook.py        # -> ../kaggle/hpc_replication_kaggle.ipynb

The notebook writes the .py files with %%writefile, so nothing has to be uploaded as a
Kaggle Dataset: import the notebook, pick the GPU accelerator, Run All. Re-run this script
after any change to the code so the notebook stays in sync.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "kaggle" / "hpc_replication_kaggle.ipynb"
FILES = ["common.py", "gpu_sampler.py", "bench_conv.py", "bench_fc.py", "run_all.py",
         "aggregate.py", "gpu_diag.py", "plot.py"]
RUNS, REPEAT = 3, 30
WORK = "/kaggle/working"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def build():
    cells = [
        md("""
# Replicating section 3 of *Parallel and Distributed Deep Learning* (Hegde & Usmani) on Kaggle

CPU vs GPU forward time of the paper's conv layer (Fig 1–2) and of the FC layer (Fig 3–4),
measured **on the same machine**, like the paper did on one AWS instance.

**Before running**
1. Settings → Accelerator → **GPU T4 x2** (recent PyTorch builds may no longer support the P100).
2. Run All (or Save Version → Save & Run All). Three full runs, roughly 35 minutes in total.
3. Download `results_kaggle.zip` from the Output panel, then stop the session (power button).

Any failing step raises an error, so Run All stops there instead of carrying on.

The code cells below are generated from the project's `code/` folder by `make_kaggle_notebook.py`.
"""),
        md("## 1 · Machine check"),
        code(f"""
!nvidia-smi
!lscpu | grep -E 'Model name|^CPU\\(s\\)|Thread|Core|Socket'
import torch
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| GPU available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "No GPU: Settings -> Accelerator -> GPU T4 x2, then restart the session."
print("GPU 0:", torch.cuda.get_device_name(0), "| compute capability", torch.cuda.get_device_capability(0))
print("torch CPU threads (max):", torch.get_num_threads())
!mkdir -p {WORK}/code
"""),
        md("## 2 · Benchmark code (written to `/kaggle/working/code`)"),
    ]
    for name in FILES:
        body = (HERE / name).read_text(encoding="utf-8")
        cells.append(code(f"%%writefile {WORK}/code/{name}\n{body}"))
    cells += [
        md("## 3 · Helper: run a command, stream its output, stop the notebook if it fails"),
        code(f"""
import os, shutil, subprocess, sys
from pathlib import Path
WORK = Path("{WORK}")
CODE, RES = WORK / "code", WORK / "results" / "kaggle"
# glibc: serve big buffers from the heap and keep them, so F.conv2d on the CPU does not pay
# page faults for a fresh output buffer on every call (recorded in env.json).
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
py("run_all.py", "--machine", "kaggle", "--quick", "--tag", "smoke", "--outdir", WORK / "smoke",
   "--devices", "cuda,cpu", "--warm-seconds", 1, "--order-seed", 1)
"""),
        md("""
## 5 · Diagnostic: cold GPU, ascending batch sizes (the conditions of the first Kaggle run)

Reproduces the jump seen around batch 96-128, now with the SM clock and the host issue time.
"""),
        code("""
import time
from datetime import datetime
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
time.sleep(30)                                  # let the GPU fall back to its idle clocks
DIAG = RES / f"diag_cold_{STAMP}"
py("run_all.py", "--machine", "kaggle", "--tag", DIAG.name, "--outdir", RES.parent,
   "--devices", "cuda", "--skip", "fc", "--warm-seconds", 0, "--order-seed", 0)
"""),
        md(f"## 6 · {RUNS} full runs, GPU first (warm GPU, shuffled sizes, {REPEAT} repeats per point)"),
    ]
    for i in range(1, RUNS + 1):
        cells.append(code(f"""
RUN{i} = RES / f"run{i}_{{STAMP}}"
py("run_all.py", "--machine", "kaggle", "--tag", RUN{i}.name, "--outdir", RES.parent,
   "--devices", "cuda,cpu", "--repeat", {REPEAT}, "--order-seed", {i})
"""))
    runs = ", ".join(f"RUN{i}" for i in range(1, RUNS + 1))
    cells += [
        md("## 7 · Combine the runs, figures, summary and GPU diagnosis"),
        code(f"""
AGG = RES / f"agg_{{STAMP}}"
py("aggregate.py", {runs}, "--out", AGG)
py("plot.py", AGG)
py("gpu_diag.py", DIAG, {runs}, "--out", AGG / "gpu_diag.md")
from IPython.display import Image, Markdown, display
display(Markdown((AGG / "figures" / "summary.md").read_text(encoding="utf-8")))
for png in sorted((AGG / "figures").glob("*.png")):
    display(Image(filename=str(png), width=900))
"""),
        md("## 8 · Package the results for download"),
        code("""
print(shutil.make_archive(str(WORK / "results_kaggle"), "zip", WORK, "results"))
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
