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
FILES = ["common.py", "bench_conv.py", "bench_fc.py", "run_all.py", "plot.py"]
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
2. Run All (or Save Version → Save & Run All). The full run takes roughly 5–10 minutes.
3. Download `results_kaggle.zip` from the Output panel and unzip it into `results\\kaggle\\` on the laptop.

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
        md("## 3 · Smoke test (a few seconds, kept out of the results)"),
        code(f"!rm -rf {WORK}/smoke && cd {WORK}/code && "
             f"python run_all.py --machine kaggle --quick --tag smoke --outdir {WORK}/smoke"),
        md("## 4 · Full run: CPU 1 thread, CPU max threads, GPU"),
        code(f"""
from datetime import datetime
TAG = datetime.now().strftime("run_kaggle_%Y%m%d_%H%M%S")
RUN = f"{WORK}/results/kaggle/{{TAG}}"
print(RUN)
!cd {WORK}/code && python run_all.py --machine kaggle --tag {{TAG}}
"""),
        md("## 5 · Figures and summary"),
        code(f"""
!cd {WORK}/code && python plot.py {{RUN}}
from pathlib import Path
from IPython.display import Image, Markdown, display
fig_dir = Path(RUN) / "figures"
display(Markdown((fig_dir / "summary.md").read_text(encoding="utf-8")))
for png in sorted(fig_dir.glob("*.png")):
    display(Image(filename=str(png), width=900))
"""),
        md("## 6 · Package the results for download"),
        code(f"!cd {WORK} && rm -f results_kaggle.zip && zip -rq results_kaggle.zip results && ls -lh results_kaggle.zip"),
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
