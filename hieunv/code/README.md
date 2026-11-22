# Benchmark code — replicating section 3 of "Parallel and Distributed Deep Learning"

| File | Purpose |
|---|---|
| `common.py` | timing (`bench`: warm-up runs, then timed runs, GPU synchronized), machine info, CSV |
| `bench_conv.py` | Experiment 1 (paper Fig 1–2): conv layer 96×11×11, stride 4, on 224×224 greyscale, vs batch size. Impls: `caffe` (per-image im2col + GEMM, main), `batched`, `conv2d` |
| `bench_fc.py` | Experiment 2 (paper Fig 3–4): `matvec` (main, O(n²)) and `gemm` (O(n³)) vs n |
| `run_all.py` | runs the full grid on every available device, saves one run folder |
| `plot.py` | Fig 1–6 + `summary.md` / `summary.csv` for one run folder |

Every benchmark first checks that its implementations agree (caffe and batched vs conv2d,
matvec/gemm vs float64), so a wrong kernel is never timed.

## Laptop (CPU only)

```powershell
D:\MASTER\HPC\.venv\Scripts\Activate.ps1
cd D:\MASTER\HPC\code
python run_all.py --machine laptop --quick      # smoke test, a few seconds
python run_all.py --machine laptop              # full grid
python plot.py ..\results\laptop\<run folder>
```

Before the full run: plug in the charger, Windows power mode "Best performance", close other apps.

## Kaggle (CPU + GPU on the same machine)

Use the self-contained notebook `../kaggle/hpc_replication_kaggle.ipynb` (it writes these
.py files itself, nothing else to upload). Regenerate it after changing any code here:
`python make_kaggle_notebook.py`.

1. kaggle.com → Create → New Notebook → File → Import Notebook → pick the .ipynb.
2. Settings → Accelerator: **GPU T4 x2** (recent PyTorch builds may no longer support the P100).
3. Run All. Then download `results_kaggle.zip` (Output panel) into `results\kaggle\`.

Results go to `<outdir>/<machine>/<tag>/` (`env.json`, `conv.csv`, `fc.csv`, `log.txt`);
outdir is `../results` locally and `/kaggle/working/results` on Kaggle.

## Measurement rules

- 3 warm-up + 10 timed runs per point (paper: 10), mean ± std reported.
- Data already on the device; host↔GPU copies are not timed (the paper times forward only).
- float32, TF32 disabled on GPU (the paper's GPU had plain FP32 only).
- CPU baseline is 1 thread (the paper's abstract says "single core CPU"); `max` threads is an extension.

Useful flags: `--threads 1,4,max`, `--batches 1,8,64`, `--skip fc`, `--devices cpu`, `--repeat 20`.
