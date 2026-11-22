"""Run the whole benchmark grid on one machine and save it as one run.

    python run_all.py --machine laptop            # full grid, every available device
    python run_all.py --machine kaggle            # same on Kaggle (CPU + GPU)
    python run_all.py --machine laptop --quick    # smoke test, a few seconds

Output: <outdir>/<machine>/<tag>/{env.json, conv.csv, fc.csv, log.txt}
outdir defaults to ../results, or /kaggle/working/results on Kaggle (input is read-only).
Then plot with:  python plot.py <that run folder>
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import bench_conv
import bench_fc
from common import ROOT, available_devices, env_info, parse_ints, resolve_threads, write_json

QUICK = {
    "batches": "1,4,8",
    "matvec": "256,1024",
    "gemm": "256,512",
    "warmup": 1,
    "repeat": 3,
}


def default_outdir():
    kaggle = Path("/kaggle/working")
    return kaggle / "results" if kaggle.is_dir() else ROOT / "results"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--machine", required=True, help="label, e.g. laptop or kaggle")
    parser.add_argument("--tag", default=datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    parser.add_argument("--outdir", type=Path, default=default_outdir())
    parser.add_argument("--devices", default=",".join(available_devices()))
    parser.add_argument("--threads", default="1,max", help="CPU thread counts")
    parser.add_argument("--batches", default=bench_conv.DEFAULT_BATCHES)
    parser.add_argument("--matvec-sizes", default=bench_fc.DEFAULT_SIZES["matvec"])
    parser.add_argument("--gemm-sizes", default=bench_fc.DEFAULT_SIZES["gemm"])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--skip", default="", help="comma list: conv,fc")
    parser.add_argument("--quick", action="store_true", help="tiny grid for a smoke test")
    args = parser.parse_args()

    if args.quick:
        args.batches, args.matvec_sizes, args.gemm_sizes = QUICK["batches"], QUICK["matvec"], QUICK["gemm"]
        args.warmup, args.repeat = QUICK["warmup"], QUICK["repeat"]

    run_dir = args.outdir / args.machine / args.tag
    run_dir.mkdir(parents=True, exist_ok=False)
    log_file = (run_dir / "log.txt").open("w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    env = env_info(args.machine)
    env["grid"] = {k: v for k, v in vars(args).items() if k not in ("outdir",)}
    write_json(run_dir / "env.json", env)
    log(f"run folder: {run_dir}")
    log(f"cpu: {env['cpu']} | torch threads (max): {env['torch_default_threads']}"
        + (f" | gpu: {env['gpu']}" if env["cuda_available"] else " | no CUDA GPU"))

    threads = resolve_threads(args.threads)
    skip = set(filter(None, args.skip.split(",")))
    start = time.perf_counter()
    for device in args.devices.split(","):
        if "conv" not in skip:
            bench_conv.run(device, threads, list(bench_conv.IMPLS), parse_ints(args.batches),
                           args.warmup, args.repeat, args.machine, run_dir / "conv.csv", log)
        if "fc" not in skip:
            sizes = {"matvec": parse_ints(args.matvec_sizes), "gemm": parse_ints(args.gemm_sizes)}
            bench_fc.run(device, threads, list(bench_fc.OPS), sizes,
                         args.warmup, args.repeat, args.machine, run_dir / "fc.csv", log)
    log(f"done in {time.perf_counter() - start:.1f} s -> {run_dir}")
    log_file.close()


if __name__ == "__main__":
    main()
