"""Several full CPU runs on this machine, then one combined folder with figures.

    python run_session.py --machine laptop                  # 3 runs, 30 repeats, ~22 min
    python run_session.py --machine laptop --quick --outdir <tmp>   # smoke test

Each run is a separate process with its own shuffled measurement order. On Windows the CPU
clock (% of base frequency, per core) and the total CPU load are logged once per second
(typeperf), so throttling and background load can be seen afterwards: cpu_perf.csv in each run.
"""
import argparse
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
COUNTERS = [r"\Processor Information(*)\% Processor Performance",
            r"\Processor Information(_Total)\% Processor Time"]


def py(*args):
    subprocess.run([sys.executable, *map(str, args)], cwd=HERE, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--machine", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--threads", default="1,max")
    parser.add_argument("--cooldown", type=float, default=60, help="seconds before each run")
    parser.add_argument("--outdir", type=Path, default=HERE.parent / "results")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = args.outdir / args.machine
    base.mkdir(parents=True, exist_ok=True)
    runs = []
    for i in range(1, args.runs + 1):
        print(f"--- run {i}/{args.runs}: cooling down {args.cooldown:.0f} s", flush=True)
        time.sleep(args.cooldown)
        run_dir, log, sampler = base / f"run{i}_{stamp}", base / f"cpu_perf_{i}_{stamp}.csv", None
        if shutil.which("typeperf"):
            sampler = subprocess.Popen(["typeperf", *COUNTERS, "-si", "1", "-f", "CSV", "-o", str(log), "-y"],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                       stdin=subprocess.DEVNULL)
        try:
            py("run_all.py", "--machine", args.machine, "--tag", run_dir.name, "--outdir", args.outdir,
               "--devices", "cpu", "--threads", args.threads, "--repeat", args.repeat,
               "--order-seed", i, *(["--quick"] if args.quick else []))
        finally:
            if sampler is not None:
                sampler.terminate()
                sampler.wait(timeout=10)
        if log.exists():
            shutil.move(str(log), str(run_dir / "cpu_perf.csv"))
        runs.append(run_dir)
    agg = base / f"agg_{stamp}"
    py("aggregate.py", *runs, "--out", agg)
    py("plot.py", agg)
    print("session done ->", agg, flush=True)


if __name__ == "__main__":
    main()
