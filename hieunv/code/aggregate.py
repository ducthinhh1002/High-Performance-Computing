"""Combine several runs of the same grid into one folder that plot.py understands.

    python aggregate.py <run1> <run2> <run3> --out ../results/kaggle/agg

mean_ms is the mean of the runs' means, run_std_ms the sample std BETWEEN runs. The runs
must come from the same machine and software; with runs from one session this shows
run-to-run noise on one machine, not differences between machines.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

KEY = ["exp", "impl", "device", "threads", "size"]
SAME = ("cpu", "gpu", "torch", "cuda", "python", "torch_default_threads")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    envs = [json.loads((r / "env.json").read_text(encoding="utf-8")) for r in args.runs]
    for field in SAME:
        values = {str(e.get(field)) for e in envs}
        if len(values) > 1:
            raise SystemExit(f"runs differ in {field}: {sorted(values)} - do not average them")

    args.out.mkdir(parents=True, exist_ok=True)
    for name in ("conv", "fc"):
        if not all((r / f"{name}.csv").exists() for r in args.runs):
            continue
        raw = pd.concat([pd.read_csv(r / f"{name}.csv").assign(run=r.name) for r in args.runs],
                        ignore_index=True)
        counts = raw.groupby(KEY)["run"].nunique()
        if (counts != len(args.runs)).any():
            raise SystemExit(f"{name}: some points are missing in some runs:\n{counts[counts != len(args.runs)]}")
        raw.to_csv(args.out / f"runs_{name}.csv", index=False)
        agg = raw.groupby(KEY, as_index=False).agg(
            mean_ms=("mean_ms", "mean"), run_std_ms=("mean_ms", "std"), std_ms=("std_ms", "mean"),
            issue_ms=("issue_ms", "mean"), gflops=("gflops", "mean"), gbps=("gbps", "mean"),
            repeats=("repeats", "max"), runs=("run", "nunique"), machine=("machine", "first"))
        agg.to_csv(args.out / f"{name}.csv", index=False)
    env = dict(envs[0], aggregated_runs=[r.name for r in args.runs])
    (args.out / "env.json").write_text(json.dumps(env, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
