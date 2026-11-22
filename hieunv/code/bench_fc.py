"""Experiment 2 (paper Fig 3-4): fully-connected layer time vs matrix size.

The paper implements the FC layer as a matrix-vector product and reports O(n^2) time.
Its Fig 3 caption says "matrix-matrix multiplication", so both are measured:
  matvec - y = A x, A is n x n (main series, comparable to the paper), O(n^2)
  gemm   - C = A B, all n x n (resolves the caption ambiguity), O(n^3)
"""
import argparse

import torch

from common import (DEFAULT_THREADS, append_rows, bench, get_device, gpu_warmup, make_row,
                    ordered, parse_ints, resolve_threads)

OPS = ("matvec", "gemm")
DEFAULT_SIZES = {
    "matvec": "256,512,1024,2048,4096,8192,12288,16384",
    "gemm": "256,512,1024,1536,2048,3072,4096",
}


def make_fn(op, n, device):
    gen = torch.Generator().manual_seed(n)
    a = torch.randn(n, n, generator=gen).to(device)
    if op == "matvec":
        x = torch.randn(n, generator=gen).to(device)
        y = torch.empty(n, device=device)
        return (lambda: torch.mv(a, x, out=y)), 2 * n * n, 4 * (n * n + 2 * n)
    if op == "gemm":
        b = torch.randn(n, n, generator=gen).to(device)
        c = torch.empty(n, n, device=device)
        return (lambda: torch.mm(a, b, out=c)), 2 * n ** 3, 4 * 3 * n * n
    raise ValueError(f"unknown op {op}")


def check_correctness(device):
    gen = torch.Generator().manual_seed(7)
    a = torch.randn(300, 300, generator=gen)
    x = torch.randn(300, generator=gen)
    b = torch.randn(300, 300, generator=gen)
    with torch.inference_mode():
        torch.testing.assert_close(torch.mv(a.to(device), x.to(device)).cpu(),
                                   (a.double() @ x.double()).float(), rtol=1e-4, atol=1e-3)
        torch.testing.assert_close(torch.mm(a.to(device), b.to(device)).cpu(),
                                   (a.double() @ b.double()).float(), rtol=1e-4, atol=1e-3)


def run(device_name, threads_list, ops, sizes, warmup, repeat, machine, out_csv, log=print,
        order_seed=0, warm_seconds=0):
    """sizes: dict op -> list of n."""
    device = get_device(device_name)
    if device.type == "cuda":
        threads_list = [DEFAULT_THREADS]
    check_correctness(device)
    rows = []
    for threads in threads_list:
        torch.set_num_threads(threads)
        for op in ops:
            gpu_warmup(device, warm_seconds)
            for n in ordered(sizes[op], order_seed):
                fn, flops, nbytes = make_fn(op, n, device)
                with torch.inference_mode():
                    stats = bench(fn, device, warmup, repeat)
                row = make_row("fc", op, device, threads, n, stats, warmup, repeat,
                               flops, nbytes, machine)
                rows.append(row)
                log(f"fc    {device.type:4s} t={threads:<2d} {op:7s} n={n:<6d} "
                    f"{stats['mean_ms']:10.3f} ms ± {stats['std_ms']:.3f}  "
                    f"({row['gflops']} GFLOP/s, {row['gbps']} GB/s)")
                del fn
        append_rows(out_csv, [r for r in rows if r["threads"] == threads])
    torch.set_num_threads(DEFAULT_THREADS)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--threads", default="1,max")
    parser.add_argument("--ops", default=",".join(OPS))
    parser.add_argument("--matvec-sizes", default=DEFAULT_SIZES["matvec"])
    parser.add_argument("--gemm-sizes", default=DEFAULT_SIZES["gemm"])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--machine", default="local")
    parser.add_argument("--out", default="fc.csv")
    args = parser.parse_args()
    sizes = {"matvec": parse_ints(args.matvec_sizes), "gemm": parse_ints(args.gemm_sizes)}
    run(args.device, resolve_threads(args.threads), args.ops.split(","), sizes,
        args.warmup, args.repeat, args.machine, args.out)


if __name__ == "__main__":
    main()
