"""Independent checks of the analysis (ANALYSIS.md section 7, PREDICTIONS.md), separate from the paper's runs.

    python bench_mem.py --machine laptop --quick                         # smoke test
    python bench_mem.py --machine kaggle --devices cuda,cpu --repeat 20  # one run

Three experiments; one CSV each in <outdir>/<machine>/<tag>, plus env.json, log.txt and gpu_clocks.csv:
  bw.csv    memory bandwidth from STREAM-like kernels that do not use matvec code:
              sum   read   4 B/element      dot   read 8 B/element
              copy  r+w    8 B/element      triad read 2 + write 1 = 12 B/element (STREAM counting)
            plus matvec itself. Arrays have N = n*n elements, matching the matvec sizes of fc.csv.
  fcb.csv   FC layer with B input vectors, (n x n) @ (n x B); impl mm1 (= mv) ... mm128.
  step.csv  Caffe-style conv timed as im2col only / GEMM only / both (the caffe impl), per batch.
The arrays are filled with random numbers so that no page is shared or compressed.
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import torch

import bench_conv
from common import (DEFAULT_THREADS, ROOT, GpuClockLog, append_rows, available_devices, bench, env_info,
                    get_device, gpu_warmup, make_row, ordered, parse_ints, resolve_threads, write_json)

BW_KERNELS = ("sum", "dot", "copy", "triad", "matvec")
DEFAULT_SIZES = "1024,4096,8192,16384"
DEFAULT_FC_SIZES = "2048,8192"
DEFAULT_B = "1,16,64,128"
DEFAULT_BATCHES = "16,64,256"
QUICK = dict(sizes="256,1024", fc_sizes="256,512", fc_b="1,8", batches="2,4", warmup=1, repeat=2)
TRIAD_S = 3.0


# ---------------------------------------------------------------- bandwidth

def bw_kernel(kernel, n, device):
    """Return (fn, flops, bytes) for one kernel over N = n*n float32 elements per array."""
    N = n * n
    gen = torch.Generator().manual_seed(n)
    x = torch.rand(N, generator=gen).to(device)
    if kernel == "sum":
        return (lambda: torch.sum(x)), N, 4 * N
    if kernel == "dot":
        y = torch.rand(N, generator=gen).to(device)
        return (lambda: torch.dot(x, y)), 2 * N, 8 * N
    if kernel == "copy":
        y = torch.empty(N, device=device)
        return (lambda: y.copy_(x)), 0, 8 * N
    if kernel == "triad":
        b = torch.rand(N, generator=gen).to(device)
        c, out = x, torch.empty(N, device=device)
        return (lambda: torch.add(b, c, alpha=TRIAD_S, out=out)), 2 * N, 12 * N
    if kernel == "matvec":
        a, v = x.view(n, n), torch.rand(n, generator=gen).to(device)
        out = torch.empty(n, device=device)
        return (lambda: torch.mv(a, v, out=out)), 2 * N, 4 * N
    raise ValueError(f"unknown kernel {kernel}")


def check_kernels(device):
    """The timed kernels must compute what they claim (small arrays, against NumPy-style references)."""
    n = 40
    for kernel in BW_KERNELS:
        fn, _, _ = bw_kernel(kernel, n, device)
        got = fn()
        x = torch.rand(n * n, generator=torch.Generator().manual_seed(n))
        if kernel == "sum":
            torch.testing.assert_close(got.cpu(), x.sum(), rtol=1e-4, atol=1e-3)
        elif kernel == "dot":
            # bw_kernel draws x first, then y from the same generator
            gen = torch.Generator().manual_seed(n)
            x2, y2 = torch.rand(n * n, generator=gen), torch.rand(n * n, generator=gen)
            torch.testing.assert_close(got.cpu(), (x2.double() @ y2.double()).float(), rtol=1e-4, atol=1e-3)
        elif kernel == "copy":
            torch.testing.assert_close(got.cpu(), x, rtol=0, atol=0)
        elif kernel == "triad":
            gen = torch.Generator().manual_seed(n)
            x2 = torch.rand(n * n, generator=gen)
            b2 = torch.rand(n * n, generator=gen)
            torch.testing.assert_close(got.cpu(), b2 + TRIAD_S * x2, rtol=1e-5, atol=1e-5)
        elif kernel == "matvec":
            gen = torch.Generator().manual_seed(n)
            x2 = torch.rand(n * n, generator=gen)
            v2 = torch.rand(n, generator=gen)
            torch.testing.assert_close(got.cpu(), (x2.view(n, n).double() @ v2.double()).float(), rtol=1e-4, atol=1e-3)


def run_bw(device_name, threads_list, sizes, warmup, repeat, machine, out_csv, log, order_seed, warm_seconds):
    device = get_device(device_name)
    if device.type == "cuda":
        threads_list = [DEFAULT_THREADS]
    check_kernels(device)
    rows = []
    for threads in threads_list:
        torch.set_num_threads(threads)
        for kernel in BW_KERNELS:
            gpu_warmup(device, warm_seconds)
            for n in ordered(sizes, order_seed):
                fn, flops, nbytes = bw_kernel(kernel, n, device)
                with torch.inference_mode():
                    stats = bench(fn, device, warmup, repeat)
                row = make_row("bw", kernel, device, threads, n, stats, warmup, repeat, flops, nbytes, machine)
                rows.append(row)
                log(f"bw    {device.type:4s} t={threads:<2d} {kernel:7s} n={n:<6d} {stats['mean_ms']:10.3f} ms ± "
                    f"{stats['std_ms']:.3f}  ({row['gbps']} GB/s)")
                del fn
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        append_rows(out_csv, [r for r in rows if r["threads"] == threads])
    torch.set_num_threads(DEFAULT_THREADS)
    return rows


# ---------------------------------------------------------------- FC with B vectors

def fcb_kernel(n, B, device):
    gen = torch.Generator().manual_seed(n)
    a = torch.randn(n, n, generator=gen).to(device)
    if B == 1:
        x, y = torch.randn(n, generator=gen).to(device), torch.empty(n, device=device)
        return (lambda: torch.mv(a, x, out=y)), 2 * n * n, 4 * n * n
    x, y = torch.randn(n, B, generator=gen).to(device), torch.empty(n, B, device=device)
    return (lambda: torch.mm(a, x, out=y)), 2 * n * n * B, 4 * n * n


def check_fcb(device):
    fn, _, _ = fcb_kernel(200, 8, device)
    gen = torch.Generator().manual_seed(200)
    a = torch.randn(200, 200, generator=gen)
    x = torch.randn(200, 8, generator=gen)
    torch.testing.assert_close(fn().cpu(), (a.double() @ x.double()).float(), rtol=1e-4, atol=1e-3)


def run_fcb(device_name, threads_list, sizes, batches, warmup, repeat, machine, out_csv, log, order_seed, warm_seconds):
    device = get_device(device_name)
    if device.type == "cuda":
        threads_list = [DEFAULT_THREADS]
    check_fcb(device)
    rows = []
    for threads in threads_list:
        torch.set_num_threads(threads)
        gpu_warmup(device, warm_seconds)
        for n, B in ordered([(n, B) for n in sizes for B in batches], order_seed):
            fn, flops, nbytes = fcb_kernel(n, B, device)
            with torch.inference_mode():
                stats = bench(fn, device, warmup, repeat)
            row = make_row("fcb", f"mm{B}", device, threads, n, stats, warmup, repeat, flops, nbytes, machine)
            rows.append(row)
            log(f"fcb   {device.type:4s} t={threads:<2d} B={B:<4d} n={n:<6d} {stats['mean_ms']:10.3f} ms ± "
                f"{stats['std_ms']:.3f}  ({row['gflops']} GFLOP/s)")
            del fn
            if device.type == "cuda":
                torch.cuda.empty_cache()
        append_rows(out_csv, [r for r in rows if r["threads"] == threads])
    torch.set_num_threads(DEFAULT_THREADS)
    return rows


# ---------------------------------------------------------------- conv steps

def step_kernel(step, B, device):
    """im2col only, GEMM only, or both: the same per-image loop as bench_conv's 'caffe'."""
    weight, bias = bench_conv.make_layer(device)
    w2, b2 = weight.view(bench_conv.C_OUT, bench_conv.PATCH_LEN), bias.view(bench_conv.C_OUT, 1)
    x = torch.randn(B, bench_conv.C_IN, bench_conv.IMG, bench_conv.IMG, device=device)
    cols = torch.empty(1, bench_conv.PATCH_LEN, bench_conv.PATCHES, device=device)
    out = torch.empty(B, bench_conv.C_OUT, bench_conv.PATCHES, device=device)
    bench_conv.im2col(x[:1], cols)                        # GEMM-only needs valid, finite columns

    if step == "im2col":
        def fn():
            for i in range(B):
                bench_conv.im2col(x[i:i + 1], cols)
            return cols
        flops = 0
    elif step == "gemm":
        def fn():
            for i in range(B):
                torch.addmm(b2, w2, cols[0], out=out[i])
            return out
        flops = bench_conv.FLOPS_PER_IMAGE * B
    elif step == "caffe":
        fn = bench_conv.make_fn("caffe", x, weight, bias)
        flops = bench_conv.FLOPS_PER_IMAGE * B
    else:
        raise ValueError(step)
    return fn, flops


def run_steps(device_name, threads_list, batches, warmup, repeat, machine, out_csv, log, order_seed, warm_seconds):
    device = get_device(device_name)
    if device.type == "cuda":
        threads_list = [DEFAULT_THREADS]
    else:
        threads_list = [1]                                # the baseline of the paper comparison
    bench_conv.check_correctness(device)
    rows = []
    for threads in threads_list:
        torch.set_num_threads(threads)
        for step in ("im2col", "gemm", "caffe"):
            gpu_warmup(device, warm_seconds)
            for B in ordered(batches, order_seed):
                fn, flops = step_kernel(step, B, device)
                with torch.inference_mode():
                    stats = bench(fn, device, warmup, repeat)
                row = make_row("step", step, device, threads, B, stats, warmup, repeat, flops, 0, machine)
                rows.append(row)
                log(f"step  {device.type:4s} t={threads:<2d} {step:7s} batch={B:<4d} {stats['mean_ms']:10.3f} ms ± "
                    f"{stats['std_ms']:.3f}  ({stats['mean_ms'] / B * 1e3:.1f} us/img)")
                del fn
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        append_rows(out_csv, [r for r in rows if r["threads"] == threads])
    torch.set_num_threads(DEFAULT_THREADS)
    return rows


# ---------------------------------------------------------------- main

def default_outdir():
    kaggle = Path("/kaggle/working")
    return kaggle / "results" if kaggle.is_dir() else ROOT / "results"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--machine", required=True)
    p.add_argument("--tag", default=datetime.now().strftime("bw_%Y%m%d_%H%M%S"))
    p.add_argument("--outdir", type=Path, default=default_outdir())
    p.add_argument("--devices", default=",".join(available_devices()))
    p.add_argument("--threads", default="1,max")
    p.add_argument("--sizes", default=DEFAULT_SIZES, help="n for the bandwidth kernels (N = n*n elements)")
    p.add_argument("--fc-sizes", default=DEFAULT_FC_SIZES)
    p.add_argument("--fc-b", default=DEFAULT_B)
    p.add_argument("--batches", default=DEFAULT_BATCHES, help="for the conv steps")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeat", type=int, default=10)
    p.add_argument("--skip", default="", help="comma list: bw,fcb,step")
    p.add_argument("--order-seed", type=int, default=1)
    p.add_argument("--warm-seconds", type=float, default=5)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.sizes, args.fc_sizes, args.fc_b, args.batches = QUICK["sizes"], QUICK["fc_sizes"], QUICK["fc_b"], QUICK["batches"]
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
    sizes, fc_sizes = parse_ints(args.sizes), parse_ints(args.fc_sizes)
    fc_b, batches = parse_ints(args.fc_b), parse_ints(args.batches)
    start = time.perf_counter()
    for device in args.devices.split(","):
        with GpuClockLog(run_dir / "gpu_clocks.csv", device == "cuda", log):
            if "bw" not in skip:
                run_bw(device, threads, sizes, args.warmup, args.repeat, args.machine, run_dir / "bw.csv", log,
                       args.order_seed, args.warm_seconds)
            if "fcb" not in skip:
                run_fcb(device, threads, fc_sizes, fc_b, args.warmup, args.repeat, args.machine, run_dir / "fcb.csv", log,
                        args.order_seed, args.warm_seconds)
            if "step" not in skip:
                run_steps(device, threads, batches, args.warmup, args.repeat, args.machine, run_dir / "step.csv", log,
                          args.order_seed, args.warm_seconds)
    log(f"done in {time.perf_counter() - start:.1f} s -> {run_dir}")
    log_file.close()


if __name__ == "__main__":
    main()
