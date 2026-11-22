"""Experiment 1 (paper Fig 1-2): forward time of the paper's first conv layer vs batch size.

Layer from the paper: greyscale 224x224 input, 96 filters of 11x11, stride 4.
Output is 96 x 54 x 54, so each image unrolls (im2col) into a 121 x 2916 matrix.
(The paper says "54 parts"; it is 54 per side, i.e. 54*54 = 2916 patches.)

Implementations:
  caffe   - what the paper describes for Caffe: loop over the images of the batch one
            at a time; per image im2col (F.unfold) then one GEMM W(96x121) @ cols(121x2916)
            plus bias. This is the main series, comparable to the paper.
  batched - im2col for the whole batch, one batched GEMM (extension).
  conv2d  - the library convolution (oneDNN on CPU, cuDNN on GPU) (extension).

caffe and batched write into preallocated buffers (Caffe keeps its column buffer and output
blob too). F.conv2d has no out= argument, so on CPU it still allocates its output on every
call and the first-touch page faults of that buffer are part of its time. On Linux, run with
MALLOC_MMAP_MAX_=0 and a large MALLOC_TRIM_THRESHOLD_ so malloc reuses memory instead.
"""
import argparse

import torch
import torch.nn.functional as F

from common import (DEFAULT_THREADS, append_rows, bench, get_device, gpu_warmup, make_row,
                    ordered, parse_ints, resolve_threads)

IMG = 224
C_IN = 1
C_OUT = 96
K = 11
STRIDE = 4
OUT_HW = (IMG - K) // STRIDE + 1           # 54
PATCHES = OUT_HW * OUT_HW                   # 2916
PATCH_LEN = C_IN * K * K                    # 121
FLOPS_PER_IMAGE = 2 * C_OUT * PATCH_LEN * PATCHES   # multiply-adds of the GEMM

IMPLS = ("caffe", "batched", "conv2d")
DEFAULT_BATCHES = "1,8,16,32,64,96,128,160,192,224,256"


def make_layer(device):
    gen = torch.Generator().manual_seed(0)
    weight = torch.randn(C_OUT, C_IN, K, K, generator=gen).to(device)
    bias = torch.randn(C_OUT, generator=gen).to(device)
    return weight, bias


def im2col(x, out):
    return torch._C._nn.im2col(x, (K, K), (1, 1), (0, 0), (STRIDE, STRIDE), out=out)


def make_fn(impl, x, weight, bias):
    """Return a zero-argument callable that runs one forward pass of the layer."""
    w2 = weight.view(C_OUT, PATCH_LEN)        # unrolled kernels, 96 x 121
    b2 = bias.view(C_OUT, 1)
    n = x.shape[0]

    if impl == "caffe":
        out = torch.empty(n, C_OUT, PATCHES, device=x.device)       # Caffe's top blob
        cols = torch.empty(1, PATCH_LEN, PATCHES, device=x.device)  # Caffe's column buffer

        def fn():
            for i in range(n):
                im2col(x[i:i + 1], cols)                            # 121 x 2916
                torch.addmm(b2, w2, cols[0], out=out[i])            # 96 x 2916
            return out
    elif impl == "batched":
        out = torch.empty(n, C_OUT, PATCHES, device=x.device)
        cols = torch.empty(n, PATCH_LEN, PATCHES, device=x.device)

        def fn():
            im2col(x, cols)                                         # n x 121 x 2916
            return torch.matmul(w2, cols, out=out).add_(b2)         # n x 96 x 2916
    elif impl == "conv2d":
        def fn():
            return F.conv2d(x, weight, bias, stride=STRIDE)
    else:
        raise ValueError(f"unknown impl {impl}")
    return fn


def check_correctness(device):
    """All three implementations must give the same output before anything is timed."""
    weight, bias = make_layer(device)
    x = torch.randn(3, C_IN, IMG, IMG, generator=torch.Generator().manual_seed(1)).to(device)
    with torch.inference_mode():
        ref = make_fn("conv2d", x, weight, bias)().reshape(3, C_OUT, PATCHES)
        for impl in ("caffe", "batched"):
            got = make_fn(impl, x, weight, bias)().reshape(3, C_OUT, PATCHES)
            torch.testing.assert_close(got, ref, rtol=1e-3, atol=1e-2,
                                       msg=lambda m, impl=impl: f"{impl} != conv2d on {device}: {m}")


def run(device_name, threads_list, impls, batches, warmup, repeat, machine, out_csv, log=print,
        order_seed=0, warm_seconds=0):
    device = get_device(device_name)
    if device.type == "cuda":
        threads_list = [DEFAULT_THREADS]      # host threads do not matter for GPU runs
    check_correctness(device)
    weight, bias = make_layer(device)
    rows = []
    for threads in threads_list:
        torch.set_num_threads(threads)
        for impl in impls:
            gpu_warmup(device, warm_seconds)
            for n in ordered(batches, order_seed):
                x = torch.randn(n, C_IN, IMG, IMG, device=device)
                with torch.inference_mode():
                    fn = make_fn(impl, x, weight, bias)
                    stats = bench(fn, device, warmup, repeat)
                flops = FLOPS_PER_IMAGE * n
                nbytes = 4 * n * (IMG * IMG + C_OUT * PATCHES)   # read input + write output
                row = make_row("conv", impl, device, threads, n, stats, warmup, repeat,
                               flops, nbytes, machine)
                rows.append(row)
                log(f"conv  {device.type:4s} t={threads:<2d} {impl:7s} batch={n:<4d} "
                    f"{stats['mean_ms']:10.3f} ms ± {stats['std_ms']:.3f}  ({row['gflops']} GFLOP/s)"
                    + (f"  issue {stats['issue_ms']:.3f} ms" if device.type == "cuda" else ""))
                del x, fn
        append_rows(out_csv, [r for r in rows if r["threads"] == threads])
    torch.set_num_threads(DEFAULT_THREADS)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--threads", default="1,max", help="CPU thread counts, e.g. 1,4,max")
    parser.add_argument("--impls", default=",".join(IMPLS))
    parser.add_argument("--batches", default=DEFAULT_BATCHES)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--machine", default="local")
    parser.add_argument("--out", default="conv.csv")
    args = parser.parse_args()
    run(args.device, resolve_threads(args.threads), args.impls.split(","),
        parse_ints(args.batches), args.warmup, args.repeat, args.machine, args.out)


if __name__ == "__main__":
    main()
