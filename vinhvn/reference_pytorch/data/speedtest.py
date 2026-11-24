"""Đo tốc độ tải từ trong cụm: Hugging Face so với nguồn gốc CIFAR-10 (Toronto).
Mỗi nguồn tải tối đa 40 MB hoặc 60 giây, không ghi ra đĩa."""
import json
import time
import urllib.request

def probe(name, url, max_bytes=40_000_000, max_sec=60):
    t0 = time.time(); got = 0; ttfb = None
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "speedtest/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            final = r.geturl()
            while got < max_bytes and time.time() - t0 < max_sec:
                c = r.read(1 << 20)
                if ttfb is None:
                    ttfb = time.time() - t0
                if not c:
                    break
                got += len(c)
        dt = time.time() - t0
        print(f"{name:12s} {got/1e6:6.1f} MB in {dt:5.1f}s = {got/dt/1e6:7.2f} MB/s  (ttfb {ttfb:.2f}s, host {final.split('/')[2]})", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"{name:12s} ERROR {e!r}", flush=True)

# Liệt kê file của dataset trên HF để lấy đúng đường dẫn.
api = "https://huggingface.co/api/datasets/uoft-cs/cifar10/tree/main/plain_text"
files = json.load(urllib.request.urlopen(api, timeout=30))
for f in files:
    print("HF file:", f["path"], f.get("size"))
train = next(f["path"] for f in files if "train" in f["path"])

probe("huggingface", f"https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/{train}")
probe("toronto", "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz")
