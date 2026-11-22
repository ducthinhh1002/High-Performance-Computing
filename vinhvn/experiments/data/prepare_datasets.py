"""Chuẩn bị dataset vào hostPath dùng chung, chỉ cần CPU.

Nguồn: Hugging Face (nhanh hơn nguồn gốc khoảng 200 lần khi đo từ cụm, 20/11/2026).
- Ghim theo commit của repo HF để tái lập được; ghi commit vào meta.json.
- Kiểm tra sha256 từng file theo thông tin LFS do HF công bố.
- Chuyển Parquet (ảnh PNG + nhãn) sang npz: x uint8 (N, 32, 32, 3), y int64 (N,).
  Đọc bằng experiments/common/data.py (CIFAR10Npz).
- Từ chối ghi nếu thư mục gốc đã vượt giới hạn (mặc định 50 GB), vì hostPath
  không có giới hạn dung lượng và đây là node dùng chung.
"""
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import urllib.request

import numpy as np

ROOT = os.environ.get("HOSTPATH_ROOT", "/hostdata")
DATA_DIR = os.path.join(ROOT, "datasets")
LIMIT_GB = float(os.environ.get("LIMIT_GB", "50"))
HF = "https://huggingface.co"

DATASETS = [
    {
        "name": "cifar10",
        "repo": "uoft-cs/cifar10",
        # Ghim commit đã dùng lần đầu (20/11/2026) để mọi lần chuẩn bị lại cho đúng cùng dữ liệu.
        "revision": os.environ.get("CIFAR10_REVISION", "0b2714987fa478483af9968de7c934580d0bb9a2"),
        "splits": {"train": "plain_text/train-00000-of-00001.parquet",
                   "test": "plain_text/test-00000-of-00001.parquet"},
        "expect": {"train": 50000, "test": 10000},
        "classes": ["airplane", "automobile", "bird", "cat", "deer",
                    "dog", "frog", "horse", "ship", "truck"],
    },
]


def du_gb(path):
    total = 0
    for dp, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return total / 1e9


def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "hpc-dl-exp/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def fetch(url, expect_sha256, expect_size):
    for attempt in range(1, 6):
        try:
            t0 = time.time()
            req = urllib.request.Request(url, headers={"User-Agent": "hpc-dl-exp/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            sha = hashlib.sha256(data).hexdigest()
            dt = time.time() - t0
            print(f"  {len(data)/1e6:.1f} MB in {dt:.1f}s ({len(data)/dt/1e6:.1f} MB/s), sha256 {sha[:12]}…", flush=True)
            if len(data) != expect_size or sha != expect_sha256:
                raise ValueError(f"integrity mismatch: size {len(data)} vs {expect_size}, sha {sha} vs {expect_sha256}")
            return data
        except Exception as e:  # noqa: BLE001
            print(f"  attempt {attempt} failed: {e!r}", flush=True)
            time.sleep(5 * attempt)
    raise RuntimeError(f"download failed: {url}")


def ensure_pyarrow():
    try:
        import pyarrow.parquet  # noqa: F401
    except ImportError:
        print("installing pyarrow…", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "pyarrow"])


def parquet_to_arrays(raw):
    import pyarrow.parquet as pq
    from PIL import Image

    table = pq.read_table(io.BytesIO(raw))
    cols = table.column_names
    img_col = next(c for c in cols if c in ("img", "image"))
    lab_col = next(c for c in cols if c in ("label", "labels"))
    imgs = table.column(img_col).to_pylist()
    labels = np.asarray(table.column(lab_col).to_pylist(), dtype=np.int64)
    x = np.stack([np.asarray(Image.open(io.BytesIO(im["bytes"])).convert("RGB"), dtype=np.uint8) for im in imgs])
    return x, labels


def prepare(d):
    out_dir = os.path.join(DATA_DIR, d["name"])
    if os.path.exists(os.path.join(out_dir, ".complete")):
        print(f"[SKIP] {d['name']} already prepared at {out_dir}")
        return
    os.makedirs(out_dir, exist_ok=True)

    rev = d["revision"] or get_json(f"{HF}/api/datasets/{d['repo']}")["sha"]
    print(f"[{d['name']}] repo {d['repo']} @ {rev}", flush=True)
    tree = {f["path"]: f for f in get_json(f"{HF}/api/datasets/{d['repo']}/tree/{rev}/plain_text")}

    meta = {"source": f"{HF}/datasets/{d['repo']}", "revision": rev, "classes": d["classes"], "splits": {}}
    for split, path in d["splits"].items():
        info = tree[path]
        print(f"[DOWNLOAD] {split}: {path}", flush=True)
        raw = fetch(f"{HF}/datasets/{d['repo']}/resolve/{rev}/{path}", info["lfs"]["oid"], info["lfs"]["size"])
        x, y = parquet_to_arrays(raw)
        n = d["expect"][split]
        counts = np.bincount(y, minlength=len(d["classes"])).tolist()
        assert x.shape == (n, 32, 32, 3), x.shape
        assert len(set(counts)) == 1, f"class counts not balanced: {counts}"
        np.savez(os.path.join(out_dir, f"{split}.npz"), x=x, y=y)
        meta["splits"][split] = {"n": int(n), "sha256_parquet": info["lfs"]["oid"],
                                 "class_counts": counts, "pixel_mean": float(x.mean())}
        print(f"[OK] {split}: x{tuple(x.shape)} uint8, per-class {counts[0]}", flush=True)

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    open(os.path.join(out_dir, ".complete"), "w").close()


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    # Dọn file tải dở từ lần thử nguồn Toronto.
    for stale in ("cifar-10-python.tar.gz.part", "cifar-10-python.tar.gz"):
        p = os.path.join(DATA_DIR, stale)
        if os.path.exists(p):
            os.remove(p)
            print(f"removed stale {p}")
    used = du_gb(ROOT)
    print(f"hostPath root {ROOT}: {used:.2f} GB used, limit {LIMIT_GB} GB", flush=True)
    if used > LIMIT_GB:
        sys.exit(f"ABORT: {ROOT} already exceeds {LIMIT_GB} GB")

    ensure_pyarrow()
    for d in DATASETS:
        prepare(d)

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "README.txt"), "w") as f:
        f.write("Owner: namespace hpc-dl-exp (research Parallel & Distributed DL).\n"
                "Temporary hostPath because VAST PVC provisioning failed on 2026-11-20.\n"
                "Size cap 50 GB (self-enforced). Delete this directory when the experiments end.\n")
    print(f"DONE. {du_gb(ROOT):.2f} GB used under {ROOT}", flush=True)
    os.system(f"ls -laR {DATA_DIR}")


if __name__ == "__main__":
    main()
