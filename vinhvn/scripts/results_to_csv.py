#!/usr/bin/env python3
"""Đổi các dòng `RESULT k=v k=v ...` trong một log thành CSV (hợp của mọi khoá). Dùng: results_to_csv.py log > out.csv"""
import csv, sys
rows = []
for line in open(sys.argv[1], errors="replace"):
    if line.startswith("RESULT "):
        rows.append(dict(tok.split("=", 1) for tok in line.split()[1:] if "=" in tok))
keys = []
for r in rows:
    for k in r:
        if k not in keys:
            keys.append(k)
w = csv.DictWriter(sys.stdout, fieldnames=keys, lineterminator="\n")
w.writeheader()
w.writerows(rows)
