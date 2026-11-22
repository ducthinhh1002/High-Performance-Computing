"""Tách các khối '----- BEGIN FILE x -----' trong log Job thành file riêng.
Dùng: python tools/extract_csv.py <log> <thư mục đích>. Khối trùng tên được thêm hậu tố theo thứ tự."""
import os, re, sys
log, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
txt = open(log, errors="replace").read()
seen = {}
for m in re.finditer(r"----- BEGIN FILE (\S+) -----\n(.*?)\n----- END FILE \1 -----", txt, re.S):
    name = m.group(1); seen[name] = seen.get(name, 0) + 1
    stem, ext = os.path.splitext(name)
    open(os.path.join(out, f"{stem}_{seen[name]}{ext}"), "w").write(m.group(2) + "\n")
print({k: v for k, v in seen.items()})
