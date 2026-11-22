"""Dataset dùng chung cho các thí nghiệm.

CIFAR-10 được Job dataset-prep chuẩn bị sẵn dạng npz (nguồn Hugging Face uoft-cs/cifar10),
nằm ở <root>/cifar10/{train,test}.npz, kèm meta.json ghi commit nguồn và sha256.
"""
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_ROOT = os.environ.get("DATA_DIR", "/data/datasets")

# Mean/std chuẩn của CIFAR-10 trên tập train (theo kênh RGB, thang 0–1).
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class CIFAR10Npz(Dataset):
    """Trả về (ảnh float32 CHW đã chuẩn hoá, nhãn int64).

    Toàn bộ split nằm trong RAM (train khoảng 150 MB). `transform` nhận tensor CHW
    và áp dụng sau khi chuẩn hoá, dùng cho augmentation.
    """

    def __init__(self, split="train", root=DEFAULT_ROOT, transform=None, normalize=True):
        d = os.path.join(root, "cifar10")
        if not os.path.exists(os.path.join(d, ".complete")):
            raise FileNotFoundError(f"{d} chưa được chuẩn bị; chạy Job dataset-prep trước")
        with np.load(os.path.join(d, f"{split}.npz")) as z:
            x, y = z["x"], z["y"]
        with open(os.path.join(d, "meta.json")) as f:
            self.meta = json.load(f)
        self.classes = self.meta["classes"]
        t = torch.from_numpy(x).permute(0, 3, 1, 2).float().div_(255)
        if normalize:
            mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
            std = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)
            t = (t - mean) / std
        self.x = t.contiguous()
        self.y = torch.from_numpy(y)
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        img = self.x[i]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.y[i]
