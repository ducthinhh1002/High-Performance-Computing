"""Mô hình dùng chung.

- alexnet_gray: theo báo cáo gốc (§1.1, §3.1): ảnh xám 1×224×224, conv đầu 96 kernel 11×11
  stride 4, đầu ra 40 lớp. Các lớp sau theo AlexNet chuẩn (bài không nêu chi tiết).
- resnet18_cifar: ResNet-18 cho ảnh 32×32 (conv đầu 3×3 stride 1, bỏ maxpool), khoảng 11M tham số.
- vgg16_cifar: VGG-16 của torchvision, classifier giữ nguyên 25088→4096→4096 nên p rất lớn
  (khoảng 134M, phần lớn ở FC). Dùng để có p lớn cho E3/E5.
"""
import torch.nn as nn
import torchvision


def alexnet_gray(num_classes=40):
    return nn.Sequential(
        nn.Conv2d(1, 96, kernel_size=11, stride=4), nn.ReLU(inplace=True),       # 224 → 54
        nn.MaxPool2d(3, 2),                                                     # 54 → 26
        nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(inplace=True),
        nn.MaxPool2d(3, 2),                                                     # 26 → 12
        nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
        nn.MaxPool2d(3, 2),                                                     # 12 → 5
        nn.AdaptiveAvgPool2d((6, 6)), nn.Flatten(),
        nn.Dropout(0.5), nn.Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=True),
        nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(inplace=True),
        nn.Linear(4096, num_classes),
    )


def resnet18_cifar(num_classes=10):
    m = torchvision.models.resnet18(num_classes=num_classes)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


def vgg16_cifar(num_classes=10):
    return torchvision.models.vgg16(num_classes=num_classes)


def resnet18_gn_cifar(num_classes=10):
    """ResNet-18 thay BatchNorm bằng GroupNorm(8): không có thống kê chạy, nên không phụ thuộc thành phần lớp của batch (E5b)."""
    m = torchvision.models.resnet18(num_classes=num_classes, norm_layer=lambda c: nn.GroupNorm(8, c))
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


EXTRA_MODELS = {"resnet18_gn_cifar": (resnet18_gn_cifar, (3, 32, 32), 10)}

# tên → (hàm tạo, shape một ảnh đầu vào, số lớp)
MODELS = {
    "alexnet_gray": (alexnet_gray, (1, 224, 224), 40),
    "resnet18_cifar": (resnet18_cifar, (3, 32, 32), 10),
    "vgg16_cifar": (vgg16_cifar, (3, 32, 32), 10),
}


def build(name, device):
    fn, shape, ncls = {**MODELS, **EXTRA_MODELS}[name]
    return fn(ncls).to(device), shape, ncls


def n_params(model):
    return sum(p.numel() for p in model.parameters())
