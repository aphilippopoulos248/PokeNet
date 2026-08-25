"""Step 5 - the models.

Two tracks on purpose:
  * PokeNet    - a hand-rolled conv stack. This is the one that teaches you what
                 a CNN is. Expect ~40-60% top-1 on 151 classes; that is normal.
  * pretrained - ResNet / EfficientNet fine-tuned from ImageNet weights. This is
                 the one that actually works. Expect 85-95%+ top-1.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU, twice, then downsample by 2.

    BatchNorm before the activation is what lets a from-scratch stack this deep
    train at all without careful init babysitting.
    """

    def __init__(self, c_in: int, c_out: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.block(x))


class PokeNet(nn.Module):
    """Baseline CNN: 5 blocks, 32->512 channels, global average pool, linear head.

    Global average pooling instead of a flattened dense layer keeps the parameter
    count ~5M instead of ~50M, which matters a lot when you have a few tens of
    thousands of images.
    """

    def __init__(self, num_classes: int = 151, width: int = 32, dropout: float = 0.3):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            ConvBlock(3, w),              # 224 -> 112
            ConvBlock(w, w * 2),          # 112 -> 56
            ConvBlock(w * 2, w * 4, 0.1),  # 56 -> 28
            ConvBlock(w * 4, w * 8, 0.1),  # 28 -> 14
            ConvBlock(w * 8, w * 16, 0.1),  # 14 -> 7
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(w * 16, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


_BACKBONES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, "fc"),
    "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1, "fc"),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2, "fc"),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1, "classifier"),
    "mobilenet_v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.IMAGENET1K_V2, "classifier"),
}


def build_model(name: str = "resnet18", num_classes: int = 151, pretrained: bool = True,
                dropout: float = 0.2, width: int = 32) -> nn.Module:
    if name == "poke_net":
        return PokeNet(num_classes=num_classes, width=width, dropout=max(dropout, 0.3))

    if name not in _BACKBONES:
        raise ValueError(f"unknown model '{name}'. options: poke_net, {', '.join(_BACKBONES)}")

    ctor, weights, head = _BACKBONES[name]
    model = ctor(weights=weights if pretrained else None)

    if head == "fc":
        in_f = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
    else:
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, num_classes)
        if dropout > 0 and isinstance(model.classifier[0], nn.Dropout):
            model.classifier[0].p = dropout
    return model


def head_parameter_names(model: nn.Module) -> list[str]:
    for attr in ("fc", "classifier", "head"):
        if hasattr(model, attr):
            return [f"{attr}."]
    return []


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze/unfreeze everything except the classification head.

    Phase 1 trains only the head on frozen ImageNet features (fast, stable).
    Phase 2 unfreezes the backbone at a much lower LR to adapt the features.
    """
    prefixes = tuple(head_parameter_names(model))
    for name, p in model.named_parameters():
        p.requires_grad = True if name.startswith(prefixes) else trainable


def param_groups(model: nn.Module, lr_head: float, lr_backbone: float) -> list[dict]:
    """Discriminative learning rates: small LR for pretrained features, big for the new head."""
    prefixes = tuple(head_parameter_names(model))
    head, backbone = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head if name.startswith(prefixes) else backbone).append(p)
    groups = [{"params": head, "lr": lr_head}]
    if backbone:
        groups.append({"params": backbone, "lr": lr_backbone})
    return groups
