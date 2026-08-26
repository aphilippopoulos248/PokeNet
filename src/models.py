"""Step 5 - the models.

Two tracks on purpose:
  * PokeNet    - a hand-rolled conv stack. This is the one that teaches you what
                 a CNN is. Expect ~40-60% top-1 on 151 classes; that is normal.
  * pretrained - ResNet / EfficientNet fine-tuned from ImageNet weights. This is
                 the one that actually works. Expect 85-95%+ top-1.

A convolutional neural network, at a glance: a stack of small learned filters
slides over the image looking for patterns (edges, then textures, then shapes,
then whole object parts, the deeper you go), each layer built on the patterns
the layer before it found. `ConvBlock` below is one "look for patterns, then
shrink the image" step; `PokeNet` stacks five of them. The pretrained models
in `_BACKBONES` are the same idea, just far deeper, and already trained on 1.28
million ImageNet photos before you ever touch them - `build_model` re-purposes
their learned filters for Pokemon instead of starting from nothing.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU, twice, then downsample by 2.

    BatchNorm before the activation is what lets a from-scratch stack this deep
    train at all without careful init babysitting.

    Reading the pieces:
      * Conv2d(3x3, padding=1) - a learned filter that slides across the image;
        padding=1 keeps the output the same width/height as the input (so the
        MaxPool below is what does the actual shrinking, not the convolution).
      * bias=False - a Conv2d's bias would just get immediately cancelled out
        by the BatchNorm that follows it, so it is dropped to save a few
        parameters at zero cost to the model's capacity.
      * BatchNorm2d - re-centres and re-scales each channel's activations
        using the current batch's statistics. Without this, stacking five
        conv blocks from random initialisation is prone to exploding or
        vanishing activations; with it, training is dramatically more stable.
      * ReLU(inplace=True) - the nonlinearity. Without something like this
        between layers, stacking multiple convolutions would collapse
        mathematically into one big linear operation, no more powerful than a
        single layer.
      * Two Conv-BN-ReLU passes before pooling - lets the block combine
        patterns detected by the first pass before committing to shrink the
        image, rather than pooling immediately after a single, shallower look.
      * MaxPool2d(2) - keeps only the strongest activation in every 2x2 patch,
        halving both height and width. This is what makes each successive
        block "see" a coarser, more zoomed-out view of the image.
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
        # Dropout2d zeroes out entire channels at random (not individual
        # pixels) during training, which discourages any one filter from
        # becoming a single point of failure the network over-relies on. Only
        # used in the deeper blocks (dropout=0.1), where overfitting risk is
        # highest because there are more channels/parameters to memorise with.
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.block(x))


class PokeNet(nn.Module):
    """Baseline CNN: 5 blocks, 32->512 channels, global average pool, linear head.

    Global average pooling instead of a flattened dense layer keeps the parameter
    count ~5M instead of ~50M, which matters a lot when you have a few tens of
    thousands of images.

    The shape of the data as it flows through, for a 224x224 input image and
    the default width=32 (channel counts double each block, spatial size
    halves each block via the MaxPool inside ConvBlock):

        input            3 x 224 x 224
        ConvBlock(3->32)    32 x 112 x 112
        ConvBlock(32->64)   64 x  56 x  56
        ConvBlock(64->128) 128 x  28 x  28
        ConvBlock(128->256) 256 x  14 x  14
        ConvBlock(256->512) 512 x   7 x   7
        AdaptiveAvgPool2d(1) 512 x   1 x   1   <- one number per channel
        Linear(512 -> 151)  151                <- one score per Pokemon

    By the last block, each of the 512 channels has learned to respond to some
    combination of colour/shape/texture pattern; global average pooling then
    just asks "how strongly, on average, did this whole image trigger that
    pattern?" for each of the 512 channels, and the final Linear layer learns
    which combinations of those 512 answers point to which Pokemon.
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
        # AdaptiveAvgPool2d(1): averages every channel's 7x7 grid of
        # activations down to a single number, regardless of the input image
        # size. This is the trick that keeps the parameter count small - the
        # alternative (flattening 512x7x7=25,088 values straight into a dense
        # layer) would need a 25,088 x 151 weight matrix, ~3.8M params in that
        # ONE layer alone, and would also hard-code a fixed input resolution.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),          # 512x1x1 -> a flat vector of 512
            nn.Dropout(dropout),   # randomly zero some of those 512 features each step, to fight overfitting
            nn.Linear(w * 16, num_classes),  # 512 -> 151 raw class scores ("logits")
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Set starting weights deliberately, instead of relying on PyTorch's defaults.

        Training from scratch (no pretrained weights to start from) is far
        more sensitive to initialisation than fine-tuning is, so this is worth
        doing explicitly:
          * Conv2d - Kaiming/He initialisation, tuned for ReLU networks
            specifically (`nonlinearity="relu"`). Keeps the scale of
            activations roughly consistent from one layer to the next at the
            start of training, instead of shrinking or blowing up layer by
            layer before training has even begun.
          * BatchNorm2d - starts as a pure identity operation (weight=1,
            bias=0), so it does nothing until training data teaches it
            otherwise.
          * Linear - small random weights (std=0.01) and a zero bias, which
            keeps the very first predictions close to a flat, unopinionated
            distribution over all 151 classes rather than an arbitrary
            strong (and wrong) initial guess.
        """
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
        # features: the conv stack (pattern detection + downsampling)
        # pool: 512x7x7 -> 512x1x1 (global average pooling)
        # classifier: 512 -> 151 raw scores. Note there is deliberately no
        # softmax here - nn.CrossEntropyLoss (used in engine.py) expects raw
        # logits and applies softmax internally, and applying it twice would
        # be both redundant and numerically worse.
        return self.classifier(self.pool(self.features(x)))


# Every entry: (constructor function, ImageNet-pretrained weights enum, name of
# the final classification layer on that architecture). torchvision names this
# last layer differently per family - ResNet calls it `fc`, EfficientNet and
# MobileNet call it `classifier` - so build_model needs to know which to swap out.
_BACKBONES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, "fc"),
    "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1, "fc"),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2, "fc"),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1, "classifier"),
    "mobilenet_v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.IMAGENET1K_V2, "classifier"),
}


def build_model(name: str = "resnet18", num_classes: int = 151, pretrained: bool = True,
                dropout: float = 0.2, width: int = 32) -> nn.Module:
    """The single entry point every other file uses to construct a model.

    `name="poke_net"` returns the from-scratch PokeNet above. Anything else
    must be a key in `_BACKBONES`: a torchvision architecture, pretrained on
    ImageNet's 1000 classes, with its final layer swapped out for one that
    outputs `num_classes` (151) scores instead - the rest of the network's
    pretrained weights are left untouched here (freezing/unfreezing them is a
    separate, later decision - see set_backbone_trainable).
    """
    if name == "poke_net":
        # PokeNet always uses a healthy dropout floor (min 0.3), since it has
        # no pretrained knowledge to fall back on and is the model most prone
        # to overfitting the training set.
        return PokeNet(num_classes=num_classes, width=width, dropout=max(dropout, 0.3))

    if name not in _BACKBONES:
        raise ValueError(f"unknown model '{name}'. options: poke_net, {', '.join(_BACKBONES)}")

    ctor, weights, head = _BACKBONES[name]
    # weights=None if pretrained is False - trains that same architecture
    # from random initialisation instead. Rarely what you want (that's what
    # poke_net is for), but useful for A/B comparisons.
    model = ctor(weights=weights if pretrained else None)

    if head == "fc":
        # ResNet family: replace the single Linear(in_features -> 1000) layer
        # with Dropout + Linear(in_features -> 151). in_features is read off
        # the existing layer rather than hard-coded, since it differs between
        # resnet18/34 (512) and resnet50 (2048).
        in_f = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
    else:
        # EfficientNet/MobileNet family: `classifier` is already a small
        # Sequential (Dropout, Linear); only the final Linear needs replacing,
        # and the existing Dropout's rate is adjusted in place to match the
        # config rather than adding a second Dropout layer.
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, num_classes)
        if dropout > 0 and isinstance(model.classifier[0], nn.Dropout):
            model.classifier[0].p = dropout
    return model


def head_parameter_names(model: nn.Module) -> list[str]:
    """Name prefix of the classification head's parameters (e.g. "fc.").

    Used by both set_backbone_trainable and param_groups below to tell "the
    part we just randomly re-initialised for 151 Pokemon" apart from "the part
    that still holds pretrained ImageNet knowledge" - those two groups get
    treated very differently during transfer learning.
    """
    for attr in ("fc", "classifier", "head"):
        if hasattr(model, attr):
            return [f"{attr}."]
    return []


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze/unfreeze everything except the classification head.

    Phase 1 trains only the head on frozen ImageNet features (fast, stable).
    Phase 2 unfreezes the backbone at a much lower LR to adapt the features.

    `requires_grad = False` on a parameter tells PyTorch's autograd not to
    compute gradients for it at all - it is skipped entirely during
    backward(), so a frozen backbone is not just "not updated", it is not
    even touched, which is also why phase 1 trains noticeably faster than
    phase 2. The classification head's parameters are always left trainable
    regardless of `trainable`, since a freshly-reinitialised head with frozen
    weights would never learn anything.
    """
    prefixes = tuple(head_parameter_names(model))
    for name, p in model.named_parameters():
        p.requires_grad = True if name.startswith(prefixes) else trainable


def param_groups(model: nn.Module, lr_head: float, lr_backbone: float) -> list[dict]:
    """Discriminative learning rates: small LR for pretrained features, big for the new head.

    PyTorch optimizers accept a list of parameter groups, each with its own
    learning rate, instead of one flat rate for the whole model. That matters
    here because the head starts from random weights (it needs to move fast
    to learn anything at all) while the backbone starts from good, pretrained
    ImageNet weights (it should only be nudged gently, or fine-tuning
    destroys - "catastrophically forgets" - what made those weights useful in
    the first place). Frozen parameters (requires_grad=False) are skipped
    entirely, since handing a frozen tensor to the optimizer would be pointless.
    """
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


def load_model_from_checkpoint(path: str | Path, device="cpu"):
    """Rebuild the architecture recorded in a checkpoint and load its weights.

    Returns (model in eval mode, class_names, config).

    A checkpoint (see engine.save_checkpoint) stores the trained weights
    alongside the config that was used to build the model - not the
    architecture itself. So loading one is a two-step process: first call
    build_model with the SAME settings (model name, dropout, width) that
    produced this checkpoint, which creates a model with the right shape but
    random weights, then overwrite those random weights with the saved ones.
    Get the config wrong here and load_state_dict raises immediately, since
    the saved tensor shapes won't match the freshly-built model's shapes.

    `engine.load_checkpoint` is imported here, inside the function, rather
    than at the top of the file, purely to keep this module's import list
    light for callers (e.g. src/export_onnx.py) that only need build_model
    and have no other reason to pull in engine.py.
    """
    from src.engine import load_checkpoint

    ck = load_checkpoint(Path(path), map_location=device)
    cfg = ck.get("config", {})
    names = ck.get("class_names", [])
    model = build_model(cfg.get("model", "resnet18"), len(names) or 151,
                        pretrained=False, dropout=cfg.get("dropout", 0.2),
                        width=cfg.get("width", 32))
    model.load_state_dict(ck["model_state"])
    # .eval() switches off dropout and freezes BatchNorm's running statistics
    # - both behave differently during training vs inference, and forgetting
    # this step is a classic way to get inconsistent or just-plain-worse
    # predictions out of an otherwise correctly-loaded model.
    return model.to(device).eval(), names, cfg
