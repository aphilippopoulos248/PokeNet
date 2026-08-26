"""Step 4 - Dataset, augmentation and DataLoaders.

This file is the bridge between "a folder full of images" and "batches of
tensors a model can train on." Three jobs live here:

  1. PokemonDataset  - reads one image + label at a time, on request.
  2. build_transforms - decides HOW an image is turned into a tensor, and
                        whether random augmentation is applied.
  3. make_loaders     - wraps datasets in DataLoaders that batch, shuffle and
                        parallelise the work above across CPU worker processes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

# ROOT: the project root, so manifest paths (relative, e.g. "data/raw/.../pic.jpg")
#   resolve correctly no matter what directory a script is run from.
# IMAGENET_MEAN / IMAGENET_STD: the per-channel statistics ImageNet was
#   normalised with. Pretrained backbones (ResNet, EfficientNet...) expect
#   input in this same normalised range - skip this and transfer learning
#   quietly performs far worse, because the pretrained filters were tuned for
#   a different input distribution.
# load_image_rgb: alpha-safe image loader (composites transparency onto white
#   instead of letting it turn black) - see src/utils.py for why that matters.
# read_json: tiny helper, used here to load classes.json.
from src.utils import (
    DATA_SPLITS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    ROOT,
    load_image_rgb,
    read_json,
)


class PokemonDataset(Dataset):
    """Reads a manifest CSV (path, class, label, split) produced by src.prepare.

    A PyTorch `Dataset` only has to answer two questions: "how many items do
    you have?" (`__len__`) and "give me item number i" (`__getitem__`). The
    DataLoader (see make_loaders below) is what turns single items into
    shuffled, batched, multi-worker-loaded tensors - this class stays
    deliberately dumb and just does one image at a time.
    """

    def __init__(self, manifest: str | Path, transform=None, root: Path = ROOT):
        # The manifest is one of train.csv / val.csv / test.csv - every row is
        # one image, already assigned to a split by src/prepare.py. Nothing
        # here decides train/val/test membership; that was decided once,
        # earlier, and frozen to disk so every run sees the same split.
        self.df = pd.read_csv(manifest)
        self.transform = transform
        self.root = Path(root)
        # Cached as a plain numpy array (not read from the DataFrame each
        # time) because make_loaders needs every label up front, cheaply, to
        # build the balanced-class sampler.
        self.labels = self.df["label"].to_numpy()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        # Images are decoded from disk lazily, one at a time, only when the
        # DataLoader actually asks for index `idx` - not all loaded into
        # memory up front. That's what makes this scale to datasets much
        # bigger than RAM.
        row = self.df.iloc[idx]
        img = load_image_rgb(self.root / row["path"])
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row["label"])


def build_transforms(img_size: int = 224, train: bool = False, aug: str = "medium",
                     mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Augmentation is the main defence against overfitting on a few-hundred-per-class set.

    Deliberately NO vertical flip and no aggressive hue jitter: Pokemon identity is
    partly colour (Charmander is orange; a hue-shifted Charmander is a different
    creature) and orientation is always upright.

    Two very different pipelines come out of this function depending on `train`:

    * train=False (val/test/inference) - deterministic. The same image always
      produces the same tensor. This is what "fair" evaluation requires: you
      cannot compare epochs, or trust a confusion matrix, if the eval images
      themselves are randomly changing.
    * train=True - randomised. Every time the model sees an image during
      training it looks slightly different (cropped differently, flipped,
      colour-jittered), which is what stops the model from memorising exact
      pixels instead of learning the underlying shape/colour/texture pattern.
    """
    if not train:
        # Resize the short side up a little past the target, then centre-crop
        # down to exactly img_size. This mirrors standard ImageNet eval
        # practice (resize to 1.14x, crop to 1x) and gives a single,
        # reproducible view of each image.
        return T.Compose([
            T.Resize(int(img_size * 1.14)),
            T.CenterCrop(img_size),
            T.ToTensor(),           # PIL image (0-255 ints) -> float tensor in [0, 1]
            T.Normalize(mean, std),  # shift/scale so each channel is roughly mean-0, std-1
        ])

    # Three preset strengths, so a config file can dial augmentation up or
    # down (configs/resnet50.yaml uses "heavy"; the others use "medium")
    # without anyone having to hand-tune individual transform parameters.
    scale = {"light": (0.85, 1.0), "medium": (0.7, 1.0), "heavy": (0.5, 1.0)}[aug]
    jitter = {"light": 0.1, "medium": 0.2, "heavy": 0.3}[aug]
    erase = {"light": 0.0, "medium": 0.25, "heavy": 0.4}[aug]

    ops = [
        # Crop a random region covering `scale` of the original area (e.g.
        # 70%-100% for "medium"), at a random aspect ratio, then resize that
        # crop up to img_size. This is the single most effective augmentation
        # for classification - it forces the model to recognise a Pokemon
        # from partial, off-centre, differently-framed views instead of
        # always seeing it centred and full-frame.
        T.RandomResizedCrop(img_size, scale=scale, ratio=(0.8, 1.25)),
        T.RandomHorizontalFlip(0.5),  # left-right mirror; fine, a Pikachu facing either way is still a Pikachu
        T.RandomApply([T.RandomRotation(15)], p=0.5),  # small tilts, applied only half the time
        # Brightness/contrast/saturation jitter simulates different lighting
        # and camera conditions. Hue jitter is deliberately kept to a quarter
        # of the others' strength - see the docstring above for why.
        T.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter, hue=jitter / 4),
    ]
    if aug == "heavy":
        # A touch of blur, applied 20% of the time, so the model isn't
        # exclusively trained on razor-sharp renders - helps a little with
        # slightly-blurry real photos at inference time.
        ops.append(T.RandomApply([T.GaussianBlur(3)], p=0.2))
    ops += [T.ToTensor(), T.Normalize(mean, std)]
    if erase > 0:
        # Randomly blacks out a small rectangular patch of the image after
        # normalising. Forces the model to use the WHOLE Pokemon to decide
        # its class rather than fixating on one distinctive patch (e.g. just
        # Pikachu's ears) that a random crop or occlusion might hide.
        ops.append(T.RandomErasing(p=erase, scale=(0.02, 0.15)))
    return T.Compose(ops)


def class_names(splits_dir: Path = DATA_SPLITS) -> list[str]:
    """The ordered list of class names - index i here is what label i means.

    This ordering is decided once by src/prepare.py and written to
    classes.json; every checkpoint stores it too, so a saved model's output
    index 42 always maps back to the same Pokemon name, in training,
    evaluation, and inference alike.
    """
    return read_json(Path(splits_dir) / "classes.json")["class_names"]


def _balanced_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """Give every CLASS roughly equal odds of being sampled, not every IMAGE.

    Without this, a DataLoader with `shuffle=True` samples images uniformly -
    so a class with 700 images shows up ~9x more often per epoch than one with
    82 images (see the dataset report's 8.8x imbalance ratio). Each image's
    sampling weight is set to 1/(size of its class), so a rare class's few
    images are drawn more often, on average balancing exposure across all 149
    classes. Used only when `balanced_sampler: true` in a config - see
    configs/resnet50.yaml.
    """
    counts = np.bincount(labels)
    weights = 1.0 / np.maximum(counts[labels], 1)
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                 num_samples=len(labels), replacement=True)


def make_loaders(img_size: int = 224, batch_size: int = 64, num_workers: int = 4,
                 aug: str = "medium", balanced: bool = False,
                 splits_dir: Path = DATA_SPLITS, pin_memory: bool = True):
    """Build the train/val/test DataLoaders that src/train.py and src/evaluate.py consume.

    A DataLoader wraps a Dataset and adds: batching, shuffling, and (via
    num_workers) parallel image decoding on background CPU processes - so the
    GPU is never left waiting on disk I/O and JPEG decoding between batches.
    """
    splits_dir = Path(splits_dir)
    # Only the training set gets randomised augmentation (train=True); val and
    # test always get the deterministic pipeline, since they exist to measure
    # the model honestly, not to be learned from.
    train_ds = PokemonDataset(splits_dir / "train.csv", build_transforms(img_size, True, aug))
    val_ds = PokemonDataset(splits_dir / "val.csv", build_transforms(img_size, False))
    test_ds = PokemonDataset(splits_dir / "test.csv", build_transforms(img_size, False))

    common = dict(num_workers=num_workers, pin_memory=pin_memory,
                  persistent_workers=num_workers > 0, drop_last=False)
    # persistent_workers keeps the worker processes alive between epochs
    # instead of respawning them every time - respawning is slow. pin_memory
    # copies batches into page-locked host memory, which speeds up the
    # host->GPU transfer; it only helps when there IS a GPU, hence
    # `pin_memory=device.type == "cuda"` is passed in from the caller.
    sampler = _balanced_sampler(train_ds.labels) if balanced else None
    # shuffle and sampler are mutually exclusive in PyTorch - a sampler
    # already decides the (weighted, randomised) order, so shuffle is turned
    # off automatically whenever a balanced sampler is in use.
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=sampler is None,
                          sampler=sampler, **common)
    # Validation/test never need shuffling (order doesn't affect a metric
    # that's averaged over the whole set) and use a bigger batch size, since
    # no gradients are computed here - no need to keep memory free for
    # backpropagation, so more images fit on the GPU at once, which is faster.
    val_dl = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, **common)
    test_dl = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, **common)
    return train_dl, val_dl, test_dl


def denormalize(t: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> torch.Tensor:
    """Undo Normalize() so a tensor can be shown as a real image again.

    A normalised tensor's pixel values are roughly in [-2, 2] and mean next
    to nothing visually - matplotlib would render a washed-out mess. This
    reverses that exact shift/scale (multiply by std, add back mean) and
    clamps to [0, 1], the range imshow() expects. Used by src/evaluate.py to
    draw the "confidently wrong" mistake grid.
    """
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    return (t.cpu() * s + m).clamp(0, 1)
