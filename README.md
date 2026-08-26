# PokemonCNN

Classify the original 151 Pokemon from an image, with PyTorch.

Two tracks run through the same pipeline: a **hand-built CNN** (`poke_net`) to
learn the mechanics and set a floor, and a **fine-tuned ImageNet backbone**
(ResNet/EfficientNet) to actually reach useful accuracy.

---

## The pipeline

| # | Stage | Command | Output |
|---|---|---|---|
| 1 | Download | `python -m src.download` | `data/raw/pokemon7k/` |
| 2 | Inspect | `python -m src.inspect_data` | `reports/dataset_report.md`, class-distribution chart |
| 3 | Split | `python -m src.prepare` | `data/splits/{train,val,test}.csv`, `classes.json` |
| 4 | Train baseline | `python -m src.train --config configs/baseline.yaml` | `outputs/baseline_pokenet/` |
| 5 | Train transfer | `python -m src.train --config configs/resnet18.yaml` | `outputs/resnet18_ft/` |
| 6 | Evaluate | `python -m src.evaluate --checkpoint outputs/resnet18_ft/best.pt` | confusion matrix, per-class report, mistake grid |
| 7 | Predict | `python -m src.predict --checkpoint outputs/resnet18_ft/best.pt --image pic.jpg` | top-5 labels |

```bash
# a file you saved
python -m src.predict --checkpoint outputs/baseline_pokenet/best.pt --image test_images/pic.jpg --tta
# straight from a URL
python -m src.predict --checkpoint outputs/baseline_pokenet/best.pt --url https://site.com/pikachu.png
# a whole folder, to CSV
python -m src.predict --checkpoint outputs/baseline_pokenet/best.pt --dir test_images --csv reports/preds.csv
```

Each prediction opens a window: the image, the predicted name in large type, and
a bar chart of the top-5 confidences. Close it to continue. Pass `--no-show` to
suppress the windows (scripts, CSV batches, headless runs); more than 12 images
only opens the first 12.

The model always returns one of its 149 classes, with a confidence. Feed it a
golden retriever and it will say Growlithe - softmax has no "none of the above".
A low top-1 with a flat top-5 is the closest it gets to "I don't know".

Optional, any time: `python -m src.benchmark --config configs/resnet18.yaml` measures
throughput on your GPU and estimates epoch and total run time. Add `--synthetic` to
run it before you have any data.

Run every command from the project root, so that `src` is importable as a package.

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux

# PyTorch first, matched to your CUDA version (check with nvidia-smi):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

That last line must print `True`. If it prints `False` you are about to train on
CPU by accident, which is roughly 30x slower.

### Kaggle credentials

`src/download.py` uses the Kaggle API. Get a token at
kaggle.com -> Settings -> API -> **Create New Token**, then save the downloaded
`kaggle.json` to `C:\Users\<you>\.kaggle\kaggle.json`.

### Which dataset

`anujckulkarni/original-151-pokemon` is a **stats table** - name, types, HP,
Attack, Defense - with no images at all. It is kept as *metadata* in
`data/metadata/Gen1_Pokemon.csv`, where its 151 official names serve as the
canonical label list. You cannot train a CNN on it.

The image datasets are registered in `src/download.py`:

```bash
python -m src.download --list                                  # see options
python -m src.download                                          # pokemon151 (default)
python -m src.download --dataset pokemon151 --dataset pokemon7k # merge two
python -m src.download --slug some/other-kaggle-dataset        # anything else
```

| key | slug | notes |
|---|---|---|
| `pokemon151` | `prestigemaster/original-pokemon-151-dataset` | Named for exactly our target; ships its own Train/Test folders |
| `pokemon7k` | `lantian773030/pokemonclassification` | ~7,000 hand-cropped images, one folder per Pokemon |
| `gen1_10k` | `thedagger/pokemon-generation-one` | ~10,000 Gen-I images, one folder per Pokemon |

Each lands in its own subfolder of `data/raw/`. Stages 2 and 3 merge classes
across subfolders automatically, so downloading two roughly doubles your data.

**On datasets that ship a Train/Test split:** we ignore it. `find_class_dirs`
treats any folder holding images as a class and merges same-named folders across
parents, so `Train/Pikachu` and `Test/Pikachu` become one class, and stage 3 draws
its own split *after* de-duplication. That is deliberate - a published split you
did not build is a split you cannot verify, and duplicates straddling it are the
most common source of a suspiciously good score.

Inspect one source on its own before committing:

```bash
python -m src.inspect_data --root data/raw/pokemon151
```

---

## Stage notes - the decisions that actually matter

### 2. Inspect before you train
`inspect_data.py` does not assume a folder layout; it treats any directory that
directly contains images as a class and merges same-named folders. It reports:

- **class count** - if it is not 151, the layout is nested differently than assumed
- **class balance** - a 10x imbalance changes whether you need a weighted sampler
- **alpha channels** - sprite datasets are full of transparent PNGs. Calling
  `.convert("RGB")` on those paints transparent pixels **black**, handing the
  model a hard black silhouette to memorise instead of the Pokemon.
  `utils.load_image_rgb` composites onto white instead.
- **exact duplicates** - dropped in stage 3, because a duplicate straddling train
  and test leaks the answer and quietly inflates your reported accuracy

### 3. Names, then split
Two datasets will not agree on spelling: `Nidoran-f`, `NidoranF`,
`Nidoran(female)`; `MrMime` vs `Mr. Mime`; `Farfetchd` vs `Farfetch'd`. Left
alone, one Pokemon becomes two classes, which caps your ceiling and scrambles the
confusion matrix. `src/names.py` folds every spelling onto the canonical name
from the metadata CSV and reports any folder that does not match.

### 3b. Split once, freeze it
The split is written to CSV and reused by every run, so model A and model B are
comparable. It is stratified per class, and never lets val/test consume a class
entirely. 70 / 15 / 15 by default.

### 4-5. Augmentation
`RandomResizedCrop`, horizontal flip, mild rotation, mild colour jitter, random
erasing. Deliberately **no vertical flip and no strong hue jitter** - Pokemon are
always upright, and identity is partly colour (a hue-shifted Charmander is a
different creature). Mixup is on for the transfer configs.

### 5. Two-phase fine-tuning
Phase 1 freezes the backbone and trains only the new 151-way head for a few
epochs. A randomly initialised head produces huge early gradients; letting those
flow into pretrained weights destroys the features you came for. Phase 2
unfreezes everything at a **10x lower backbone LR** (`lr_backbone`) than the head.

### 6. Read the confusion matrix, not just the accuracy
The interesting output is the most-confused pairs. Expect the model to struggle
with Nidoran-M vs Nidoran-F, the three Eeveelutions, and adjacent evolution
stages. `reports/*_mistakes.png` shows the confidently-wrong predictions, which
is where label noise in the dataset usually surfaces.

---

## Expected results (151 classes, ~1/151 = 0.7% chance baseline)

| Model | Top-1 | Top-5 | Notes |
|---|---|---|---|
| `poke_net` from scratch | **82.2%** measured | **92.8%** | Actual result on this data, 60 epochs, ~40 min |
| `resnet18` fine-tuned | 88-95% | 98%+ | Best accuracy-per-minute. Start here. |
| `resnet50` fine-tuned | 90-96% | 99% | A couple of points for ~3x the compute. |

The from-scratch number is far above the 40-60% you would expect on photographs.
That is not leakage - it is the data. Much of this set is official artwork and
game sprites: canonical, stereotyped renders on clean backgrounds, which is a
much easier problem than photos. Verified by the error structure, not the score
(see below).

### Reading the baseline's mistakes

Every top confusion is an evolution-line neighbour:

```
Charmeleon -> Charmander  7x      Dugtrio    -> Diglett     2x
Marowak    -> Cubone      5x      Arcanine   -> Growlithe   2x
Mew        -> Mewtwo      3x      Tentacruel -> Tentacool   2x
Primeape   -> Mankey      3x      Koffing    -> Weezing     2x
```

And the weakest classes are overwhelmingly **middle evolution stages** -
Charmeleon, Pidgeotto, Machoke, Nidorina, Marowak. A middle stage shares features
with the form below it and the form above it, so it has the least distinctive
silhouette of the three. That is a real visual property of the problem, and a
model that had memorised leaked images would not reproduce it.

Ditto (0.58) is a different failure: it is a featureless pink blob whose whole
gimmick is transforming into other Pokemon, so a chunk of its images arguably
depict something else.

---

## The web app - for everyone else

Other people should not have to touch a terminal. `src/app.py` serves a page
where they drop in an image and press a button.

```bash
pip install flask
python -m src.app
# open http://127.0.0.1:5000
```

An HTML page cannot run PyTorch, so this is a small Flask server doing both jobs:
it serves `web/index.html`, and `POST /api/predict` runs the image through the
checkpoint on your GPU and returns the top-5 as JSON. The page itself is one
self-contained file - no build step, no framework.

It picks the best checkpoint it can find (resnet50 -> resnet18 -> baseline), or
pass `--checkpoint` to force one.

| Flag | Effect |
|---|---|
| `--checkpoint outputs/<run>/best.pt` | use a specific model |
| `--port 8080` | change the port |
| `--host 0.0.0.0` | let others on your wifi use it, at `http://<your-ip>:5000` |
| `--no-tta` | skip mirror-averaging (marginally faster) |

The page supports drag-and-drop, click-to-browse, and **Ctrl+V paste** - so an
image copied straight off a Google results page works without saving a file. Under
40% confidence it says so, because softmax always returns an answer whether or not
the picture contains a Pokemon at all.

## Hosting it publicly (Vercel)

```bash
pip install onnx onnxruntime
python -m src.export_onnx --checkpoint outputs/resnet18_ft/best.pt
cd deploy/vercel
npm i -g vercel && vercel login     # one time
vercel deploy --prod
```

That gives you a permanent public URL on Vercel's free Hobby plan.

Two constraints shaped the design, both from Vercel's documented limits:

**Python functions cap at 500 MB uncompressed.** The PyPI `torch` wheel for Linux
bundles CUDA and blows past that on its own. So `src/export_onnx.py` converts the
checkpoint to ONNX and the function runs ONNX Runtime (~15 MB) instead of PyTorch
(~800 MB). Cold starts drop from several seconds to about one, and the export
verifies that ONNX and PyTorch produce identical predictions before you ship.

**Request bodies cap at 4.5 MB**, and phone photos routinely exceed it. The page
downscales to 640px on a canvas before uploading, putting typical requests under
150 KB. The canvas is filled white first - drawing a transparent PNG straight to
JPEG turns the transparency black, the same trap `load_image_rgb` avoids on the
Python side.

`web/index.html` stays the single source of truth; the export copies it into
`deploy/vercel/`. Re-run both commands after retraining to ship a new model.

`deploy/vercel/app.py` is one Flask app that serves the page at `/` and the API
at `/api/*` - not a split `api/predict.py`-per-route convention. That convention
has a sharp edge: any declared entrypoint flips the whole project into "one app
owns every route" mode, so a handler written only for `/api/predict` crashes
when it gets hit at `/`. Building the whole thing as one Flask app sidesteps
that rather than fighting it.

## How long does this take?

Short answer: minutes, not hours. These datasets are small by deep-learning
standards - roughly 7,000-17,000 images total, versus ImageNet's 1.28 million.

For a ~5,000-image training split at 224px with AMP on a modern NVIDIA GPU:

| Run | Per epoch | Full run |
|---|---|---|
| `resnet18` (30 epochs) | ~10-25 s | ~5-15 min |
| `poke_net` (60 epochs) | ~8-20 s | ~10-20 min |
| `resnet50` at 256px (35 epochs) | ~30-60 s | ~20-40 min |

Merging two datasets roughly doubles those numbers and is still a coffee break.

The usual bottleneck at this scale is **not** the GPU - it is JPEG decode and
augmentation on the CPU. `src/benchmark.py` tells you which one you are hitting;
if it says "data loader", raise `num_workers` before touching anything else.

The real constraint is the opposite of speed: ~7,000 images over 150 classes is
about 47 per class, which is thin. That is what caps accuracy, and it is why
merging datasets is worth more than any hyperparameter you could tune.

## Resuming / extending a run

```bash
python -m src.train --config configs/baseline.yaml --epochs 120 \
    --resume outputs/baseline_pokenet/best.pt
```

`--epochs` is the new **total**, not epochs to add - 60 already done plus 60 more
is `--epochs 120`. Get this wrong (leave it at 60) and the run does nothing: it
sees you are already past epoch 60 and exits immediately.

What resuming does and does not carry over:

- **Model and optimizer weights** load from the checkpoint - training continues
  from where it left off, not from scratch.
- **The learning rate schedule restarts** - a fresh warmup-then-cosine-decay
  curve is built for the new total. This is a deliberate warm restart, not a
  bug: by epoch 60 your baseline's LR had already decayed to ~1e-5 and the loss
  had flattened, so continuing at that same near-zero LR would barely move the
  needle. Jumping back up and re-annealing is the standard technique (SGDR) for
  pushing a plateaued run further, and it is the same idea as the freeze/unfreeze
  phase change already used for transfer learning.
- **history.csv and curves.png accumulate** - epochs 61-120 append to the
  existing record rather than replacing it, so the plot still shows the full
  120-epoch curve, not just the extension.
- **best.pt** only updates if a later epoch actually beats the val top-1
  already in the checkpoint, so extending a run can never silently regress it.

If loss was still visibly dropping at epoch 60 - not your baseline's case, where
it had flattened - plain `--epochs 120` from scratch (no `--resume`) is usually
simpler and just as fast, since nothing was thrown away yet worth preserving.

### The early-stopping trap

`EarlyStopping` restarts fresh on every resume - it has no memory of the run
before it. Its first reading becomes the number every later epoch must beat, and
that first reading is taken **before** the warm restart's LR has ramped up
enough to hurt. A warm restart deliberately makes val loss worse for several
epochs before it (hopefully) gets better; the default `patience: 15` is
routinely not enough runway to survive that round trip; the run stops having
made zero net progress, `best.pt` unchanged, right as it was starting to recover.

Pass a much larger patience on a resume so the schedule gets to actually finish:

```bash
python -m src.train --config configs/baseline.yaml --epochs 120 \
    --resume outputs/baseline_pokenet/best.pt --patience 999
```

`best.pt` is still protected regardless - it only updates when an epoch actually
beats what's already saved, so there is no downside to letting the full 120
epochs run rather than guessing a patience number that might cut it off again.

## Tuning knobs

Any config key can be overridden on the command line:

```bash
python -m src.train --config configs/resnet18.yaml --epochs 40 --batch-size 96 --aug heavy
```

| Symptom | Try |
|---|---|
| CUDA out of memory | halve `batch_size`, or `--img-size 160` |
| Train acc >> val acc | `--aug heavy`, raise `mixup_alpha`, raise `weight_decay` |
| Both accuracies low / flat | raise `lr_head`, lower `freeze_epochs`, train longer |
| Rare classes always wrong | `balanced_sampler: true` |
| Fewer than 151 classes found | Normal - most datasets miss a few. Merge `gen1_10k` for coverage |
| Loss goes NaN | lower LR, keep `clip_grad: 1.0`, check `amp` |
| DataLoader stalls on Windows | `--num-workers 0` |

---

## Layout

```
PokemonCNN/
├── configs/          baseline.yaml, resnet18.yaml, resnet50.yaml
├── web/index.html    the drop-an-image page (source of truth)
├── deploy/vercel/    generated Vercel deployment (ONNX + serverless function)
├── data/
│   ├── raw/<source>/ downloaded images, one subfolder per dataset (gitignored)
│   ├── metadata/     Gen1_Pokemon.csv - the canonical 151 names
│   └── splits/       frozen train/val/test manifests + classes.json
├── outputs/<run>/    best.pt, last.pt, history.csv, curves.png, config.json
├── reports/          dataset report, confusion matrices, mistake grids
└── src/
    ├── utils.py        paths, seeding, device, alpha-safe image loading
    ├── download.py     Kaggle API fetch (image datasets + metadata)
    ├── names.py        folds folder-name spellings onto the canonical 151
    ├── inspect_data.py dataset QC report
    ├── prepare.py      dedupe + stratified split -> CSV manifests
    ├── dataset.py      Dataset, transforms, DataLoaders
    ├── models.py       PokeNet + pretrained backbones, freeze/unfreeze
    ├── engine.py       train/eval loops, AMP, mixup, cosine LR, checkpoints
    ├── train.py        config-driven training CLI
    ├── benchmark.py     throughput + run-time estimator for your GPU
    ├── app.py          Flask server for the web UI
    └── export_onnx.py  checkpoint -> ONNX for serverless hosting
    ├── evaluate.py     test metrics, confusion matrix, mistake analysis
    └── predict.py      inference on new images
```
