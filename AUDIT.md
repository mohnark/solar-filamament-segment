# Audit: MAGFiLO filament segmentation pipeline

Date: 2026-07-26 (post-fix review). Prior blockers (throughput, annotator
union) and all High/Medium items from the original audit are fixed — see
FIX.md for verification evidence. This file lists what's still open,
including new issues found while verifying the fixes. Submission schema
verification deferred (user call, not re-audited this pass).

## Blockers

### 1. `segres/losses.py:22` — `DiceLoss` overflows to `inf` under AMP
`preds.sum(dim=1)` and `targets.sum(dim=1)` run at fp16 precision inside
`torch.autocast`. A 512×512 tile has 262144 pixels; fp16 max is 65504, so the
sum overflows once mean sigmoid output exceeds ~0.25 — trivially true for an
early-training or all-positive-leaning model. Confirmed directly:

```
torch.full((1, 512*512), 0.3, dtype=torch.float16).sum(dim=1)  # -> inf
```

`union = inf` → `dice_score = 0` → `dice_loss = 1.0` pinned, and the gradient
through the overflowed sum is `NaN`. `GradScaler` will detect the `NaN` and
skip the optimizer step — every step, indefinitely, since the same tiles
overflow every batch. `config.json["use_amp"]: true`, so this fires on the
first GPU batch and silently stalls training (loss never moves, no error
raised — scaler just keeps skipping and shrinking its scale factor).

Fix: cast to fp32 before the reduction in `DiceLoss.forward`:
```python
preds = torch.sigmoid(preds.float())
targets = targets.float()
```

## High

### 2. `segres/train.py:94-112` — `validate()` buffers entire val set in RAM
Every val tile's prediction and ground truth (float32, 512×512) is appended
to a dict and held until all files are processed, only then stitched. On the
real train/val split (~107 val files × 25 tiles × 2 arrays × 512×512×4 bytes)
this is ~5.6 GB, measured by extrapolation from a 3-file sample (157 MB for
3 files → ~52 MB/file). Add DataLoader worker prefetch on top and this is a
real OOM risk on constrained instances (Kaggle inference kernels, laptop
dev).

`val_loader` is `shuffle=False` and the dataset index is file-major, so tiles
for one file arrive contiguously — flush and score each file as soon as its
`len(dataset.tile_coords)` tiles have arrived, instead of holding everything
until the loop ends. Cheaper partial fix if a full rewrite isn't wanted:
store `gt` as `uint8` and `preds` as `float16` in the buffer (~2.1 GB
instead of 5.6 GB).

### 3. No GPU in this environment
`torch.cuda.is_available()` → False here. Not a code defect — CPU forward
pass measured at ~32s/image in this session, so training (and even a full
val pass, ~110s for 3 images here) is impractical without GPU. Re-confirm
this is just an environment gap, not needed elsewhere.

## Medium

4. `segres/decode.py` category filter (`{1, 2}` only) drops 5251→3296
   annotations on the real data, and **44 of 707 files end up with zero
   foreground annotations** (every annotation on that file was category
   3/4). Those files still train as pure background — deliberate given the
   category-3/4 rationale, but wasn't quantified before. Confirm intent:
   keep them as legitimate negative examples, or check whether any of those
   44 images visibly contain filaments that are now unlabeled.
5. `segres/losses.py` `pos_weight=546` is raw inverse pixel frequency,
   unweighted against the dice term (`dice_weight=bce_weight=0.5`). At that
   magnitude the BCE term likely dominates and pushes toward
   over-prediction. Needs empirical tuning once a GPU run is possible —
   start lower (10-50) or move to focal/Tversky as originally suggested.
6. `segres/train.py:train_one_epoch` divides running loss by
   `len(loader.dataset)`, but `drop_last=True` means up to
   `batch_size - 1` samples are silently excluded from that denominator's
   numerator — reported train loss is a slightly low estimate. Cosmetic,
   fix by dividing by samples actually seen instead.

## Low / housekeeping

- `scripts/compute_masks.ipynb` — import fixed to `segres.decode`, but the
  same change emptied the `sys.path.insert(0, os.path.abspath(".."))`
  bootstrap cell. Notebook only imports successfully when run with cwd ==
  repo root; from `scripts/` (its own directory) it'll `ModuleNotFoundError`.
  Either restore the bootstrap cell or document the required cwd.
- `workflow.ipynb` still duplicates `run.py` but filters only `.jpeg`
  (vs `.jpg`/`.png` too) and calls `build_submission(...)` with the old
  signature (no `image_height`/`image_width`) — will crash if run. Not
  touched by any fix; `run.py` is the maintained copy. Still two sources of
  truth that disagree.
- `segres/__init__.py` still exports `get_image_and_masks`
  (`segres/decode.py`) and `tile_image_and_mask` (`segres/tiling.py`) —
  both dead code, unused by the pipeline.
- `segres/transform.py` — normalizes to [-1,1] (mean/std 0.5) while encoder
  is ImageNet-pretrained. Minor mismatch, unchanged from original audit.
- Left/Right chirality (category 1 vs 2, 62% of labels) is still collapsed
  into one binary mask. The category filter decides what counts as
  foreground at all, not whether chirality is predicted. Still unconfirmed
  whether the competition metric scores chirality.

## Deferred (not re-audited this pass)

- Submission schema (`filament_id,segmentation_rle` format, COCO compressed
  RLE, no `sample_submission.csv` present) — user has deprioritized this for
  now. Still genuinely unverified; get `sample_submission.csv` and diff
  before spending GPU time on a real run.
