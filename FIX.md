# Fixes: MAGFiLO filament segmentation pipeline

Verification pass date: 2026-07-26. All items below re-checked against current
code (read + ran) in this session. Submission schema (sample_submission.csv
diff) deferred, out of scope for this pass.

## Fixed this pass (third pass)

- **`DiceLoss` overflowed to `inf` under AMP** (`segres/losses.py`, was the
  open blocker): `preds.sum(dim=1)` ran at fp16 inside `torch.autocast`, and
  a 512x512 tile is 262144 pixels against an fp16 max of 65504 — so the sum
  overflowed once mean sigmoid output passed ~0.25, pinning `dice_loss` at
  1.0 and sending NaN gradients that `GradScaler` then skipped every step,
  forever, silently. `forward` now does `torch.sigmoid(preds.float())` and
  `targets.float()` before the reduction. Verified: the exact fp16 input that
  used to produce `union=inf` / `loss=1.0` / NaN grad now gives `loss=0.981`
  with an all-finite gradient. Note the AMP path still hasn't run on a real
  GPU — no CUDA here.

- **`validate()` buffered the entire val set in RAM** (`segres/train.py`):
  every tile's prediction and ground truth was held until the loop ended
  (~52 MB/file, ~5.6 GB over a 107-file val split, before DataLoader
  prefetch). Now each image is scored and freed as soon as its
  `len(dataset.tile_coords)` tiles have arrived; predictions are buffered as
  `float16` and ground truth as `uint8`. Per-image scoring moved into
  `_ImageScoreAccumulator`; a post-loop pass scores any straggler rather
  than dropping it. Verified two ways on real data: (a) 5 files, each scored
  with exactly 25/25 tiles, metrics unchanged in shape; (b) against a loader
  truncated after 6 batches, the first file was already scored (25 tiles)
  while only a 5-tile partial remained — proving the flush happens mid-loop,
  so peak buffer is one image, not the whole split.

- **Train loss divided by the wrong denominator** (`segres/train.py`):
  `running_loss / len(loader.dataset)` under `drop_last=True` excluded up to
  `batch_size - 1` samples from the numerator but not the denominator,
  under-reporting train loss. Now divides by samples actually seen.

- **Normalization mismatched the pretrained encoder** (`segres/transform.py`):
  was mean/std 0.5/0.5 while the resnet34 encoder is ImageNet-pretrained
  (smp adapts it to `in_channels=1` by summing the RGB conv weights, so it
  still expects ImageNet-normalized input). Now 0.449/0.226 — the ImageNet
  RGB stats collapsed to luminance.

- **`scripts/compute_masks.ipynb` bootstrap cell was empty**: the notebook
  only imported when cwd == repo root. Restored a `sys.path` cell that
  resolves the repo root from either location. Verified importing from
  `scripts/` and from the repo root.

- **Dead code**: deleted `tile_image_and_mask` (`segres/tiling.py`) and
  `get_image_and_masks` (`segres/decode.py`), both unused by the pipeline,
  and dropped them from `segres/__init__.py`'s exports. `import segres`
  verified clean.

- **44 zero-foreground files documented**, not changed: the user's call is to
  keep them as negative examples. Recorded in `build_combined_mask`'s
  docstring with the measured numbers so it isn't re-flagged as a finding.

## Verified fixed (earlier passes)

- **Throughput bug** (`segres/dataset.py`, full 2048² JPEG re-decoded per
  tile, 417ms/tile shuffled → 26 min/epoch, 8.7h/20 epochs): `_load_file`
  caches image+mask together per file (one `lru_cache`), `FileGroupedSampler`
  clusters each file's tiles together per epoch instead of global tile
  shuffle. Wired into `run.py:build_loaders` (`sampler=train_sampler`,
  `set_epoch(epoch)` called each epoch). Code inspected, cache/sampler logic
  correct.

- **Ground truth was union of 3 disagreeing annotators** (`segres/decode.py`):
  `build_file_to_annotations` now takes only the first annotator pass (lowest
  `image_id`) per file instead of merging all. Ran against the real
  annotation file: 1154 image records / 8199 annotations → 707 unique files /
  5251 annotations (was 8199 unioned). 707 files, matches disk file count
  exactly.

- **`val_dice` inflated, metric-mismatched** (`segres/train.py`): `validate()`
  stitches tiles to full images before scoring, splits dice (non-empty images
  only) from `empty_correct_frac` (empty images), adds instance-level
  precision/recall/F1 via greedy IoU matching. Ran on 3 real training images
  with an untrained (random-init) model: `dice=0.0023`, `empty_correct=1.0`
  (n=0 empty in this sample), `instance_f1=0.0` — no longer floors near 0.73
  the way the old smoothed per-tile dice did.

- **No `pos_weight` against 1:546 imbalance** (`segres/losses.py`):
  `DiceBCELoss` takes `pos_weight`, registered as a buffer, wired from
  `config.json["pos_weight"] = 546`. Confirmed passed through in `run.py`.

- **Inference downloaded ImageNet weights needlessly** (`segres/predict.py`):
  `load_model` calls `build_model(encoder_weights=None)`. Confirmed
  `build_model` accepts and threads the param instead of hardcoding
  `"imagenet"`.

- **No category filtering** (`segres/decode.py`): `build_combined_mask`
  defaults to `category_ids={1, 2}` (Left/Right only, excludes 3
  Unidentifiable / 4 Ambiguous). Ran against real annotations: 5251 anns →
  3296 after filter. Caveat found this pass: 44 of 707 files end up with
  *zero* foreground annotations after the filter (all their anns were
  category 3/4) — decided: kept as negatives, see AUDIT.md.

- **`min_area: 20` ~60x too small** (`config.json`): now `200`. Confirmed in
  config and threaded through `run.py` to both `validate()` and
  `build_submission`.

- **Image size inferred from tile extents** (`segres/predict.py`):
  `build_submission` now takes explicit `image_height`/`image_width` params
  instead of inferring from tile coords; `run.py:predict()` passes them from
  config. Confirmed no more `max(coords) + tile_size` inference in the file.

- **Non-uniform stride at tile edges** (`segres/tiling.py`): `_axis_coordinates`
  spaces tiles evenly via `np.linspace`. Ran
  `get_tile_coordinates(2048, 2048, 512, 64)`: y ∈ {0, 384, 768, 1152, 1536},
  uniform stride 384 throughout (was {448,448,448,448,192}), still 25
  tiles/image. Latent `y_clamped`/`UnboundLocalError` risk from the old
  while-loop is gone with the rewrite.

- **`ndimage.label` ran twice over the full mask** (`segres/predict.py`):
  merged into one `label_and_filter()` — labels once, filters by area via
  `np.bincount`, returns cleaned mask + instance list together. Used in both
  `build_submission` and `train.py:validate`'s instance matching. Confirmed
  single call site each, no duplicate labeling.

- **`predict()` never created its output dir** (`run.py`): `os.makedirs`
  before `predict()`'s `to_csv`. Confirmed present at `run.py:133`.

- **No CLI** (`run.py`): `argparse` with `--train-only`, `--predict-only`
  (mutually exclusive), `--resume`. Confirmed `main()` branches correctly on
  both flags.

- **No LR schedule, no AMP, no early stopping, no `drop_last`**: all present
  — `drop_last=True` on train loader, `ReduceLROnPlateau(mode="max")` stepped
  on `metrics["dice"]`, AMP via `torch.autocast` + `GradScaler` gated on
  `use_amp` and CUDA, early stop after `early_stop_patience` epochs without
  improvement. (The AMP fp16 bug found that pass is now fixed — see above.)

## Not re-verified this pass (unchanged from prior audit, still true)

- `#4` No GPU in this environment (`torch.cuda.is_available()` → False,
  confirmed again this pass). Not fixable in code; UNet++/resnet34 on CPU
  measured at ~32s/image forward pass here — training remains infeasible
  without GPU access. Still blocking an actual training run, just not a code
  defect.
- Submission schema (`sample_submission.csv` diff) — user has deprioritized
  this for the current pass. Still genuinely unverified, revisit before
  spending GPU time on a real run.

## Stale claims in the old audit doc, corrected this pass

- Previous doc said `scripts/compute_masks.ipynb` was fixed to import
  `segres.decode` — true, but the same edit also emptied the
  `sys.path.insert(0, os.path.abspath(".."))` bootstrap cell. The notebook
  now only imports successfully if run with cwd == repo root; from
  `scripts/` it will `ModuleNotFoundError`. Fixed this pass, see above.
- Previous doc said `src/` → `segres/` rename was unstaged. Already committed
  (`f82e9c6`); working tree is clean of it now.
