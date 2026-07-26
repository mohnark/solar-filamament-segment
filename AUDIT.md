# Audit: MAGFiLO filament segmentation pipeline

Date: 2026-07-26 (third pass, post-fix). The AMP blocker, the `validate()`
memory issue, the train-loss denominator, and all Low/housekeeping code items
from the previous pass are fixed — see FIX.md for verification evidence.
What remains below is either not a code defect (no GPU), an explicit user
decision to leave as-is, or still deferred.

## Open — environment, not code

### 1. No GPU in this environment
`torch.cuda.is_available()` → False here. Not a code defect — CPU forward
pass measured at ~32s/image, so training (and even a full val pass) is
impractical without GPU. This is the one thing genuinely blocking a real
training run.

Consequence: the AMP path (`use_amp: true` + `GradScaler`) has never executed
on real hardware. The fp16 overflow fix in `DiceLoss` was verified by feeding
fp16 tensors directly (see FIX.md), not by an actual autocast GPU step.
Re-check the first GPU epoch's loss actually moves.

## Open — decided, left deliberately

These were raised and the user chose to keep current behavior. Recorded here
so they aren't re-flagged as new findings next pass.

2. `pos_weight=546` (`config.json`) is raw inverse pixel frequency,
   unweighted against the dice term (`dice_weight=bce_weight=0.5`). At that
   magnitude the BCE term likely dominates and pushes toward
   over-prediction. **Decision: leave at 546**, tune empirically once a GPU
   run is possible. If val shows heavy over-prediction, try 10-50 or move to
   focal/Tversky.
3. `workflow.ipynb` duplicates `run.py` but filters only `.jpeg` (vs
   `.jpg`/`.png` too) and calls `build_submission(...)` with the old
   signature (no `image_height`/`image_width`) — it will crash if run.
   **Decision: leave as-is**; `run.py` is the maintained copy. Two sources
   of truth that disagree, knowingly.
4. The `category_ids={1, 2}` filter leaves 44 of 707 files with zero
   foreground annotations (every annotation on them was category 3/4), and
   those files still train as pure background. **Decision: keep them as
   legitimate negative examples**, same reasoning as the filter itself.
   Now documented in `build_combined_mask`'s docstring.

## Open — unresolved

5. Left/Right chirality (category 1 vs 2, 62% of labels) is collapsed into
   one binary mask. The category filter decides what counts as foreground at
   all, not whether chirality is predicted. Still unconfirmed whether the
   competition metric scores chirality — if it does, the model architecture
   needs a second output channel, not just a decode change.
6. `scripts/compute_masks.ipynb` iterates `images_by_id` directly, so it
   still writes one mask per *annotator pass* rather than per file, and does
   not apply the category filter — i.e. it disagrees with what
   `SolarFilamentDataset` actually trains on. Its output isn't consumed by
   the pipeline (masks are decoded on the fly), so this is inert today, but
   any mask PNG produced from it is not the training ground truth.

## Deferred (not audited)

- Submission schema (`filament_id,segmentation_rle` format, COCO compressed
  RLE, no `sample_submission.csv` present) — user has deprioritized this.
  Still genuinely unverified; get `sample_submission.csv` and diff before
  spending GPU time on a real run.
