# Audit: MAGFiLO filament segmentation pipeline

## Open — decided, left deliberately

2. `pos_weight=546` (`config.json`) is raw inverse pixel frequency,
   unweighted against the dice term (`dice_weight=bce_weight=0.5`). At that
   magnitude the BCE term likely dominates and pushes toward
   over-prediction. **Decision: leave at 546**, tune empirically once a GPU
   run is possible. If val shows heavy over-prediction, try 10-50 or move to
   focal/Tversky.

3. ~~`workflow.ipynb` duplicates `run.py` but disagrees with it~~ — resolved
   2026-07-26: `run.py` deleted, `workflow.ipynb` rewritten to match its
   logic (correct file-extension filter, `FileGroupedSampler`, resume,
   AMP, early stop, fixed `build_submission` signature). Notebook is now
   the single maintained copy; run/toggle flags live in its first cell.

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


## Deferred (not audited)

- Submission schema (`filament_id,segmentation_rle` format, COCO compressed
  RLE, no `sample_submission.csv` present) — user has deprioritized this.
  Still genuinely unverified; get `sample_submission.csv` and diff before
  spending GPU time on a real run.
