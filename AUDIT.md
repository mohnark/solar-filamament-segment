# Audit: MAGFiLO filament segmentation pipeline

## Open — decided, left deliberately

2. ~~`pos_weight=546`~~ — corrected 2026-07-26 to 226. 546 was the inverse
   pixel frequency of the *truncated* masks (see item 4). Measured over 120
   files with the fixed decode, foreground is 0.441% of pixels, so the raw
   inverse frequency is 226. Still unweighted against the dice term
   (`dice_weight=bce_weight=0.5`); if val still shows heavy over-prediction
   after the decode fix, try 10-50 or move to focal/Tversky.

3. ~~`workflow.ipynb` duplicates `run.py` but disagrees with it~~ — resolved
   2026-07-26: `run.py` deleted, `workflow.ipynb` rewritten to match its
   logic (correct file-extension filter, `FileGroupedSampler`, resume,
   AMP, early stop, fixed `build_submission` signature). Notebook is now
   the single maintained copy; run/toggle flags live in its first cell.

4. ~~The `category_ids={1, 2}` filter leaves 44 of 707 files with zero
   foreground annotations~~ — **reversed 2026-07-26.** The decision was
   wrong, and it was the main cause of the flat `dice=0.14` /
   `instance_precision=0.0007` run. Two independent truncations stacked:

   - `build_file_to_annotations` keyed `file_name -> a single image_id`.
     Image ids are `<annotator>-<file>` (`010101`, `010102`, `010103`) and
     1154 image entries cover 707 files, so 447 entries and 2948 of 8199
     annotations were unreachable.
   - `category_ids={1, 2}` dropped all 3074 category-3 (Unidentifiable)
     annotations. Category 4 is declared but has zero instances, so the
     "3/4" framing above was never accurate.

   Together the training masks held 58.5% of annotated filament pixels.
   The other 41.5% are visible filaments labeled background, so the model
   was punished for finding them — which is what the near-zero instance
   precision was measuring. Fixed: all annotators unioned, categories
   `{1, 2, 3}`. Zero files are now empty-foreground.

   The union is deliberate. Pairwise IoU between annotators on multi-
   annotator files is only 0.25-0.57, so consensus/majority voting would
   shrink masks a lot; unverified which convention the private test GT
   uses. Worth an A/B once the union baseline has a score.

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
