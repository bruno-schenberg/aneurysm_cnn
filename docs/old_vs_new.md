# Old Pipeline vs New Pipeline: Performance Gap Analysis

## Context

The old pipeline (single-fold, fixed 20 epochs, Adam optimizer) achieved on dataset D with
inverse-frequency class weighting:

- Accuracy: 0.713 | Precision: 0.406 | Recall: 0.619 | F2: 0.560

Current pipeline results on the same dataset (D128, wcf) are consistently worse:

- Job 4303 (f2 checkpoint): Recall 0.484, F2 0.452
- Job 4394 (auc checkpoint): Recall 0.087, F2 0.105 (near-degenerate)

The following differences between the two codebases are the most likely explanations.

---

## Differences

### 1. Checkpoint selection — AUC checkpoint selecting zero-recall models (job 4394)

**Old:** No checkpointing. Final epoch weights used.
**New:** Best checkpoint selected by `checkpoint_metric` (AUC, F2, or val_loss).

AUC is computed on soft probabilities, not argmax predictions. A model that never crosses the
0.5 threshold for a hard positive prediction can still achieve AUC ~0.70 by mildly ranking
positives above negatives in probability space. The AUC-based checkpoint selects these early,
high-AUC-but-zero-recall states (best_epoch 4–12 in most failing 4394 runs).

This explains the pattern seen throughout job 4394: `precision=0, recall=0, f2=0,
accuracy=0.7965` — the model predicts all-negative, and 0.7965 is just the negative class
proportion in the test set.

F2-based checkpointing (job 4303) avoids this because F2=0 whenever recall=0, so the
checkpoint requires actual true positives. This is why 4303 performs consistently better
than 4394 despite identical architecture and data.

The old code's final-epoch strategy happened to land past the point where the model had
learned to detect positives, and it used that model directly.

**Severity: High — fully explains 4394 failures.**

---

### 2. LR scheduler: flat 1e-4 → warmup + cosine at 3e-5

**Old:** Constant `lr=1e-4` for all 20 epochs.
**New (stable configs / job 4303):** Linear warmup from `0.01 × 3e-5 = 3e-7` over 5 epochs,
peak `3e-5`, then cosine decay to near zero.

Consequences:
- The effective LR during warmup is 10–100× smaller than the old code's LR. The model
  barely updates in the first 5 epochs.
- With `MIN_EPOCHS_BEFORE_CHECKPOINT=10` and `EARLY_STOPPING_PATIENCE=15`, the best
  checkpoint can be selected at epoch 10–15 when the model is still on the downslope of
  cosine decay.
- For D128 wcf in job 4303: best_epoch=7, total=22. Epoch 7 is just past the end of warmup
  — the model had ~2 epochs at full LR before checkpointing.

In the old code, the model trained at full LR from epoch 1 and used the epoch 20 weights.
That's ~20 epochs of effective learning vs. ~2 in the new stable config.

**Severity: High — explains why 4303 underperforms old code even with better checkpointing.**

---

### 3. Optimizer: Adam → AdamW with weight_decay=1e-4

**Old:** `optim.Adam(model.parameters(), lr=1e-4)`
**New:** `optim.AdamW(model.parameters(), lr=..., weight_decay=1e-4)`

AdamW applies true L2 weight regularization decoupled from the adaptive gradient scaling.
On ~430 training samples, `weight_decay=1e-4` adds regularization pressure that may be
excessive for a model of this capacity. Adam and AdamW have different update dynamics even
at identical nominal LR values, which changes convergence trajectories.

**Severity: Medium.**

---

### 4. Augmentation: light discrete → heavy MONAI pipeline

**Old:** Per sample: LR flip (p=0.5), Gaussian noise (p=0.3, std=0.01×data_range), rot90 in
axial plane (p=0.5). All discrete transforms, no interpolation.

**New:** Adds on top of the above:
- Flip on all 3 axes independently (3× more flip probability)
- `RandRotated ±15°` in all 3 axes (continuous rotation with trilinear interpolation)
- `RandScaleIntensityd ±10%` (p=0.5)
- `RandZoomd 90–110%` (p=0.3, with interpolation)
- `RandGaussianNoised std=0.05` on [0,1]-scaled data (5× higher relative noise than old)

Continuous rotation and zoom introduce trilinear interpolation artifacts (blurring). On small
intracranial vessels and saccular aneurysms, fine-grained structural details may be exactly
the discriminative signal — blurring these on every augmented sample may be
counterproductive. The old augmentation preserved sharpness entirely (discrete flips/rot90
are lossless).

**Severity: Medium — especially relevant given the small dataset size.**

---

### 5. Test set composition: different patients

**Old:** `torch.random_split` with `torch.Generator().manual_seed(42)`. Not stratified.
**New:** sklearn `train_test_split(stratify=labels, random_state=42)`.

Different splitting method, different patient ordering (old uses `glob.glob` ordering, new
uses `os.listdir`), and stratification vs. no stratification all mean the test sets contain
different patients even with the same seed. Direct metric comparisons between old and new
experiments are therefore not fully valid — part of the gap may reflect test set difficulty
rather than model quality.

**Severity: Confounds direct comparison but does not explain poor absolute results.**

---

### 6. Effective batch size: 5 → 8

**Old:** `batch_size=5`, no accumulation → gradient update every 5 samples.
**New:** `batch_size=4, GRAD_ACCUM_STEPS=2` → gradient update every 8 samples.

Larger effective batch size reduces gradient noise per update. On a small dataset with high
class imbalance, noisier gradients (small batch) can help the optimizer escape flat regions
associated with the majority-class collapse. This is a minor effect relative to the issues
above.

**Severity: Low.**

---

### 7. Mixed precision: float32 → bfloat16 AMP

**Old:** Full float32 throughout.
**New:** `torch.amp.autocast(dtype=torch.bfloat16)` in both train and validation passes.

bfloat16 has 7 mantissa bits vs. 23 in float32. On AMD/ROCm (the cluster), bfloat16
behavior and precision may differ from NVIDIA CUDA. Unlikely to cause large performance
differences but worth disabling as part of debugging.

**Severity: Low, but worth isolating.**

---

## Priority Summary

| # | Factor | Severity | Explains |
|---|--------|----------|---------|
| 1 | AUC checkpoint selecting zero-recall models | High | All 4394 all-negative failures |
| 2 | LR warmup + low peak LR → best epoch mid-warmup | High | 4303 underperforms old |
| 3 | Heavy MONAI augmentation (continuous rotation, zoom) | Medium | Slower/worse convergence |
| 4 | AdamW weight decay on small dataset | Medium | Extra regularization pressure |
| 5 | Different test set (stratified vs random split) | Confound | Metric incomparability |
| 6 | Larger effective batch size | Low | Minor gradient noise change |
| 7 | bfloat16 AMP on ROCm | Low | Possible precision artifacts |

---

## Next Step: Overfit Sanity Check

Before tuning hyperparameters, confirm the pipeline can learn at all by attempting to overfit
a tiny balanced subset (10–20 real samples). A working pipeline should reach near 100%
training accuracy within 200 epochs on 10 samples. If it cannot, there is a bug upstream of
the hyperparameter issues listed above.
