# Experiment log

Detailed record of what was tried and compared while building this. `report.md` has
the required summary for the assignment; this file has the full working notes behind
it, including why only one of these approaches is actually shipped.

## What's shipped, and why this file has so many numbers in it

**Only one model ships**: linear SVM on frozen MobileNetV3-small embeddings
(`train.py` / `predict.py`). Everything else below — the hand-crafted-feature
pipeline, other classifiers on the CNN embeddings, various hyperparameter settings —
was tried and compared to justify that choice, not shipped alongside it. The brief
explicitly allows (and rewards) figuring out the right approach through comparison;
this file is that comparison, kept separate from `report.md` so the report itself
stays focused on what was actually built.

## 1. Two approaches tried

**A. Hand-crafted features** (`train_handcrafted_features.py`, `features.py`): ~34
classic signals (FFT/moire, color, blur, glare, bezel, JPEG artifacts), reduced to the
10 most important, feeding a small ExtraTrees forest. Honest accuracy: **92.1% ± 4.6%**
(see section 3 for how this number was arrived at cleanly).

**B. Transfer learning** (`train.py`, shipped): frozen MobileNetV3-small
(ImageNet-pretrained, no fine-tuning — 114 images is too little to fine-tune without
overfitting) as a 576-dim feature extractor, linear SVM on top. Honest accuracy:
**96.5% ± 3.3%**.

B beat A, with lower variance, and actually clears the assignment's 95% bar while A
doesn't — so B is what's shipped. The tradeoff: B needs a CNN runtime (`onnxruntime`,
~3.7MB model + runtime library) rather than A's pure-numpy footprint (~14KB model,
no ML-framework dependency at all). Given the assignment's explicit 95%+ target and
that `onnxruntime` (not full PyTorch) is a genuinely lightweight, standard choice for
on-device inference, the accuracy win was judged worth that dependency.

## 2. Model comparison within each approach

**Hand-crafted features** (7×5-fold nested CV, feature selection redone inside every
fold, standard 0.5 threshold):

| Model | Accuracy |
|---|---|
| logreg | 78.7% ± 7.8% |
| svm_rbf (C=32, gamma=0.05) | 89.7% ± 4.7% |
| gboost (150 trees, depth 2) | 89.0% ± 6.9% |
| random_forest (400 trees, depth 6) | 90.0% ± 5.5% |
| extra_trees (400 trees, depth 6) | **92.1% ± 4.6%** |

**CNN embeddings** (7×5-fold repeated CV, no feature selection needed — all 576 dims
used directly, standard decision rule per model):

| Classifier on frozen embeddings | Accuracy |
|---|---|
| logreg | 94.7% ± 4.2% |
| svm_rbf | 86.3% ± 6.7% |
| svm_linear | **96.5% ± 3.3%** |
| extra_trees | 85.5% ± 7.8% |

Reproduce the CNN embeddings: `experiments/export_onnx.py` (one-time, separate venv —
see `experiments/requirements.txt`, torch's numpy pin conflicts with the main
project's) produces `models/mobilenet_v3_small_embedding.onnx`, then `train.py` uses
it directly via `onnxruntime` (no conflict there).

## 3. Data leakage in the hand-crafted pipeline: found and fixed

Three issues were found in an earlier version, all now fixed in
`train_handcrafted_features.py`:

1. **Feature selection on the full dataset.** The top-10 features were chosen once
   using all 114 images, then reused for every CV fold. **Fix:** feature ranking is
   now redone inside every fold, using only that fold's training data.
2. **Decision threshold tuned on the evaluation data.** The reported confusion-matrix
   accuracy used a threshold chosen via Youden's J on the exact predictions being
   scored. **Fix:** accuracy is now reported at a fixed 0.5 threshold; a Youden's-J
   threshold is still used for that model's own shipped-equivalent artifact, kept
   separate from the accuracy figures.
3. **An extra, unrelated seed used for the confusion matrix.** The single
   representative split used seed=42, not one of the 7 seeds in the repeated-CV sweep.
   **Fix:** it now reuses one of the same 7 seeds.

Before (leaky) vs. after (fixed), same model, same data:

| | Repeated CV | Single-split confusion matrix |
|---|---|---|
| Before | 93.5% ± 4.5% | 94.7% (108/114) |
| After | 92.1% ± 4.6% | 91.2% (104/114) |

The CNN-embedding comparison in section 2 doesn't have the feature-selection version
of this problem (no feature selection is done — all 576 dims are used), and uses the
same fixed-threshold discipline from the start.

## 4. Hyperparameter tuning (hand-crafted approach, informal, pre-dates the leakage fix)

Before the fixes above, exploration across ExtraTrees settings (fit once on the full
114 images, not nested) showed:

- **Number of top features:** 8 → 92.5%, 9 → 92.4%, **10 → 93.9%**, 11 → 93.6%,
  12 → 93.2%, 13 → 93.0%, 14 → 93.0% (repeated CV, non-nested). 10 was the most stable
  choice, not just the peak value.
- **Trees / depth:** 300 trees was similar to 400/500; depth 6 outperformed depth 5
  and 8. 400/depth 6 was picked as a reasonable middle ground, not an exhaustively
  swept optimum.

These numbers are from the pre-fix (non-nested) process — directional evidence for why
these settings were picked, not validated accuracy figures (section 3 has those). The
CNN embeddings didn't need this kind of tuning since there's no feature subset to
choose and the backbone is frozen/pretrained, not trained on this data.
