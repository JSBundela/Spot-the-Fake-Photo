# Experiment log

Detailed record of what was tried, compared, and found while building the model in
`train.py`. `report.md` has the required summary for the assignment; this file has the
full working notes behind it.

## 1. Feature selection

`features.py` computes 34 hand-crafted signals (FFT/moire, color, blur, glare, bezel,
JPEG artifacts -- see its docstrings for what each targets). With 114 training images,
using all 34 overfits; ranking by importance and keeping the top 10 generalizes better.

That ranking must be done **inside each cross-validation fold**, using only that fold's
training images -- doing it once on the full 114-image dataset and reusing the same 10
features for every fold lets test images influence which features get chosen. See
section 3 for the before/after impact of this.

## 2. Model comparison

Five model families, evaluated with 7×5-fold nested repeated cross-validation (feature
selection redone inside every fold, fixed 0.5 decision threshold, no leakage):

| Model | Accuracy |
|---|---|
| logreg | 78.7% ± 7.8% |
| svm_rbf (C=32, gamma=0.05) | 89.7% ± 4.7% |
| gboost (150 trees, depth 2) | 89.0% ± 6.9% |
| random_forest (400 trees, depth 6) | 90.0% ± 5.5% |
| **extra_trees (400 trees, depth 6)** | **92.1% ± 4.6%** |

ExtraTrees wins consistently and is what's shipped.

## 3. Data leakage: found and fixed

Three issues were found in an earlier version of `train.py`, all now fixed:

1. **Feature selection on the full dataset.** The top-10 features were chosen once
   using all 114 images, then reused for every CV fold -- test-fold images influenced
   which features were picked. **Fix:** feature ranking is now redone inside every
   fold, using only that fold's training data (`select_top_features` in `train.py`).
2. **Decision threshold tuned on the evaluation data.** The reported confusion-matrix
   accuracy used a threshold (0.60, chosen via Youden's J) fit on the exact same
   out-of-fold predictions being scored -- inflating that number. **Fix:** accuracy is
   now always reported at a fixed 0.5 threshold; the Youden's-J threshold is still used
   for the *shipped* model (a legitimate production operating-point choice, see
   report.md), but kept fully separate from the accuracy figures.
3. **An extra, unrelated seed used for the confusion matrix.** The single
   representative split used seed=42 -- not one of the 7 seeds in the repeated-CV
   sweep, chosen out of habit rather than for any reason. **Fix:** the confusion matrix
   now reuses `CV_SEEDS[0]` (seed=1), i.e. one of the same splits already in the sweep.

**Before (leaky) vs. after (fixed), same model, same data:**

| | Repeated CV | Single-split confusion matrix |
|---|---|---|
| Before | 93.5% ± 4.5% | 94.7% (108/114) |
| After | 92.1% ± 4.6% | 91.2% (104/114) |

The fixed numbers (92.1% ± 4.6% repeated CV) are what `report.md` and `model.pkl`
report as the honest accuracy.

**Remaining, unresolved leakage:** the hyperparameters themselves (400 trees, depth 6,
10 features) were chosen by informally trying different values and picking whichever
scored best on this same 114-image dataset (see section 4) -- that exploration wasn't
redone with a further layer of nested CV (i.e. hyperparameter search nested inside
feature selection nested inside the outer evaluation). Fully removing this would need
either much more data or a 3-way nested CV that's arguably overkill for 114 images in a
1-day exercise. Worth flagging rather than ignoring.

## 4. Hyperparameter tuning (informal, pre-dates the nested-CV fix)

Before the leakage fixes above, exploration across ExtraTrees settings (fit once on the
full 114 images, not nested) showed:

- **Number of top features:** 8 → 92.5%, 9 → 92.4%, **10 → 93.9%**, 11 → 93.6%,
  12 → 93.2%, 13 → 93.0%, 14 → 93.0% (repeated CV, non-nested). 10 was the most stable
  choice, not just the peak value.
- **Trees / depth:** 300 trees was similar to 400/500; depth 6 outperformed depth 5 and
  8. 400/depth 6 was picked as a reasonable middle ground, not an exhaustively swept
  optimum.

These numbers are from the pre-fix (non-nested) process, so treat them as directional
evidence for why these settings were picked, not as validated accuracy figures --
section 3's numbers are the validated ones.

## 5. A tested alternative: transfer learning (CNN embeddings)

The assignment explicitly allows non-ML approaches and doesn't require training a
model at all; it's also worth checking whether a heavier, learned approach beats the
hand-crafted features, since "any algorithm" includes that too.

**Setup:** frozen MobileNetV3-small (ImageNet-pretrained, `torchvision`), used as a
576-dim feature extractor -- no fine-tuning, since 114 images is far too little to
fine-tune a CNN without overfitting. A simple classifier on top, evaluated with the
same repeated stratified CV as above (all 576 dims used directly, so no feature
selection and no leakage from that source).

Reproduce: `experiments/extract_embeddings.py` (separate venv, see
`experiments/requirements.txt` -- torch's numpy version pin conflicts with the main
project's), then `experiments/evaluate_embeddings.py` (main project venv).

| Classifier on frozen embeddings | Accuracy |
|---|---|
| logreg | 94.7% ± 4.2% |
| svm_rbf | 86.3% ± 6.7% |
| **svm_linear** | **96.5% ± 3.3%** |
| extra_trees | 85.5% ± 7.8% |

**This beats the hand-crafted-feature model (92.1% ± 4.6%), with lower variance too.**
Latency (~70ms/image for the embedding forward pass) and raw model size (3.8MB,
float32) are actually comparable to the current approach.

**Why it wasn't shipped anyway:** the accuracy gain doesn't come for free. Running this
means bundling a deep-learning runtime (PyTorch Mobile / TFLite / Core ML -- tens of MB,
platform-specific conversion and quantization work) into the phone app, versus the
current pipeline's zero ML-framework dependency (plain numpy, trivially portable to any
language/platform, including a from-scratch native reimplementation if needed). The
assignment explicitly rewards "small, fast, cheap" -- the CNN route is a defensible
choice on accuracy alone, but a meaningfully heavier one on every other axis the brief
cares about. With more real training data, this tradeoff is worth revisiting --
transfer learning tends to pull further ahead as data grows, since it starts from
richer pretrained features than 10 hand-picked scalars ever can.
