# Spot the Fake Photo — report

## Approach

**One model is shipped**: a linear SVM on top of frozen MobileNetV3-small
(ImageNet-pretrained) embeddings — `cnn_embed.py` extracts a 576-dim embedding via
`onnxruntime` running `models/mobilenet_v3_small_embedding.onnx` (no fine-tuning; 114
images is far too little to fine-tune a CNN without overfitting), and `train.py` fits a
linear SVM on top. `predict.py` exports the SVM as a plain numpy dot product
(`svm_light.py`), so inference needs only numpy + Pillow + onnxruntime — no
scikit-learn, no torch.

Several other approaches were tried and compared before landing here — a hand-crafted
frequency/color/texture feature set (`train_handcrafted_features.py`, 92.1% ± 4.6%
honest accuracy) and other classifiers on the CNN embeddings (logistic regression, RBF
SVM, ExtraTrees). Only the winner is shipped; the comparison exists to justify the
choice, not to ship a model zoo. Full comparison, hyperparameter tuning history, and
the leakage issues found and fixed along the way are in `EXPERIMENTS.md`.

## Honest accuracy

Repeated 7×5-fold cross-validation (all 576 embedding dimensions used directly, no
feature selection, so no leakage from that source; each model's own standard decision
rule, no threshold tuned on the evaluation data): **96.5% ± 3.3%**. One specific split
from that same sweep (for a concrete confusion matrix): **95.6%** (109/114), ROC AUC
0.998 — 53/56 real correct, 56/58 screen correct. Clears the 95% bar on the repeated,
leak-free estimate.

## Latency & cost

**~80-85ms/image** warm (in a long-running process — the realistic serving scenario)
on this machine's CPU (Intel Core i3-1000NG4 @ 1.1GHz laptop): the CNN forward pass is the bulk of it,
image decode is a few ms. A fresh `python3 predict.py image.jpg` process (importing
onnxruntime + loading the model fresh) takes **~500-570ms** — that's a one-time
process-startup cost, not a per-image one; a server or app keeping the process warm
pays it once. **Cost:** on-device is free; cloud, assuming a 512MB AWS Lambda at
~0.1s billed compute, is roughly **$1-1.2 per 1M images**.

**Deployed size**: `models/mobilenet_v3_small_embedding.onnx` (3.7MB, float32) +
`model.pkl` (14KB, just the SVM coefficients). Not quantized — int8 quantization
would likely cut the backbone to under 1MB with little accuracy loss, and is the
first thing to do if size becomes a real constraint on a target phone.

## What I'd improve with more time

Quantize the ONNX backbone for a smaller/faster on-device footprint. More real
photos — especially more camera/screen combinations (a laptop webcam photographing a
phone screen was misread as real during live testing, a gap the current 114 photos
don't cover) — and enough of them to make a held-out test set (rather than
cross-validation on 114 images) meaningful. A same-scene multi-frame check (2 frames —
moire shifts with tiny camera movement in a real recapture, real textures don't) as a
near-free, strong extra signal if the product can support it.

## Keeping it accurate as cheaters adapt, and picking the cutoff

Retrain periodically on flagged/appealed cases (that's exactly the distribution of new
cheating techniques), monitor for drift, and keep a small held-out "attack" set of
newest techniques to catch regressions before shipping a retrain. For the cutoff:
`model.pkl`'s threshold corresponds to the SVM's own native decision boundary,
expressed in calibrated probability space — not a value tuned to maximize the reported
accuracy. In production I'd weight it by actual cost — lower if flagged cases go to
cheap human review rather than an automatic reject, higher if the action is
irreversible (e.g. account ban).
