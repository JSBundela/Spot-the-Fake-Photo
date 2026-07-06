# Spot the Fake Photo — report

## Approach

No deep net shipped (see below for why one was still tested). `features.py` extracts
~34 classic, hand-engineered signals per image — radial power-spectrum profile and its
deviation from the natural 1/f falloff (moire shows up as a bump above that line, the
single biggest signal), FFT high-frequency ratio/spikiness, spatial autocorrelation,
color saturation/gamut/cast, glare (highlight ratio + softness), detail loss (Laplacian
variance, high-frequency noise), bezel edges (border straight-line score), JPEG
blockiness, and cross-channel high-frequency correlation (moire decorrelates color
channels). With only 114 training photos, the full 34-feature set overfits; keeping
only the top 10 by importance (re-selected inside every cross-validation fold, not once
on the full dataset — see `EXPERIMENTS.md`) generalizes better. A small ExtraTrees
forest (400 trees, depth 6) turns those into a 0-1 score; `train.py` exports it as
plain numpy arrays (`rf_light.py`, ~20 lines) so `predict.py` never imports scikit-learn.

**A pretrained-CNN alternative was tested, not just assumed against:** frozen
MobileNetV3-small embeddings + a linear SVM score **96.5% ± 3.3%** on the same
evaluation — better than the hand-crafted approach. It wasn't shipped because it needs
a full deep-learning runtime bundled into the phone app (tens of MB, platform-specific
work), versus the current pipeline's zero-ML-framework footprint. Full comparison,
model/hyperparameter tuning history, and the leakage issues found and fixed along the
way are in `EXPERIMENTS.md`.

## Data

`data/real`, `data/screen`: 56 + 58 genuine phone photos (phone camera vs.
phone-photographing-a-laptop-screen). An earlier synthetic proxy (real photos +
simulated recaptures) scored 95% on its own synthetic test split but only ~50% on
genuine photos — it had learned the simulator's artifacts, not the general signal — so
it was dropped once real photos were available.

A laptop webcam photographing a phone screen (a different camera/screen combination
than any training photo) was misread as real during live testing — a real
generalization gap that more training data covering that combination would fix.

## Honest accuracy

With only 114 images, a single train/test split is noisy. Averaging over 35 splits (7
seeds × 5-fold, feature selection redone inside every fold, fixed 0.5 decision
threshold — no leakage from either): **92.1% ± 4.6%**. One specific split from that
same sweep (for a concrete confusion matrix): **91.2%** (104/114), ROC AUC 0.958 —
48/56 real correct, 56/58 screen correct. Short of the 95% bar. `EXPERIMENTS.md` has
the full before/after numbers for the leakage fixes that produced this figure.

## Latency & cost

**~70-90ms/image** (images resized to a 1600px longest edge before saving — features
already downsample to 384px internally, so nothing is lost) on this machine's CPU
(Apple-silicon laptop): feature extraction is ~50ms, the rest is JPEG decode.
**Cost:** on-device is free; cloud, assuming a 512MB AWS Lambda at ~0.09s billed
compute, is roughly **$0.90-1 per 1M images**.

## What I'd improve with more time

More real photos, especially more camera/screen combinations (the webcam-vs-phone gap
above), and enough of them to make a held-out test set meaningful — 114 isn't. Revisit
the CNN-vs-hand-crafted-features tradeoff once there's more data, since transfer
learning tends to pull further ahead as data grows. Then: a same-scene multi-frame
check (2 frames — moire shifts with tiny camera movement in a real recapture, real
textures don't) as a near-free, strong extra signal if the product can support it.

## Keeping it accurate as cheaters adapt, and picking the cutoff

Retrain periodically on flagged/appealed cases (that's exactly the distribution of new
cheating techniques), monitor feature drift, and keep a small held-out "attack" set of
newest techniques to catch regressions before shipping a retrain. For the cutoff:
`model.pkl` uses a threshold chosen via Youden's J on the full training set (balances
false-accusation vs. missed-cheat rate equally) — kept separate from the accuracy
figures above, which use a fixed 0.5 threshold so this choice can't inflate them. In
production I'd weight the cutoff by actual cost — lower if flagged cases go to cheap
human review rather than an automatic reject, higher if the action is irreversible
(e.g. account ban).
