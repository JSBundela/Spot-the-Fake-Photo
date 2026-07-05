# Spot the Fake Photo — report

## Approach

No deep net. `features.py` extracts ~34 classic, hand-engineered signals per image —
radial power-spectrum profile and its deviation from the natural 1/f falloff (moire
shows up as a bump above that line, the single biggest signal), FFT high-frequency
ratio/spikiness, spatial autocorrelation, color saturation/gamut/cast, glare
(highlight ratio + softness), detail loss (Laplacian variance, high-frequency noise),
bezel edges (border straight-line score), JPEG blockiness, and cross-channel
high-frequency correlation (moire decorrelates color channels). With only 114 real
training photos, the full 34-feature set overfits; ranking by importance and keeping
just the top 10 (`sat_std`, `radial_bin_9`, `hf_noise_energy`, `unique_color_ratio`,
`autocorr_peak_strength`, `laplacian_var`, `radial_bin_1`, `sat_mean`,
`border_line_score`, `radial_bin_0`) generalizes noticeably better. A small ExtraTrees
forest (400 trees, depth 6) turns those into a 0-1 score; `train.py` exports it as
plain numpy arrays (`rf_light.py`, ~20 lines) so `predict.py` never imports scikit-learn.

## Data

`data/real`, `data/screen`: 56 + 58 genuine phone photos (phone camera vs.
phone-photographing-a-laptop-screen). An earlier synthetic proxy (real photos + simulated
recaptures) scored 95% on its own synthetic test split but only ~50% on genuine photos —
it had learned the simulator's artifacts, not the general signal — so it was dropped once
real photos were available; mixing it back in measurably hurt real accuracy.

A laptop webcam photographing a phone screen (a different camera/screen combination than
any training photo) was misread as real during live testing — a real generalization gap
that more training data covering that combination would fix.

## Honest accuracy

With only 114 images, a single train/test split is noisy — the same model scores
anywhere from ~85% to ~95% depending on the random split. Averaging over 35 splits (7
seeds x 5-fold) gives a stable estimate: **93.5% ± 4.5%**. One representative split
(reported for a concrete confusion matrix): **94.7%** (108/114), ROC AUC 0.987 — 54/56
real correct, 54/58 screen correct. That's the number to trust, not a cherry-picked
single run. It's close to, but short of, the 95% bar on the honest, repeated estimate.

## Latency & cost

**~70-80ms/image** (median/mean; images resized to a 1600px longest edge before saving
-- our features already downsample to 384px internally, so nothing is lost) on this
machine's CPU (Apple-silicon laptop): feature extraction is ~50ms, the rest is JPEG
decode. **Cost:** on-device is free; cloud, assuming a 512MB AWS Lambda at ~0.08s
billed compute, is roughly **$0.80-1 per 1M images**.

## What I'd improve with more time

More real photos, especially more camera/screen combinations, and enough of them to
make a proper held-out test set meaningful — 114 images isn't. Then: a same-scene
multi-frame check (2 frames — moire shifts with tiny camera movement in a real
recapture, real textures don't) as a near-free, strong extra signal if the product can
support it.

## Keeping it accurate as cheaters adapt, and picking the cutoff

Retrain periodically on flagged/appealed cases (that's exactly the distribution of new
cheating techniques), monitor feature drift, and keep a small held-out "attack" set of
newest techniques to catch regressions before shipping a retrain. For the cutoff:
picked via Youden's J on the ROC curve today (balances false-accusation vs.
missed-cheat rate equally); in production I'd weight it by actual cost — a lower
threshold is fine if flagged cases go to cheap human review rather than an automatic
reject, a higher threshold if the action is irreversible (e.g. account ban).
