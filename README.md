# Spot the Fake Photo

Submission for the "Spot the Fake Photo" assignment: given an image, score it 0 (real
photo) to 1 (photo of a screen / recapture). See [report.md](report.md) for the
approach, accuracy, latency, and cost, and [EXPERIMENTS.md](EXPERIMENTS.md) for the
full experiment log (model/hyperparameter comparison, a tested CNN alternative, and
data-leakage issues found and fixed).

**Live demo:** https://spot-fake-photo-webdemo.onrender.com (deployed on Render; may
take ~30-60s to wake up on the first request after a period of inactivity, a free-tier
limitation, not a bug).

## Quick start

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python3 predict.py path/to/image.jpg
# -> 0.0731
```

`predict.py` only needs `requirements-predict.txt` (numpy + opencv) at runtime — no
scikit-learn dependency at inference, see report.md for why.

## Files

- `data/real/`, `data/screen/` — the primary dataset: 56 + 58 genuine phone photos
  (converted from `data/REAL_DATA_captured_by_phone/`). Not included in this repository
  (see `.gitignore`); regenerate with `convert_phone_photos.py`.
- `features.py` — hand-engineered feature extraction (shared by training & inference).
- `rf_light.py` — numpy-only decision-tree-forest evaluator, used by `predict.py` so it
  doesn't need to import scikit-learn.
- `predict.py` — the one-line predictor: `python3 predict.py image.jpg`.
- `train.py` — trains the model on `data/{real,screen}`, writes `model.pkl`.
- `convert_phone_photos.py` — converts the HEIC phone photos in
  `data/REAL_DATA_captured_by_phone/{REAl,screen}` to JPEG in `data/{real,screen}`.
- `eval_phone_real.py` — in-sample check against `data/{real,screen}` (the
  out-of-fold accuracy in `model.pkl` is the honest number, see report.md).
- `data/synthetic_proxy/`, `download_real_photos.py`, `make_screen_photos.py` — an
  earlier synthetic proxy dataset, not used to train the shipped model (see report.md).
- `experiments/` — a tested transfer-learning (CNN) alternative and other experiment
  scripts; see `EXPERIMENTS.md`. Not part of the shipped model's dependencies.
- `model.pkl` — the trained model (numpy arrays only, ~670KB).
- `webdemo/` — live camera demo (Flask), deployed at the URL above. Run locally with
  `python3 webdemo/server.py`, then open `http://localhost:5050`.
- `streamlit_app.py` — a snapshot-per-click alternative demo (`st.camera_input`),
  deployable on Streamlit Community Cloud.
- `render.yaml` — Render Blueprint config for `webdemo/`.

## Retraining

```bash
python3 convert_phone_photos.py     # data/REAL_DATA_captured_by_phone -> data/{real,screen}
python3 train.py                    # writes model.pkl, prints repeated-CV + out-of-fold accuracy
python3 eval_phone_real.py          # in-sample check
```

To add more training photos: drop them into `data/REAL_DATA_captured_by_phone/{REAl,screen}`
(or directly into `data/{real,screen}` as JPEGs) and re-run `train.py`.
