# Spot the Fake Photo

Submission for the "Spot the Fake Photo" assignment: given an image, score it 0 (real
photo) to 1 (photo of a screen / recapture). See [report.md](report.md) for the
approach, accuracy, latency, and cost, and [EXPERIMENTS.md](EXPERIMENTS.md) for the
full experiment log (why this approach was chosen over the alternatives tried, model
comparison, and data-leakage issues found and fixed along the way).

**Live demo:** https://spot-fake-photo-webdemo.onrender.com (deployed on Render; may
take ~30-60s to wake up on the first request after a period of inactivity, a free-tier
limitation, not a bug).

## Quick start

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python3 predict.py path/to/image.jpg
# -> 0.0064
```

`predict.py` only needs `requirements-predict.txt` (numpy, Pillow, onnxruntime) at
runtime — no scikit-learn or torch dependency at inference, see EXPERIMENTS.md for why.

## Files

**Shipped model** (linear SVM on frozen CNN embeddings — see report.md/EXPERIMENTS.md
for why this was chosen over the hand-crafted-feature alternative below):
- `predict.py` — the one-line predictor: `python3 predict.py image.jpg`.
- `train.py` — trains the model on `data/{real,screen}`, writes `model.pkl`.
- `cnn_embed.py` — 576-dim embedding extraction via `onnxruntime` (not torch — no
  numpy version conflicts, and a lighter runtime for on-device use).
- `svm_light.py` — numpy-only linear SVM decision function, used by `predict.py` so it
  doesn't need scikit-learn.
- `models/mobilenet_v3_small_embedding.onnx` — the frozen, pretrained backbone (3.7MB).
- `model.pkl` — the trained SVM (numpy arrays only, ~14KB).

**Data:**
- `data/real/`, `data/screen/` — the primary dataset: 56 + 58 genuine phone photos
  (converted from `data/REAL_DATA_captured_by_phone/`). Not included in this repository
  (see `.gitignore`, personal photos); regenerate with `convert_phone_photos.py`.
- `convert_phone_photos.py` — converts the HEIC phone photos in
  `data/REAL_DATA_captured_by_phone/{REAl,screen}` to JPEG in `data/{real,screen}`.
- `eval_phone_real.py` — in-sample check against `data/{real,screen}` (the
  out-of-fold accuracy in `model.pkl` is the honest number, see report.md).

**Tested alternatives, not shipped, kept for reference and comparison (see EXPERIMENTS.md):**
- `train_handcrafted_features.py`, `features.py`, `rf_light.py` — the hand-crafted
  ~34-feature + ExtraTrees approach tried first (92.1% ± 4.6%), superseded by the CNN
  approach (96.5% ± 3.3%). Writes to `model_handcrafted.pkl`, not `model.pkl`.
- `data/synthetic_proxy/`, `download_real_photos.py`, `make_screen_photos.py` — an
  even earlier synthetic proxy dataset, from before genuine phone photos existed.
- `experiments/export_onnx.py` — one-time script (separate venv, see
  `experiments/requirements.txt`) that produced `models/mobilenet_v3_small_embedding.onnx`
  from torchvision's pretrained weights.

**Demos:**
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

To regenerate `models/mobilenet_v3_small_embedding.onnx` (not normally needed —
it's a frozen, pretrained backbone with no dependency on this project's training
data): see `experiments/export_onnx.py`.
