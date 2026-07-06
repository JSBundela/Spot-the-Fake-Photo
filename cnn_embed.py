"""
576-dim embedding extraction via the frozen MobileNetV3-small ONNX model in
models/mobilenet_v3_small_embedding.onnx (see experiments/export_onnx.py).
Runs on onnxruntime, not torch -- no numpy version conflicts, and a much
lighter runtime for on-device use than bundling all of PyTorch. Preprocessing
is plain PIL + numpy (matches torchvision's Resize(256)+CenterCrop(224)+
ImageNet normalization, without needing torchvision installed).
"""
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_PATH = Path(__file__).parent / "models" / "mobilenet_v3_small_embedding.onnx"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_session = None


def _get_session():
    global _session
    if _session is None:
        _session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    return _session


def preprocess(rgb):
    """rgb: HxWx3 uint8 RGB numpy array. Returns (1, 3, 224, 224) float32."""
    img = Image.fromarray(rgb)
    w, h = img.size
    scale = 256 / min(w, h)
    img = img.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
    w, h = img.size
    left, top = (w - 224) // 2, (h - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(2, 0, 1)[None].astype(np.float32)


def embed(rgb):
    """rgb: HxWx3 uint8 RGB numpy array. Returns a (576,) float32 embedding."""
    x = preprocess(rgb)
    out = _get_session().run(None, {"input": x})[0]
    return out[0]
