"""
Extracts frozen MobileNetV3-small (ImageNet-pretrained) embeddings for all
images in data/{real,screen}. Run in its own venv (see requirements.txt in
this folder) -- torch conflicts with the main project's numpy/opencv/sklearn
versions. Saves embeddings.npz for evaluate_embeddings.py (run separately,
in the main project's venv) to consume.

Usage:
    python3 experiments/extract_embeddings.py
"""
import glob
import os
import warnings

import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")
import torch
import torchvision.models as models
import torchvision.transforms as T

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_PATH = os.path.join(os.path.dirname(__file__), "embeddings.npz")


def main():
    backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    backbone.classifier = torch.nn.Identity()
    backbone.eval()

    preprocess = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    X, y = [], []
    for label, sub in ((0, "real"), (1, "screen")):
        paths = sorted(glob.glob(os.path.join(DATA_DIR, sub, "*.jpg")))
        for i, p in enumerate(paths):
            img = Image.open(p).convert("RGB")
            t = preprocess(img).unsqueeze(0)
            with torch.no_grad():
                feat = backbone(t)
            X.append(feat.squeeze(0).numpy())
            y.append(label)
            if i % 20 == 0:
                print(f"{sub}: {i+1}/{len(paths)}")

    X = np.array(X); y = np.array(y)
    print(f"Embedding shape: {X.shape}")
    np.savez(OUT_PATH, X=X, y=y)
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
