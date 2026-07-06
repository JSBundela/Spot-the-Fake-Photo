"""
One-time export: MobileNetV3-small (ImageNet-pretrained, torchvision) as a
frozen embedding extractor, saved to models/mobilenet_v3_small_embedding.onnx.
Run once, in a separate venv (see requirements.txt in this folder -- torch's
numpy version pin conflicts with the main project's). The exported .onnx
file is then loaded by train.py/predict.py via onnxruntime, which has no
such conflict and is the appropriate lightweight runtime for on-device use
(vs. bundling all of PyTorch).

Usage:
    python3 -m venv experiments/venv && source experiments/venv/bin/activate
    pip install -r experiments/requirements.txt
    python3 experiments/export_onnx.py
"""
import os
import warnings

import torch
import torchvision.models as models

warnings.filterwarnings("ignore")

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "mobilenet_v3_small_embedding.onnx")


def main():
    backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    backbone.classifier = torch.nn.Identity()  # drop the ImageNet classification head, keep pooled features
    backbone.eval()

    dummy = torch.randn(1, 3, 224, 224)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    torch.onnx.export(
        backbone, dummy, OUT_PATH,
        input_names=["input"], output_names=["embedding"], opset_version=13,
    )
    print(f"Exported to {OUT_PATH}")


if __name__ == "__main__":
    main()
