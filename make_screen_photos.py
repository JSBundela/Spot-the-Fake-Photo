"""
Generates the SCREEN ("photo of a screen") class from the REAL photos.

No phone/monitor was available to physically recapture photos in this
environment, so this simulates the physical process of "displaying a photo
on a screen, then photographing that screen" by stacking the actual optical
artifacts that recapturing introduces:

  1. subpixel RGB stripe grid (LCD/OLED structure) + aliasing/decimation
     -> produces moire interference patterns
  2. reduced color gamut / contrast + color-temperature cast + banding
  3. specular glare blob(s) from the glass/glossy surface
  4. double resampling + mild blur (display's own scaling + camera focus)
  5. occasional bezel edge in frame
  6. perspective skew (camera not perfectly perpendicular to the screen)
  7. double JPEG compression (source image already compressed once; the
     recapture gets compressed again on save)

Not used to train the shipped model; reads/writes data/synthetic_proxy,
never data/real or data/screen.
"""
import glob
import io
import os
import random

import cv2
import numpy as np
from PIL import Image

random.seed(7)
np.random.seed(7)

IN_DIR = os.path.join(os.path.dirname(__file__), "data", "synthetic_proxy", "real")
OUT_DIR = os.path.join(os.path.dirname(__file__), "data", "synthetic_proxy", "screen")


def add_subpixel_grid_and_moire(img):
    h, w = img.shape[:2]
    pitch = random.uniform(2.4, 4.2)
    angle = random.uniform(-8, 8)

    xs = np.arange(w)
    ys = np.arange(h)
    xv, yv = np.meshgrid(xs, ys)
    theta = np.deg2rad(angle)
    xr = xv * np.cos(theta) + yv * np.sin(theta)

    phase = (xr % pitch) / pitch
    stripe = np.zeros((h, w, 3), dtype=np.float32)
    stripe[..., 0] = np.clip(1.0 - np.abs(phase - 0.0) * 3, 0, 1)
    stripe[..., 1] = np.clip(1.0 - np.abs(phase - 1 / 3) * 3, 0, 1)
    stripe[..., 2] = np.clip(1.0 - np.abs(phase - 2 / 3) * 3, 0, 1)
    stripe = 0.85 + 0.15 * stripe

    out = img.astype(np.float32) * stripe

    scale = random.uniform(0.4, 0.6)
    small = cv2.resize(out, (max(1, int(w * scale)), max(1, int(h * scale))),
                        interpolation=cv2.INTER_NEAREST)
    out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(out, 0, 255).astype(np.uint8)


def color_and_contrast_shift(img):
    img = img.astype(np.float32)
    cast = np.array([
        random.uniform(0.93, 1.02),
        random.uniform(0.97, 1.02),
        random.uniform(1.0, 1.08),
    ], dtype=np.float32)
    img = img * cast

    mean = img.mean()
    contrast = random.uniform(0.80, 0.93)
    img = (img - mean) * contrast + mean

    levels = random.choice([32, 48, 64])
    img = np.round(img / (256 / levels)) * (256 / levels)
    dither = np.random.uniform(-2, 2, img.shape)
    img = img + dither
    return np.clip(img, 0, 255).astype(np.uint8)


def add_glare(img):
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    n_blobs = random.choice([1, 1, 2])
    for _ in range(n_blobs):
        cx = random.uniform(0, w)
        cy = random.uniform(0, h)
        ry = random.uniform(h * 0.15, h * 0.45)
        rx = random.uniform(w * 0.15, w * 0.45)
        yy, xx = np.mgrid[0:h, 0:w]
        d2 = ((xx - cx) ** 2) / (rx ** 2) + ((yy - cy) ** 2) / (ry ** 2)
        blob = np.exp(-d2 * 2.5) * random.uniform(35, 90)
        out += blob[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def double_resample_blur(img):
    h, w = img.shape[:2]
    f = random.uniform(0.6, 0.8)
    small = cv2.resize(img, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
    up = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    k = random.choice([0, 3])
    if k > 0:
        up = cv2.GaussianBlur(up, (k, k), 0)
    return up


def add_bezel(img):
    if random.random() > 0.4:
        return img
    h, w = img.shape[:2]
    out = img.copy()
    side = random.choice(["left", "right", "top", "bottom"])
    thickness = int(min(h, w) * random.uniform(0.02, 0.06))
    color = random.choice([(10, 10, 10), (20, 20, 22), (5, 5, 5)])
    if side == "left":
        out[:, :thickness] = color
    elif side == "right":
        out[:, w - thickness:] = color
    elif side == "top":
        out[:thickness, :] = color
    else:
        out[h - thickness:, :] = color
    return out


def perspective_skew(img):
    h, w = img.shape[:2]
    d = random.uniform(0.005, 0.02)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jitter = lambda: (random.uniform(-d, d) * w, random.uniform(-d, d) * h)
    dst = np.float32([[p[0] + j[0], p[1] + j[1]] for p, j in zip(src, [jitter() for _ in range(4)])])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def double_jpeg(img, quality=None):
    quality = quality or random.randint(45, 72)
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    reloaded = Image.open(buf).convert("RGB")
    return cv2.cvtColor(np.array(reloaded), cv2.COLOR_RGB2BGR)


def simulate_screen_capture(bgr):
    img = bgr
    img = add_subpixel_grid_and_moire(img)
    img = color_and_contrast_shift(img)
    img = double_resample_blur(img)
    img = add_glare(img)
    img = perspective_skew(img)
    img = add_bezel(img)
    img = double_jpeg(img)
    return img


VARIANTS_PER_PHOTO = 2


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(IN_DIR, "*.jpg")))
    count = 0
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        for v in range(VARIANTS_PER_PHOTO):
            out_path = os.path.join(OUT_DIR, f"screen_{name}_v{v}.jpg")
            screen = simulate_screen_capture(bgr)
            cv2.imwrite(out_path, screen, [cv2.IMWRITE_JPEG_QUALITY, 90])
            count += 1
    print(f"Done. {count} screen photos written to {OUT_DIR}")


if __name__ == "__main__":
    main()
