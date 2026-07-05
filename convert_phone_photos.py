"""
Converts the HEIC photos in data/REAL_DATA_captured_by_phone/{REAl,screen}
to JPEG, into data/{real,screen} -- the primary training data.
"""
import glob
import os

import pillow_heif
from PIL import Image

pillow_heif.register_heif_opener()

SRC_DIRS = {
    "real": "data/REAL_DATA_captured_by_phone/REAl",
    "screen": "data/REAL_DATA_captured_by_phone/screen",
}
OUT_ROOT = "data"


def main():
    for label, src_dir in SRC_DIRS.items():
        out_dir = os.path.join(OUT_ROOT, label)
        os.makedirs(out_dir, exist_ok=True)
        paths = sorted(glob.glob(os.path.join(src_dir, "*.HEIC")) + glob.glob(os.path.join(src_dir, "*.heic")))
        n = 0
        for p in paths:
            name = os.path.splitext(os.path.basename(p))[0]
            out_path = os.path.join(out_dir, f"{name}.jpg")
            try:
                img = Image.open(p).convert("RGB")
                img.save(out_path, quality=95)
                n += 1
            except Exception as e:
                print(f"FAIL {p}: {e}")
        print(f"{label}: converted {n} -> {out_dir}")


if __name__ == "__main__":
    main()
