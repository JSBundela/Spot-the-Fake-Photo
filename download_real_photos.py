"""
Downloads real-world photos from picsum.photos as a synthetic-proxy REAL
class, used before genuine phone photos were available. Not used to train
the shipped model; writes to data/synthetic_proxy, not data/real.
"""
import io
import os
import sys

import requests
from PIL import Image

OUT_DIR = os.path.join(os.path.dirname(__file__), "data", "synthetic_proxy", "real")
N_IMAGES = 160
TARGET_LONG_EDGE = 1024


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    listing = []
    for page in (1, 2):
        r = requests.get(f"https://picsum.photos/v2/list?page={page}&limit=100", timeout=20)
        r.raise_for_status()
        listing.extend(r.json())

    listing = listing[:N_IMAGES]
    ok = 0
    for item in listing:
        pid = item["id"]
        out_path = os.path.join(OUT_DIR, f"real_{pid}.jpg")
        if os.path.exists(out_path):
            ok += 1
            continue
        url = f"https://picsum.photos/id/{pid}/1024"
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            w, h = img.size
            scale = TARGET_LONG_EDGE / max(w, h)
            if scale < 1:
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            img.save(out_path, quality=92)
            ok += 1
            print(f"saved {out_path}  ({img.size[0]}x{img.size[1]})")
        except Exception as e:
            print(f"skip {pid}: {e}", file=sys.stderr)

    print(f"Done. {ok} real photos in {OUT_DIR}")


if __name__ == "__main__":
    main()
