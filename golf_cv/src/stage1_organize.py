"""
Stage 1 – Data Organisation
────────────────────────────
• Validates every image in data/raw/
• Removes perceptual duplicates
• Prints dataset statistics
• Saves train / val / test splits to data/splits.json  (70 / 15 / 15)

Run:  python src/stage1_organize.py
"""
import json
import sys
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, RAW_DIR

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ── helpers ──────────────────────────────────────────────────────────────────

def _validate(path: Path):
    try:
        img = cv2.imread(str(path))
        if img is None:
            return False, "unreadable"
        h, w = img.shape[:2]
        if min(h, w) < 100:
            return False, f"too small ({w}x{h})"
        return True, "ok"
    except Exception as e:
        return False, str(e)


def _phash(path: Path) -> str:
    try:
        return str(imagehash.phash(Image.open(path)))
    except Exception:
        return ""


# ── main ─────────────────────────────────────────────────────────────────────

def organize_dataset():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    all_files = [p for p in RAW_DIR.rglob("*") if p.suffix.lower() in IMG_EXTS]

    print(f"\n{'='*55}")
    print(" Stage 1 – Data Organisation")
    print(f"{'='*55}")
    print(f"  Found {len(all_files)} files in  {RAW_DIR}\n")

    # 1 – validate
    print("[1/4] Validating images …")
    valid, invalid = [], []
    for p in tqdm(all_files):
        ok, reason = _validate(p)
        (valid if ok else invalid).append((p, reason))

    print(f"      Valid   : {len(valid)}")
    print(f"      Invalid : {len(invalid)}")
    for p, r in invalid[:5]:
        print(f"        ✗  {p.name}  →  {r}")

    # 2 – deduplicate
    print("\n[2/4] Removing perceptual duplicates …")
    seen, unique, n_dup = {}, [], 0
    for p, _ in tqdm(valid):
        h = _phash(p)
        if h and h in seen:
            n_dup += 1
        else:
            if h:
                seen[h] = p
            unique.append(p)

    print(f"      Duplicates removed : {n_dup}")
    print(f"      Unique images      : {len(unique)}")

    # 3 – statistics
    print("\n[3/4] Computing statistics (sample of 300) …")
    sample = unique[:300]
    widths, heights = [], []
    for p in tqdm(sample):
        img = cv2.imread(str(p))
        if img is not None:
            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)
    if widths:
        print(f"      Resolution  min : {min(widths)}×{min(heights)}")
        print(f"      Resolution  max : {max(widths)}×{max(heights)}")
        print(f"      Resolution  avg : {int(np.mean(widths))}×{int(np.mean(heights))}")

    # 4 – split
    print("\n[4/4] Creating train / val / test splits …")
    np.random.seed(42)
    idx = np.random.permutation(len(unique))
    n   = len(unique)
    n_tr = int(n * 0.70)
    n_va = int(n * 0.15)

    splits = {
        "train": [str(unique[i]) for i in idx[:n_tr]],
        "val":   [str(unique[i]) for i in idx[n_tr : n_tr + n_va]],
        "test":  [str(unique[i]) for i in idx[n_tr + n_va :]],
    }

    out = DATA_DIR / "splits.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"      Train : {len(splits['train'])}")
    print(f"      Val   : {len(splits['val'])}")
    print(f"      Test  : {len(splits['test'])}")
    print(f"\n  Splits saved → {out}")
    print(f"\n✓  Stage 1 complete.  Dataset: {n} images\n")
    return splits


if __name__ == "__main__":
    organize_dataset()
