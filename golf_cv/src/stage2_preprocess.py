"""
Stage 2 – Preprocessing & Patch Extraction
────────────────────────────────────────────
• Resizes every image to IMG_SIZE × IMG_SIZE
• Extracts overlapping PATCH_SIZE patches
• Auto-labels each patch via HSV colour analysis (weak supervision)
• Applies data augmentation on train patches
• Saves patches to  data/patches/{train|val}/{class}/

Run:  python src/stage2_preprocess.py
"""
import json
import sys
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CLASSES, DATA_DIR, HSV_RANGES, IMG_SIZE, LABEL_CONFIDENCE,
    PATCH_SIZE, PATCHES_DIR, STRIDE,
)

# ── augmentation pipeline (train only) ──────────────────────────────────────
AUGMENT = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.4),
    A.GaussNoise(var_limit=(5, 25), p=0.2),
    A.Blur(blur_limit=3, p=0.1),
])


# ── lighting normalisation ────────────────────────────────────────────────────

def _clahe_normalise(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ── labelling ────────────────────────────────────────────────────────────────

def _has_mowing_stripe_pattern(patch_bgr: np.ndarray) -> bool:
    """
    Detect periodic brightness bands (mowing stripes) in a patch.
    Returns True when the patch looks like healthy mowed grass rather than
    genuinely stressed or uniformly discoloured turf.
    """
    hsv   = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h_mean = hsv[:, :, 0].mean()
    s_mean = hsv[:, :, 1].mean()
    # Must be clearly green (H > 38) — excludes the yellow-green H=15..38 range
    # that corresponds to stressed/yellowing grass, so those patches are not
    # wrongly promoted to healthy_grass by this check.
    if not (38 < h_mean < 82 and s_mean > 28):
        return False
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    row_std = gray.mean(axis=1).std()
    col_std = gray.mean(axis=0).std()
    max_std = max(row_std, col_std)
    # Directional variance within a moderate range → alternating stripes
    # Too low = uniform; too high = patch contains a class boundary
    return 12.0 < max_std < 65.0


def _classify_patch(patch_bgr: np.ndarray) -> str:
    """Return the dominant class label for a patch, or 'other'."""
    # Mowing stripe patches (periodic bands, green hue) should be healthy_grass,
    # even though the bright stripes may look slightly yellow in isolation.
    if _has_mowing_stripe_pattern(patch_bgr):
        return "healthy_grass"

    hsv   = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    total = patch_bgr.shape[0] * patch_bgr.shape[1]

    # Priority: water > bunker > stressed_grass > healthy_grass
    priority = ["water", "bunker", "stressed_grass", "healthy_grass"]
    scores = {}
    for cls in priority:
        combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in HSV_RANGES[cls]:
            combined |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        scores[cls] = int(combined.sum()) // 255 / total

    best = max(scores, key=scores.get)
    return best if scores[best] >= LABEL_CONFIDENCE else "other"


# ── per-image patch extraction ───────────────────────────────────────────────

def _extract_patches(img_bgr: np.ndarray, augment: bool) -> dict:
    patches = {cls: [] for cls in CLASSES}
    h, w    = img_bgr.shape[:2]

    for y in range(0, h - PATCH_SIZE + 1, STRIDE):
        for x in range(0, w - PATCH_SIZE + 1, STRIDE):
            patch = img_bgr[y : y + PATCH_SIZE, x : x + PATCH_SIZE]
            cls   = _classify_patch(patch)
            patches[cls].append(patch.copy())
            if augment:
                for _ in range(2):
                    patches[cls].append(AUGMENT(image=patch)["image"])

    return patches


# ── main ─────────────────────────────────────────────────────────────────────

def preprocess_stage():
    splits_file = DATA_DIR / "splits.json"
    if not splits_file.exists():
        sys.exit("ERROR: splits.json not found – run Stage 1 first.")

    with open(splits_file) as f:
        splits = json.load(f)

    # create output directories
    for split in ("train", "val"):
        for cls in CLASSES:
            (PATCHES_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(" Stage 2 – Preprocessing & Patch Extraction")
    print(f"{'='*55}\n")

    counts = {sp: {c: 0 for c in CLASSES} for sp in ("train", "val")}

    for split in ("train", "val"):
        imgs    = splits[split]
        augment = split == "train"
        print(f"  Processing '{split}' ({len(imgs)} images, augment={augment}) …")

        for img_path in tqdm(imgs):
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = _clahe_normalise(img)   # normalise before labelling

            for cls, patches in _extract_patches(img, augment).items():
                cls_dir  = PATCHES_DIR / split / cls
                stem     = Path(img_path).stem
                for i, patch in enumerate(patches):
                    cv2.imwrite(str(cls_dir / f"{stem}_{i}.jpg"), patch)
                    counts[split][cls] += 1

    # summary
    print("\n  Patch counts:")
    for split in ("train", "val"):
        print(f"\n    [{split}]")
        for cls in CLASSES:
            bar = "█" * min(counts[split][cls] // 500, 30)
            print(f"      {cls:22s}: {counts[split][cls]:7d}  {bar}")

    stats = {"patch_counts": counts, "patch_size": PATCH_SIZE, "img_size": IMG_SIZE}
    with open(DATA_DIR / "preprocess_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n✓  Stage 2 complete.  Patches saved → {PATCHES_DIR}\n")


if __name__ == "__main__":
    preprocess_stage()
