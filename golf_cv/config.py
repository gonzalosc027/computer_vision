from pathlib import Path

BASE_DIR = Path(__file__).parent

DATA_DIR      = BASE_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PATCHES_DIR   = DATA_DIR / "patches"
RESULTS_DIR   = DATA_DIR / "results"
MODELS_DIR    = BASE_DIR / "models"

# ── Image & patch settings ──────────────────────────────────────────────────
IMG_SIZE   = 512   # images are resized to this before patching
PATCH_SIZE = 64    # each patch fed to the CNN
STRIDE     = 32    # overlap between patches

# ── Training ────────────────────────────────────────────────────────────────
BATCH_SIZE    = 32
NUM_EPOCHS    = 25
LEARNING_RATE = 3e-4
VAL_SPLIT     = 0.2
PATIENCE      = 6   # early stopping

# ── Classes ──────────────────────────────────────────────────────────────────
CLASSES = ["healthy_grass", "stressed_grass", "bunker", "water", "other"]
NUM_CLASSES = len(CLASSES)

# BGR colors for overlay visualisation
CLASS_COLORS_BGR = {
    "healthy_grass":  (34,  139,  34),   # forest green
    "stressed_grass": (0,   165, 255),   # orange
    "bunker":         (113, 179, 210),   # sandy beige
    "water":          (196, 128,  40),   # blue
    "other":          (128, 128, 128),   # grey
}

# ── HSV auto-labelling thresholds ────────────────────────────────────────────
# OpenCV HSV: H ∈ [0,180], S ∈ [0,255], V ∈ [0,255]
# Each class is a list of (lo, hi) tuples — multiple ranges are OR-combined.
HSV_RANGES = {
    # Green hues — V_max raised to 255 so bright/overexposed healthy grass is
    # not misclassified; S_min lowered slightly to catch mowing stripe averages.
    "healthy_grass":  [((30, 30, 20),  (90, 255, 255))],
    # Yellow-orange hues, S_min=45: real stressed grass shows discolouration.
    # Lower than 70 to capture mildly yellowed fairways; mowing stripe patches
    # are now excluded by the H>38 gate in _has_mowing_stripe_pattern so they
    # won't collide with this range.
    "stressed_grass": [((15, 45, 55),  (40, 210, 230))],
    # Sandy/beige: low saturation, high brightness
    "bunker":         [((10,  5, 140), (35,  90, 255))],
    # Two ranges: clear/blue water + dark/overcast water (common in Denmark)
    "water":          [
        ((90,  40,  50), (130, 255, 255)),
        ((90,  10,  10), (130,  70, 110)),
    ],
}

# Minimum confidence (fraction of patch pixels) to assign a label
LABEL_CONFIDENCE = 0.50
