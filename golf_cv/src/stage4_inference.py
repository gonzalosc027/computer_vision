"""
Stage 4 – Inference
────────────────────
• Loads the trained model
• Runs overlapping sliding-window segmentation on any image
• Computes zone coverage, health score, stress index, uniformity
• Generates: segmentation overlay, health heatmap, summary figure + JSON report

Run:  python src/stage4_inference.py --image path/to/golf.jpg
"""
import argparse
import json
import math
import sys
import warnings
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CLASS_COLORS_BGR, CLASSES, IMG_SIZE, MODELS_DIR, NUM_CLASSES,
    PATCH_SIZE, RESULTS_DIR, STRIDE,
)

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()         else
    torch.device("mps")  if torch.backends.mps.is_available() else
    torch.device("cpu")
)

TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize((PATCH_SIZE, PATCH_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Grass-only health weight per class
HEALTH_W = {"healthy_grass": 1.0, "stressed_grass": 0.25,
            "bunker": 0.0, "water": 0.0, "other": 0.0}

# Virtual index for trees (display-only, not a model class)
TREE_IDX       = len(CLASSES)   # 5
TREE_COLOR_BGR = (0, 80, 0)     # dark forest green


# ── lighting normalisation ────────────────────────────────────────────────────

def _clahe_normalise(image_bgr: np.ndarray) -> np.ndarray:
    """
    Two-channel CLAHE normalisation.
    V (brightness) is equalised strongly to handle sun/shadow exposure.
    S (saturation) is normalised mildly to reduce glare colour artefacts.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    clahe_v = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_s = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16))
    hsv[:, :, 2] = clahe_v.apply(hsv[:, :, 2])
    hsv[:, :, 1] = clahe_s.apply(hsv[:, :, 1])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ── model loading ─────────────────────────────────────────────────────────────

def load_model() -> nn.Module:
    model_path = MODELS_DIR / "best_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}\n  → Run Stage 3 first."
        )
    model = models.efficientnet_b0(weights=None)
    in_f  = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_f, NUM_CLASSES),
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    return model.to(DEVICE).eval()


# ── sliding-window inference ──────────────────────────────────────────────────

@torch.no_grad()
def sliding_window_inference(model: nn.Module, image_bgr: np.ndarray):
    """
    Returns:
        seg_map  – (H, W) uint8 with class indices, original image resolution
        prob_map – (IMG_SIZE, IMG_SIZE, NUM_CLASSES) float32
    """
    img = cv2.resize(image_bgr, (IMG_SIZE, IMG_SIZE))
    h, w = img.shape[:2]

    prob_map  = np.zeros((h, w, NUM_CLASSES), dtype=np.float32)
    count_map = np.zeros((h, w),              dtype=np.float32)

    patches, coords = [], []

    def _flush():
        if not patches:
            return
        tensor = torch.stack(patches).to(DEVICE)
        probs  = torch.softmax(model(tensor), dim=1).cpu().numpy()
        for (py, px), p in zip(coords, probs):
            prob_map [py : py + PATCH_SIZE, px : px + PATCH_SIZE] += p
            count_map[py : py + PATCH_SIZE, px : px + PATCH_SIZE] += 1
        patches.clear(); coords.clear()

    for y in range(0, h - PATCH_SIZE + 1, STRIDE):
        for x in range(0, w - PATCH_SIZE + 1, STRIDE):
            patch = img[y : y + PATCH_SIZE, x : x + PATCH_SIZE]
            patches.append(TRANSFORM(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)))
            coords.append((y, x))
            if len(patches) == 32:
                _flush()
    _flush()

    count_map = np.maximum(count_map, 1)
    prob_map /= count_map[:, :, None]
    seg_small = prob_map.argmax(axis=2).astype(np.uint8)

    seg_map = cv2.resize(
        seg_small, (image_bgr.shape[1], image_bgr.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return seg_map, prob_map


# ── adaptive zone-based grass reclassification ────────────────────────────────

def adaptive_zone_reclassify(seg_map: np.ndarray, image_bgr: np.ndarray) -> np.ndarray:
    """
    Re-evaluate grass health relative to the LOCAL illumination zone rather
    than absolute HSV thresholds.

    Algorithm:
      1. Smooth V channel with a large kernel → illumination zone map.
      2. Divide the grass area into zones by comparing each pixel's local V
         to the median grass V (sunny / neutral / shadow).
      3. Within each zone find the "reference green": the top-30 % brightest
         green pixels (H 30-90).  This is what healthy grass looks like HERE.
      4. Re-label grass pixels relative to their zone reference:
           • healthy → stressed  if H is >H_STRESS_DELTA more yellow than ref
                                  AND S deviates significantly
           • stressed → healthy  if within the normal green range of the zone
      5. A global floor ensures that if ALL grass is yellowish (truly stressed
         field), we do not flip everything back to healthy.
    """
    healthy_idx  = CLASSES.index("healthy_grass")
    stressed_idx = CLASSES.index("stressed_grass")

    refined = seg_map.copy()
    img = cv2.resize(image_bgr, (seg_map.shape[1], seg_map.shape[0]))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    grass_mask = (seg_map == healthy_idx) | (seg_map == stressed_idx)
    if grass_mask.sum() < 500:
        return refined

    # ── Step 1: illumination zone map via large Gaussian blur ────────────────
    V_smooth = cv2.GaussianBlur(V, (0, 0), sigmaX=80)

    grass_v_smooth = V_smooth[grass_mask]
    v_med = float(np.median(grass_v_smooth))
    v_std = float(grass_v_smooth.std())
    zone_thr = max(8.0, v_std * 0.35)

    sunny_zone   = grass_mask & (V_smooth > v_med + zone_thr)
    shadow_zone  = grass_mask & (V_smooth < v_med - zone_thr)
    neutral_zone = grass_mask & ~sunny_zone & ~shadow_zone

    # ── Step 2: reference green per zone ─────────────────────────────────────
    H_DELTA_STRESS = 12   # hue units more yellow than reference → stressed
    H_DELTA_OK     = 7    # within this → healthy

    # Global absolute floor: if the median grass H across the whole image is
    # already very yellow (< 38), the field may genuinely be stressed — cap
    # how aggressively we flip stressed→healthy.
    global_h_med = float(np.median(H[grass_mask]))
    aggressive_flip = global_h_med >= 38   # true = field looks green overall

    def _zone_reference(zone):
        """Return (h_ref, s_ref, v_ref) for the healthiest pixels in zone."""
        if zone.sum() < 100:
            return None
        green_in_zone = zone & (H > 30) & (H < 90)
        if green_in_zone.sum() < 50:
            green_in_zone = zone
        v_thr = np.percentile(V[green_in_zone], 70)
        ref = green_in_zone & (V >= v_thr)
        if ref.sum() < 20:
            return None
        return float(np.median(H[ref])), float(np.median(S[ref])), float(np.median(V[ref]))

    for zone_mask in (sunny_zone, shadow_zone, neutral_zone):
        if zone_mask.sum() < 100:
            continue
        ref = _zone_reference(zone_mask)
        if ref is None:
            continue
        h_ref, s_ref, v_ref = ref

        # Pixels in this zone whose H is too yellow compared to reference
        # → mark as stressed regardless of model output
        too_yellow = (
            zone_mask &
            (H < h_ref - H_DELTA_STRESS) &
            (S > 35)   # not just a washed-out reflection
        )
        refined[too_yellow] = stressed_idx

        # Pixels in this zone that ARE within the healthy green range of this
        # zone → flip stressed→healthy (only if field looks green overall)
        if aggressive_flip:
            close_to_ref = (
                zone_mask &
                (refined == stressed_idx) &
                (H >= h_ref - H_DELTA_OK) &
                (H <= h_ref + 20)
            )
            refined[close_to_ref] = healthy_idx

    return refined


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(seg_map: np.ndarray, original_bgr: np.ndarray) -> dict:
    total = seg_map.size
    zone_pct = {}
    for idx, cls in enumerate(CLASSES):
        zone_pct[cls] = round(float((seg_map == idx).sum() / total * 100), 2)
    zone_pct["trees"] = round(float((seg_map == TREE_IDX).sum() / total * 100), 2)
    # Trees already have their own index — "other" count is already correct

    grass = zone_pct["healthy_grass"] + zone_pct["stressed_grass"]
    health_score = round(
        (zone_pct["healthy_grass"] / grass * 100) if grass > 0 else 0.0, 1
    )

    stress_level = (
        "Bajo"     if health_score >= 80 else
        "Moderado" if health_score >= 55 else
        "Alto"
    )

    # Uniformity: lower std in the green channel of grass pixels → more uniform
    grass_mask = (
        (seg_map == CLASSES.index("healthy_grass")) |
        (seg_map == CLASSES.index("stressed_grass"))
    )
    if grass_mask.sum() > 200:
        g = original_bgr[:, :, 1].astype(float)
        uniformity = round(float(100 - min(np.std(g[grass_mask]) / 128 * 100, 100)), 1)
    else:
        uniformity = 0.0

    # Stress index: how much of visible grass is stressed
    stress_index = round(
        (zone_pct["stressed_grass"] / grass * 100) if grass > 0 else 0.0, 1
    )

    recommendations = []
    if zone_pct["stressed_grass"] > 20:
        recommendations.append("Hay zonas con hierba en mal estado que superan el 20% del campo. Conviene revisar el riego y el abonado en esas áreas.")
    if zone_pct["healthy_grass"] < 25:
        recommendations.append("La hierba sana cubre menos del 25% del campo. Se recomienda resembrar las zonas más dañadas.")
    if zone_pct["other"] > 20:
        recommendations.append("Una parte importante de la imagen no corresponde a hierba ni a elementos habituales del campo. Puede deberse al encuadre de la foto.")
    if uniformity < 50:
        recommendations.append("El césped presenta un aspecto irregular, con diferencias notables entre zonas. Puede ser conveniente revisar la altura y frecuencia de corte.")
    if not recommendations:
        recommendations.append("El campo está en buen estado.")

    return {
        "health_score":    health_score,
        "stress_level":    stress_level,
        "stress_index":    stress_index,
        "uniformity_score": uniformity,
        "zone_percentages": zone_pct,
        "recommendations": recommendations,
        "timestamp": datetime.now().isoformat(),
    }


# ── shape refinement ─────────────────────────────────────────────────────────

def refine_shapes(seg_map: np.ndarray) -> np.ndarray:
    """
    Post-process bunker and water regions with two rules:
      1. Shape: blobs with too few contour vertices (roads, buildings) → other.
      2. Context (bunkers only): a real bunker is always surrounded by grass.
         If less than 35 % of a bunker blob's border touches grass, it is a
         false positive (image edge, path, roof) and is removed.
    """
    MIN_AREA_PX      = 500
    MIN_VERTICES     = 6
    EPSILON_FRAC     = 0.02
    GRASS_BORDER_MIN = 0.35   # bunker border must be ≥35 % grass

    refined      = seg_map.copy()
    other_idx    = CLASSES.index("other")
    bunker_idx   = CLASSES.index("bunker")
    healthy_idx  = CLASSES.index("healthy_grass")
    stressed_idx = CLASSES.index("stressed_grass")
    grass_bin    = (
        (seg_map == healthy_idx) | (seg_map == stressed_idx)
    ).astype(np.uint8)

    for cls in ("water", "bunker"):
        cls_idx  = CLASSES.index(cls)
        mask     = (seg_map == cls_idx).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_AREA_PX:
                cv2.drawContours(refined, [contour], -1, other_idx, -1)
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            approx = cv2.approxPolyDP(contour, EPSILON_FRAC * perimeter, True)
            if len(approx) < MIN_VERTICES:
                cv2.drawContours(refined, [contour], -1, other_idx, -1)
                continue

            # Bunker-only: require grass neighbourhood
            if cls == "bunker":
                blob_mask = np.zeros(seg_map.shape, dtype=np.uint8)
                cv2.drawContours(blob_mask, [contour], -1, 1, -1)
                border = cv2.dilate(blob_mask, np.ones((15, 15), np.uint8)) - blob_mask
                border_px = border.sum()
                if border_px == 0:
                    cv2.drawContours(refined, [contour], -1, other_idx, -1)
                    continue
                grass_frac = float((grass_bin * border).sum()) / border_px
                if grass_frac < GRASS_BORDER_MIN:
                    cv2.drawContours(refined, [contour], -1, other_idx, -1)

    return refined


# ── visualisation ─────────────────────────────────────────────────────────────

def create_overlay(image_bgr: np.ndarray, seg_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    colour_layer = np.zeros_like(image_bgr)
    for idx, cls in enumerate(CLASSES):
        colour_layer[seg_map == idx] = CLASS_COLORS_BGR[cls]
    colour_layer[seg_map == TREE_IDX] = TREE_COLOR_BGR
    return cv2.addWeighted(image_bgr, 1 - alpha, colour_layer, alpha, 0)


def _detect_stripe_period(profile: np.ndarray, min_period: int = 4, max_period: int = 50) -> float:
    """Return the dominant periodic stripe period in a 1-D brightness profile, or 0."""
    valid = np.isfinite(profile)
    if valid.sum() < min_period * 3:
        return 0.0
    p = profile[valid].astype(np.float64)
    if p.std() < 3.0:
        return 0.0
    p -= p.mean()
    n = len(p)
    mag = np.abs(np.fft.rfft(p))
    mag[0] = 0.0
    freqs = np.fft.rfftfreq(n)
    band = (freqs > 1.0 / max_period) & (freqs < 1.0 / min_period)
    if not band.any():
        return 0.0
    peak = int(np.argmax(mag * band))
    total = mag[1:].sum()
    if total > 0 and mag[peak] / total > 0.22:
        return float(1.0 / freqs[peak]) if freqs[peak] > 0 else 0.0
    return 0.0


def smooth_mowing_stripes(seg_map: np.ndarray, image_bgr: np.ndarray = None) -> np.ndarray:
    """
    Four-pass correction for mowing stripe and solar artefact misclassification.

    Pass 1 – blob isolation: small stressed blobs surrounded by healthy grass
             are reclassified (original logic, relaxed thresholds).
    Pass 2 – reflection fix: stressed pixels with HIGH brightness and LOW
             saturation are solar reflections off mowing stripes → healthy.
    Pass 3 – shadow fix: dark green "other" pixels adjacent to grass are
             grass in shadow → reclassify as healthy.
    Pass 4 – FFT stripe detection: blocks whose brightness profile shows a
             clear periodic pattern (mowing stripes) are smoothed to healthy.
    """
    healthy_idx  = CLASSES.index("healthy_grass")
    stressed_idx = CLASSES.index("stressed_grass")
    other_idx    = CLASSES.index("other")
    refined      = seg_map.copy()
    stressed_bin = (seg_map == stressed_idx).astype(np.uint8)
    healthy_bin  = (seg_map == healthy_idx).astype(np.uint8)

    # ── Pass 1: small stressed blobs inside a healthy neighbourhood ───────────
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(stressed_bin)
    kernel25 = np.ones((25, 25), np.uint8)
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area > 20_000:
            continue
        component = (labels == lbl)
        border = cv2.dilate(component.astype(np.uint8), kernel25).astype(bool) & ~component
        if border.sum() == 0:
            continue
        if healthy_bin[border].sum() / border.sum() > 0.50:
            refined[component] = healthy_idx

    if image_bgr is None:
        return refined

    img_rsz = cv2.resize(image_bgr, (seg_map.shape[1], seg_map.shape[0]))
    hsv     = cv2.cvtColor(img_rsz, cv2.COLOR_BGR2HSV).astype(np.float32)
    H_ch, S_ch, V_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    grass_bin = ((refined == healthy_idx) | (refined == stressed_idx)).astype(np.uint8)
    grass_dil = cv2.dilate(grass_bin, np.ones((15, 15), np.uint8)).astype(bool)

    # ── Adaptive brightness reference from confirmed healthy grass ────────────
    healthy_px = refined == healthy_idx
    grass_v_mean = float(V_ch[healthy_px].mean()) if healthy_px.sum() > 200 else 140.0
    grass_h_mean = float(H_ch[healthy_px].mean()) if healthy_px.sum() > 200 else 55.0

    # Reflection threshold: brighter than 125 % of typical grass brightness.
    # Shadow threshold: darker than 62 % of typical grass brightness.
    # Both are anchored to THIS image's actual lighting conditions.
    reflect_v_thr = min(210, max(140, grass_v_mean * 1.25))
    shadow_v_thr  = max(40,  min(115, grass_v_mean * 0.62))

    # Hue tolerance around the confirmed grass hue (±22 OpenCV units ≈ ±44°)
    h_lo = max(18,  grass_h_mean - 22)
    h_hi = min(100, grass_h_mean + 22)

    bunker_idx = CLASSES.index("bunker")

    # ── Pass 2: solar reflections ────────────────────────────────────────────
    # Pixels brighter than reflect_v_thr with green hue next to grass are
    # sun glare off mowed turf.
    reflection = (
        (V_ch > reflect_v_thr) &
        (S_ch < 100) &
        (H_ch > h_lo) & (H_ch < h_hi) &
        ((refined == stressed_idx) | (refined == other_idx) | (refined == bunker_idx))
    )
    refined[reflection & grass_dil] = healthy_idx

    # ── Pass 3: shadowed grass mistaken for stressed / other ─────────────────
    # Dark green pixels with the same hue as detected grass but below the
    # shadow brightness threshold are grass in shadow.
    # Excludes pixels near already-detected trees (those are tree shadows).
    grass_dil30  = cv2.dilate(grass_bin, np.ones((30, 30), np.uint8)).astype(bool)
    tree_bin_now = (refined == TREE_IDX).astype(np.uint8)
    near_tree    = cv2.dilate(tree_bin_now, np.ones((22, 22), np.uint8)).astype(bool)
    shadow_green = (
        (V_ch < shadow_v_thr) &
        (H_ch > h_lo) & (H_ch < h_hi) &
        ~near_tree &
        ((refined == other_idx) | (refined == stressed_idx))
    )
    refined[shadow_green & grass_dil30] = healthy_idx

    # ── Pass 3b: large uniform shadow blobs ──────────────────────────────────
    # A connected region of "other" that is mostly dark-green and completely
    # surrounded by grass is a cloud/building shadow, not a different surface.
    shadow_bin = shadow_green.astype(np.uint8)
    n_lbl, lbl_map, lbl_stats, _ = cv2.connectedComponentsWithStats(shadow_bin)
    grass_dil50 = cv2.dilate(grass_bin, np.ones((50, 50), np.uint8)).astype(bool)
    for lbl in range(1, n_lbl):
        area = lbl_stats[lbl, cv2.CC_STAT_AREA]
        if area < 300:
            continue
        blob = (lbl_map == lbl)
        border = cv2.dilate(blob.astype(np.uint8), np.ones((12, 12), np.uint8)).astype(bool) & ~blob
        if border.sum() == 0:
            continue
        # Accept if most of the border is grass and the blob itself is in the
        # dilated grass zone (i.e., it's an island inside the fairway)
        grass_border_frac = float(grass_bin[border].sum()) / border.sum()
        if grass_border_frac > 0.55 and grass_dil50[blob].mean() > 0.85:
            refined[blob] = healthy_idx

    # ── Pass 4: FFT-based stripe pattern detection ────────────────────────────
    gray      = cv2.cvtColor(img_rsz, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grass_now = ((refined == healthy_idx) | (refined == stressed_idx)).astype(np.uint8)
    H_img, W_img = seg_map.shape
    block   = 128
    bstride = block // 2
    stripe_acc = np.zeros((H_img, W_img), dtype=np.float32)
    cnt_acc    = np.zeros((H_img, W_img), dtype=np.float32)

    for by in range(0, H_img - block + 1, bstride):
        for bx in range(0, W_img - block + 1, bstride):
            g_blk = grass_now[by:by+block, bx:bx+block]
            if g_blk.mean() < 0.5:
                continue
            br = gray[by:by+block, bx:bx+block].copy()
            br[g_blk == 0] = np.nan
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                rp = _detect_stripe_period(np.nanmean(br, axis=1))
                cp = _detect_stripe_period(np.nanmean(br, axis=0))
            if rp > 0 or cp > 0:
                stripe_acc[by:by+block, bx:bx+block] += 1.0
            cnt_acc[by:by+block, bx:bx+block] += 1.0

    cnt_acc    = np.maximum(cnt_acc, 1)
    stripe_conf = stripe_acc / cnt_acc
    stripe_zone = (stripe_conf > 0.45) & (grass_now == 1)
    refined[(refined == stressed_idx) & stripe_zone] = healthy_idx

    return refined


def reclassify_trees(seg_map: np.ndarray, image_bgr: np.ndarray) -> np.ndarray:
    """
    Multi-feature tree detection pipeline.

    Features used:
      • Color: dark green HSV range (trees are darker than grass)
      • Excess Green Index: trees are greener relative to R+B channels
      • Texture: high local brightness std-dev (crown structure)
      • Relative darkness: trees are darker than their local surroundings
      • Shape: circularity + aspect ratio filter (reject roads, fences, shadows)
      • Shadow expansion: dark green pixels adjacent to detected trees → tree
    """
    other_idx    = CLASSES.index("other")
    stressed_idx = CLASSES.index("stressed_grass")
    healthy_idx  = CLASSES.index("healthy_grass")

    # seed_candidate: classes where the model already suspects non-grass
    # expand_candidate: adds healthy_grass for adaptive expansion only
    seed_candidate   = (seg_map == other_idx) | (seg_map == stressed_idx)
    expand_candidate = seed_candidate | (seg_map == healthy_idx)

    img = cv2.resize(image_bgr, (seg_map.shape[1], seg_map.shape[0]))
    bgr = img.astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    H_ch, S_ch, V_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    B, G, R = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]

    # ── Brightness thresholds ─────────────────────────────────────────────────
    # tree_v_max   : upper V for colour-based detection (requires texture too)
    # certain_dark : V below which a greenish pixel is certainly a tree
    #               (no secondary filter needed — grass never gets this dark)
    grass_px = (seg_map == healthy_idx) | (seg_map == stressed_idx)
    if grass_px.sum() > 1000:
        grass_v_mean = float(V_ch[grass_px].mean())
        # Only adapt upward for bright images; dark images use a safe fixed value
        tree_v_max   = min(148, grass_v_mean * 0.88) if grass_v_mean > 140 else 130
    else:
        grass_v_mean = None
        tree_v_max   = 130
    certain_dark = 72   # V < 72 + green hue = tree with no further conditions

    # ── Feature 1: dark green color ──────────────────────────────────────────
    color_tree = cv2.inRange(
        hsv.astype(np.uint8),
        np.array([15, 25,  5]),
        np.array([90, 230, int(tree_v_max)]),
    ).astype(bool)

    # ── Feature 2: excess green index ────────────────────────────────────────
    # Trees viewed from above reflect more green relative to red+blue
    egi         = 2.0 * G - R - B
    excess_green = egi > 10

    # ── Feature 3: high local texture ────────────────────────────────────────
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k15  = np.ones((15, 15), np.float32) / 225
    local_std = np.sqrt(np.maximum(
        cv2.filter2D(gray ** 2, -1, k15) - cv2.filter2D(gray, -1, k15) ** 2, 0
    ))
    high_texture = local_std > 18

    # ── Feature 4: darker than local neighbourhood ───────────────────────────
    # Tree crowns are darker than the surrounding grass
    local_mean_v = cv2.blur(V_ch, (45, 45))
    darker_local = V_ch < local_mean_v * 0.83

    # ── Seed detection: strict, only from other/stressed pixels ──────────────
    seeds = seed_candidate & color_tree & excess_green & (high_texture | darker_local)

    # ── Adaptive color sampling from seeds ───────────────────────────────────
    def _range(vals, n_std=1.5, lo_clip=0.0, hi_clip=255.0):
        m, s = float(vals.mean()), max(float(vals.std()), 5.0)
        return max(lo_clip, m - n_std * s), min(hi_clip, m + n_std * s)

    if seeds.sum() > 200:
        h_lo, h_hi = _range(H_ch[seeds], hi_clip=180)
        s_lo, s_hi = _range(S_ch[seeds])
        v_lo, v_hi = _range(V_ch[seeds])
        v_hi = min(v_hi, 140)   # never let trees be brighter than grass

        adaptive_color = cv2.inRange(
            hsv.astype(np.uint8),
            np.array([h_lo, s_lo, v_lo], dtype=np.uint8),
            np.array([h_hi, s_hi, v_hi], dtype=np.uint8),
        ).astype(bool)

        # Non-grass pixels (other/stressed): adaptive_color + one feature is enough
        raw_tree_seed = seed_candidate & adaptive_color & (high_texture | color_tree | darker_local)
        # Healthy_grass pixels: require ALL three features — very conservative to avoid
        # reclassifying correctly-detected fairway grass as trees
        grass_only = (seg_map == healthy_idx)
        raw_tree_grass = grass_only & adaptive_color & color_tree & high_texture & darker_local
        raw_tree = raw_tree_seed | raw_tree_grass
    else:
        # Fallback: only seed_candidate (never touch healthy_grass in fallback)
        raw_tree = seed_candidate & (color_tree | high_texture)

    # ── Certain-dark cluster ──────────────────────────────────────────────────
    # V < certain_dark + green hue: too dark to be grass under any lighting.
    # Uses expand_candidate so isolated dark-green trees inside fairways are caught.
    certain_cluster = (
        expand_candidate &
        (V_ch < certain_dark) &
        (H_ch > 20) & (H_ch < 92) &
        excess_green
    )
    raw_tree = raw_tree | certain_cluster

    # ── Shape filter: keep only tree-like blobs ───────────────────────────────
    # Shape rules vary by blob size:
    #   Small  (<2 000 px) : must be compact AND not too elongated (single crown)
    #   Medium (<15 000 px): just not extremely elongated (row of trees OK, road not)
    #   Large  (≥15 000 px): always accept — it is a forest area
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        raw_tree.astype(np.uint8)
    )
    refined = seg_map.copy()

    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < 600:
            continue

        w_box  = stats[lbl, cv2.CC_STAT_WIDTH]
        h_box  = stats[lbl, cv2.CC_STAT_HEIGHT]
        aspect = max(w_box, h_box) / max(min(w_box, h_box), 1)

        if area < 2_000:
            if aspect > 7:
                continue
            mask = (labels == lbl).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                p = cv2.arcLength(contours[0], True)
                if p > 0 and (4 * math.pi * area / p ** 2) < 0.10:
                    continue
        elif area < 15_000:
            if aspect > 10:   # reject roads/fences, allow tree rows
                continue
        # large blobs (forest): always accept

        refined[labels == lbl] = TREE_IDX

    # ── Shadow expansion ──────────────────────────────────────────────────────
    # Dark green pixels directly adjacent to confirmed tree blobs are cast
    # shadows — they should not count as playable grass.
    tree_mask  = (refined == TREE_IDX).astype(np.uint8)
    tree_dil   = cv2.dilate(tree_mask, np.ones((18, 18), np.uint8)).astype(bool)
    shadow_px  = (
        tree_dil &
        (V_ch < 70) &
        (H_ch > 22) & (H_ch < 95) &
        ((refined == healthy_idx) | (refined == stressed_idx) | (refined == other_idx))
    )
    refined[shadow_px] = TREE_IDX

    return refined


def create_confidence_map(prob_map: np.ndarray, target_hw: tuple) -> np.ndarray:
    """Bright = model is certain, dark = model is uncertain."""
    conf = prob_map.max(axis=2)
    conf_resized = cv2.resize(conf, (target_hw[1], target_hw[0]))
    return cv2.applyColorMap((conf_resized * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)


def create_heatmap(prob_map: np.ndarray, target_hw: tuple) -> np.ndarray:
    """Green = healthy, Red = stressed.  target_hw = (H, W)."""
    h_prob = cv2.resize(prob_map[:, :, CLASSES.index("healthy_grass")],
                        (target_hw[1], target_hw[0]))
    s_prob = cv2.resize(prob_map[:, :, CLASSES.index("stressed_grass")],
                        (target_hw[1], target_hw[0]))
    health = ((h_prob - s_prob + 1) / 2 * 255).astype(np.uint8)
    # COLORMAP_RdYlGn doesn't exist in OpenCV; build red->yellow->green LUT
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(256):
        if i < 128:
            lut[i, 0] = [0, i * 2, 255]          # red → yellow (BGR)
        else:
            lut[i, 0] = [0, 255, 255 - (i - 128) * 2]  # yellow → green (BGR)
    return cv2.LUT(cv2.merge([health, health, health]), lut)


def save_results(image_bgr, seg_map, prob_map, metrics, out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = image_bgr.shape[:2]
    overlay = create_overlay(image_bgr, seg_map)
    heatmap = create_heatmap(prob_map, (h, w))

    cv2.imwrite(str(out_dir / f"{stem}_original.jpg"),     image_bgr)
    cv2.imwrite(str(out_dir / f"{stem}_segmentation.jpg"), overlay)
    cv2.imwrite(str(out_dir / f"{stem}_heatmap.jpg"),      heatmap)

    with open(out_dir / f"{stem}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # ── summary figure ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Golf Course Analysis  —  Health Score: {metrics['health_score']:.0f}/100  "
        f"| Stress: {metrics['stress_level']}",
        fontsize=13, fontweight="bold",
    )
    for ax, img, title in zip(axes,
                               [image_bgr, overlay, heatmap],
                               ["Original", "Zone Segmentation", "Health Heatmap"]):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=11); ax.axis("off")

    handles = [
        mpatches.Patch(color=np.array(v[::-1]) / 255,
                       label=k.replace("_", " ").title())
        for k, v in CLASS_COLORS_BGR.items()
    ]
    axes[1].legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.7)

    plt.tight_layout()
    report_path = out_dir / f"{stem}_report.png"
    plt.savefig(report_path, dpi=150, bbox_inches="tight")
    plt.close()

    # ── console summary ───────────────────────────────────────────────────────
    print(f"\n  ┌─ Results: {stem} ──────────────────────────")
    print(f"  │  Health Score  : {metrics['health_score']}/100")
    print(f"  │  Stress Level  : {metrics['stress_level']}")
    print(f"  │  Stress Index  : {metrics['stress_index']}%")
    print(f"  │  Uniformity    : {metrics['uniformity_score']}/100")
    print(f"  │  Zone coverage :")
    for cls, pct in metrics["zone_percentages"].items():
        bar = "█" * int(pct / 2)
        print(f"  │    {cls:22s}: {pct:5.1f}%  {bar}")
    print(f"  │  Recommendations:")
    for r in metrics["recommendations"]:
        print(f"  │    • {r}")
    print(f"  └────────────────────────────────────────────")

    return report_path


# ── public entry point (also used by Stage 5) ─────────────────────────────────

def run_inference(image_path: str, out_dir: Path = None) -> dict:
    if out_dir is None:
        out_dir = RESULTS_DIR

    print(f"\n{'='*55}")
    print(" Stage 4 – Inference")
    print(f"{'='*55}")
    print(f"  Input  : {image_path}")
    print(f"  Device : {DEVICE}\n")

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")

    model = load_model()
    print("  Normalising lighting (CLAHE) …")
    image_bgr = _clahe_normalise(image_bgr)

    print("  Running sliding-window inference …")
    seg_map, prob_map = sliding_window_inference(model, image_bgr)

    print("  Refining shapes …")
    seg_map = refine_shapes(seg_map)

    water_idx = CLASSES.index("water")
    other_idx = CLASSES.index("other")
    seg_map[seg_map == water_idx] = other_idx

    # Stripe/shadow correction first so the model's grass coverage is solid,
    # then tree detection runs on the cleaned segmentation.
    seg_map = smooth_mowing_stripes(seg_map, image_bgr)
    seg_map = reclassify_trees(seg_map, image_bgr)
    seg_map = adaptive_zone_reclassify(seg_map, image_bgr)
    metrics = compute_metrics(seg_map, image_bgr)
    save_results(image_bgr, seg_map, prob_map, metrics, out_dir, Path(image_path).stem)

    print(f"\n✓  Stage 4 complete.  Results → {out_dir}\n")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Golf Course Inference")
    parser.add_argument("--image",  required=True, help="Path to input image")
    parser.add_argument("--output", default=None,  help="Output directory (optional)")
    args = parser.parse_args()
    run_inference(args.image, Path(args.output) if args.output else None)
