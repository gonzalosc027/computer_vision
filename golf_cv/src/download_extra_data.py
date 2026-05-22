"""
Download Extra Training Data
─────────────────────────────
Descarga orthofotos aéreas de campos de golf españoles usando:
  • Overpass API (OpenStreetMap) — localiza todos los campos en España
  • PNOA WMS (IGN España, licencia CC-BY 4.0) — imágenes aéreas 0.25 m/px

Uso:
    python src/download_extra_data.py
    python src/download_extra_data.py --max-courses 50
    python src/download_extra_data.py --max-courses 200 --tiles-per-course 6
"""

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DIR

# ── constants ────────────────────────────────────────────────────────────────

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
PNOA_WMS     = "https://www.ign.es/wms-inspire/pnoa-ma"

# Spain bounding box
SPAIN_BBOX = (36.0, -9.5, 43.8, 4.5)   # min_lat, min_lon, max_lat, max_lon

# Download settings
IMG_W, IMG_H = 1600, 900                # pixels — same as existing dataset
MIN_COURSE_M = 200                      # ignore tiny bboxes (noise in OSM)
TILE_OVERLAP  = 0.25                    # 25 % tile overlap for coverage
PAUSE_S       = 0.8                     # seconds between requests (be polite)

OUT_DIR = RAW_DIR / "pnoa_spain"


# ── geo helpers ───────────────────────────────────────────────────────────────

def _deg_to_m(lat: float) -> tuple[float, float]:
    """Approx metres per degree at given latitude."""
    m_per_lat = 111_320.0
    m_per_lon = 111_320.0 * math.cos(math.radians(lat))
    return m_per_lat, m_per_lon


def _bbox_size_m(minlat, minlon, maxlat, maxlon) -> tuple[float, float]:
    m_lat, m_lon = _deg_to_m((minlat + maxlat) / 2)
    return (maxlat - minlat) * m_lat, (maxlon - minlon) * m_lon


def _tile_bboxes(center_lat: float, center_lon: float, n_tiles: int) -> list:
    """
    Generate n_tiles bboxes around a course center.
    Each tile covers ~700 m wide × ~400 m tall (16:9 aspect, ~1 m/px equivalent).
    Tiles are arranged in a grid with 25 % overlap.
    """
    m_lat, m_lon = _deg_to_m(center_lat)
    tile_lat = 400.0 / m_lat    # ~400 m in degrees
    tile_lon = 700.0 / m_lon    # ~700 m in degrees (16:9)
    step_lat = tile_lat * (1 - TILE_OVERLAP)
    step_lon = tile_lon * (1 - TILE_OVERLAP)

    # Grid size: 1×1 → 4, 2×2 → 4, 2×3 → 6, etc.
    cols = max(1, math.ceil(math.sqrt(n_tiles * tile_lon / tile_lat)))
    rows = max(1, math.ceil(n_tiles / cols))

    # Centre the grid on the course centre
    start_lat = center_lat - (rows - 1) * step_lat / 2 - tile_lat / 2
    start_lon = center_lon - (cols - 1) * step_lon / 2 - tile_lon / 2

    tiles = []
    for r in range(rows):
        for c in range(cols):
            s_lat = start_lat + r * step_lat
            s_lon = start_lon + c * step_lon
            tiles.append((
                round(s_lat,            7),
                round(s_lon,            7),
                round(s_lat + tile_lat, 7),
                round(s_lon + tile_lon, 7),
            ))
            if len(tiles) >= n_tiles:
                break
        if len(tiles) >= n_tiles:
            break
    return tiles


# ── OSM query ────────────────────────────────────────────────────────────────

def fetch_golf_courses(max_courses: int) -> list[dict]:
    """
    Return list of {'name', 'lat', 'lon'} for golf courses in Spain.
    Uses 'out center;' which is universally supported by all Overpass versions.
    """
    min_lat, min_lon, max_lat, max_lon = SPAIN_BBOX
    query = (
        f"[out:json][timeout:90];"
        f"("
        f"way[\"leisure\"=\"golf_course\"]({min_lat},{min_lon},{max_lat},{max_lon});"
        f"relation[\"leisure\"=\"golf_course\"]({min_lat},{min_lon},{max_lat},{max_lon});"
        f");"
        f"out center;"
    )
    headers = {"User-Agent": "GolfCVResearch/1.0 (academic, golf course aerial imagery)"}
    print("  Consultando Overpass API (OpenStreetMap) …")
    try:
        r = requests.get(OVERPASS_URL, params={"data": query},
                         headers=headers, timeout=120)
        r.raise_for_status()
    except requests.RequestException as e:
        sys.exit(f"ERROR al contactar Overpass API: {e}")

    elements = r.json().get("elements", [])
    seen, courses = set(), []
    for el in elements:
        center = el.get("center") or {}
        lat = center.get("lat") or el.get("lat")
        lon = center.get("lon") or el.get("lon")
        if lat is None or lon is None:
            continue
        key = (round(lat, 3), round(lon, 3))
        if key in seen:
            continue
        seen.add(key)
        courses.append({
            "name": el.get("tags", {}).get("name", f"course_{el['id']}"),
            "lat":  float(lat),
            "lon":  float(lon),
        })

    print(f"  Encontrados {len(courses)} campos en OSM → usando los primeros {min(max_courses, len(courses))}")
    return courses[:max_courses]


# ── PNOA download ─────────────────────────────────────────────────────────────

def download_tile(minlat: float, minlon: float, maxlat: float, maxlon: float) -> np.ndarray | None:
    """Download one PNOA WMS tile and return as BGR numpy array, or None on error."""
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.1.1",
        "REQUEST": "GetMap",
        "LAYERS":  "OI.OrthoimageCoverage",
        "STYLES":  "",
        "SRS":     "EPSG:4326",
        "BBOX":    f"{minlon},{minlat},{maxlon},{maxlat}",
        "WIDTH":   IMG_W,
        "HEIGHT":  IMG_H,
        "FORMAT":  "image/jpeg",
    }
    try:
        r = requests.get(PNOA_WMS, params=params, timeout=60)
        r.raise_for_status()
        if "image" not in r.headers.get("Content-Type", ""):
            return None
        arr = np.frombuffer(r.content, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except requests.RequestException:
        return None


def _is_valid_image(img: np.ndarray) -> bool:
    """Reject blank/grey tiles (PNOA returns solid grey when no data)."""
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Solid colour tiles have near-zero std
    return float(gray.std()) > 8.0


# ── main ─────────────────────────────────────────────────────────────────────

def download(max_courses: int = 100, tiles_per_course: int = 4):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(" Descarga de datos extra — PNOA España")
    print(f"{'='*55}")
    print(f"  Destino    : {OUT_DIR}")
    print(f"  Max campos : {max_courses}")
    print(f"  Tiles/campo: {tiles_per_course}\n")

    courses = fetch_golf_courses(max_courses)
    if not courses:
        print("  No se encontraron campos. Comprueba la conexión.")
        return

    total_saved  = 0
    total_failed = 0

    for i, course in enumerate(courses, 1):
        name  = course["name"].replace("/", "_").replace(" ", "_")[:40]
        tiles = _tile_bboxes(course["lat"], course["lon"], tiles_per_course)

        saved_this = 0
        for j, (s_lat, s_lon, n_lat, n_lon) in enumerate(tiles):
            out_path = OUT_DIR / f"{name}_{j:02d}.jpg"
            if out_path.exists():
                saved_this += 1
                continue

            img = download_tile(s_lat, s_lon, n_lat, n_lon)
            if _is_valid_image(img):
                cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved_this  += 1
                total_saved += 1
            else:
                total_failed += 1

            time.sleep(PAUSE_S)

        status = f"{saved_this}/{len(tiles)} tiles"
        print(f"  [{i:3d}/{len(courses)}]  {name:40s}  {status}")

    print(f"\n  ✓ Descargadas : {total_saved} imágenes  →  {OUT_DIR}")
    if total_failed:
        print(f"  ✗ Sin datos   : {total_failed} tiles (zonas sin cobertura PNOA)")
    print(f"\n  Ahora ejecuta:")
    print(f"    python run_pipeline.py 2   # re-etiquetar con las nuevas imágenes")
    print(f"    python run_pipeline.py 3   # reentrenar\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Descarga orthofotos de campos de golf — PNOA España")
    parser.add_argument("--max-courses",      type=int, default=100,
                        help="Número máximo de campos (default: 100)")
    parser.add_argument("--tiles-per-course", type=int, default=4,
                        help="Tiles por campo (default: 4)")
    args = parser.parse_args()
    download(args.max_courses, args.tiles_per_course)
