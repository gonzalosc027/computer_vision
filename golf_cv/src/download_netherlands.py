"""
Download Extra Training Data — PDOK Netherlands
─────────────────────────────────────────────────
Descarga orthofotos aéreas de campos de golf holandeses usando:
  • Overpass API (OpenStreetMap) — localiza campos en Países Bajos
  • PDOK WMS (Gobierno holandés, completamente gratuito, sin token)
    Resolución: 25 cm/px  |  Licencia: CC0 (dominio público)

Uso:
    python3 src/download_netherlands.py
    python3 src/download_netherlands.py --max-courses 300 --tiles-per-course 8
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
PDOK_WMS     = "https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0"
PDOK_LAYER   = "Actueel_ortho25"

# Netherlands bounding box
NL_BBOX = (50.75, 3.35, 53.55, 7.25)   # min_lat, min_lon, max_lat, max_lon

IMG_W, IMG_H  = 1600, 900
TILE_OVERLAP  = 0.25
PAUSE_S       = 0.6

OUT_DIR = RAW_DIR / "pdok_netherlands"


# ── geo helpers ───────────────────────────────────────────────────────────────

def _deg_to_m(lat: float) -> tuple[float, float]:
    m_per_lat = 111_320.0
    m_per_lon = 111_320.0 * math.cos(math.radians(lat))
    return m_per_lat, m_per_lon


def _tile_bboxes(center_lat: float, center_lon: float, n_tiles: int) -> list:
    m_lat, m_lon = _deg_to_m(center_lat)
    tile_lat = 400.0 / m_lat
    tile_lon = 700.0 / m_lon
    step_lat = tile_lat * (1 - TILE_OVERLAP)
    step_lon = tile_lon * (1 - TILE_OVERLAP)

    cols = max(1, math.ceil(math.sqrt(n_tiles * tile_lon / tile_lat)))
    rows = max(1, math.ceil(n_tiles / cols))

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


# ── OSM query ─────────────────────────────────────────────────────────────────

def fetch_golf_courses() -> list[dict]:
    min_lat, min_lon, max_lat, max_lon = NL_BBOX
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
        key = (round(float(lat), 3), round(float(lon), 3))
        if key in seen:
            continue
        seen.add(key)
        courses.append({
            "name": el.get("tags", {}).get("name", f"nl_course_{el['id']}"),
            "lat":  float(lat),
            "lon":  float(lon),
        })

    print(f"  Encontrados {len(courses)} campos en Países Bajos")
    return courses


# ── WMS download ──────────────────────────────────────────────────────────────

def download_tile(minlat, minlon, maxlat, maxlon) -> np.ndarray | None:
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS":  PDOK_LAYER,
        "STYLES":  "",
        "CRS":     "EPSG:4326",
        "BBOX":    f"{minlat},{minlon},{maxlat},{maxlon}",
        "WIDTH":   IMG_W,
        "HEIGHT":  IMG_H,
        "FORMAT":  "image/jpeg",
    }
    try:
        r = requests.get(PDOK_WMS, params=params, timeout=60)
        r.raise_for_status()
        if "image" not in r.headers.get("Content-Type", ""):
            return None
        arr = np.frombuffer(r.content, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except requests.RequestException:
        return None


def _is_valid_image(img: np.ndarray) -> bool:
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(gray.std()) > 8.0


# ── connection check ──────────────────────────────────────────────────────────

def verify_connection() -> bool:
    print("  Verificando conexión con PDOK …")
    # Center of Netherlands as test tile
    img = download_tile(52.09, 5.10, 52.10, 5.115)
    if _is_valid_image(img):
        print("  Conexión OK ✓")
        return True
    print("  ERROR: no se pudo conectar con PDOK.")
    print("  Comprueba tu conexión a internet.")
    return False


# ── main ─────────────────────────────────────────────────────────────────────

def download(max_courses: int = 300, tiles_per_course: int = 8):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(" Descarga de datos extra — PDOK Países Bajos")
    print(f"{'='*55}")
    print(f"  Destino    : {OUT_DIR}")
    print(f"  Max campos : {max_courses}")
    print(f"  Tiles/campo: {tiles_per_course}\n")

    if not verify_connection():
        sys.exit(1)

    courses = fetch_golf_courses()
    courses = courses[:max_courses]

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

        print(f"  [{i:3d}/{len(courses)}]  {name:40s}  {saved_this}/{len(tiles)} tiles")

    print(f"\n  ✓ Descargadas : {total_saved} imágenes  →  {OUT_DIR}")
    if total_failed:
        print(f"  ✗ Sin datos   : {total_failed} tiles")
    print(f"\n  Ahora ejecuta:")
    print(f"    python3 run_pipeline.py 2")
    print(f"    python3 run_pipeline.py 3\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Descarga orthofotos holandesas de campos de golf — PDOK"
    )
    parser.add_argument("--max-courses",      type=int, default=300)
    parser.add_argument("--tiles-per-course", type=int, default=8)
    args = parser.parse_args()
    download(args.max_courses, args.tiles_per_course)
