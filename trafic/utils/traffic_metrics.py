import numpy as np

CLASS_NAMES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]

CLASS_COLORS_BGR = {
    "person":     (0,   200, 243),
    "bicycle":    (153,  85, 255),
    "car":        (217, 144,  74),
    "motorcycle": (60,  220, 160),
    "bus":        (50,  130, 240),
    "truck":      (50,  50,  220),
}

CLASS_COLORS_HEX = {
    "person":     "#F3C800",
    "bicycle":    "#FF55CC",
    "car":        "#4A90D9",
    "motorcycle": "#3CB371",
    "bus":        "#E67E22",
    "truck":      "#E74C3C",
}

VEHICLE_WEIGHTS = {
    "person":     0,
    "bicycle":    0.5,
    "car":        1.0,
    "motorcycle": 0.7,
    "bus":        3.0,
    "truck":      2.5,
}


def get_detection_stats(detections: list[dict]) -> dict:
    counts = {cls: 0 for cls in CLASS_NAMES}
    for det in detections:
        name = det.get("name", "")
        if name in counts:
            counts[name] += 1

    vehicle_classes = ["car", "truck", "bus", "motorcycle", "bicycle"]
    vehicle_count = sum(counts[c] for c in vehicle_classes)
    total_count = sum(counts.values())

    return {
        "counts": counts,
        "vehicle_count": vehicle_count,
        "total_count": total_count,
        "person_count": counts["person"],
    }


def calculate_traffic_density(detections: list[dict], img_w: int, img_h: int) -> float:
    img_area = img_w * img_h
    if img_area == 0:
        return 0.0

    vehicle_classes = {"car", "truck", "bus", "motorcycle", "bicycle"}
    weighted_sum = 0.0
    for det in detections:
        name = det.get("name", "")
        if name not in vehicle_classes:
            continue
        box = det.get("box", {})
        obj_area = (box.get("x2", 0) - box.get("x1", 0)) * (box.get("y2", 0) - box.get("y1", 0))
        weight = VEHICLE_WEIGHTS.get(name, 1.0)
        weighted_sum += weight * (obj_area / img_area)

    raw = min(weighted_sum * 500, 100)
    return round(raw, 1)


def classify_congestion(density_score: float) -> tuple[str, str]:
    if density_score < 20:
        return "Fluido", "#27AE60"
    elif density_score < 50:
        return "Moderado", "#F39C12"
    elif density_score < 75:
        return "Denso", "#E67E22"
    else:
        return "Congestionado", "#E74C3C"


def estimate_vehicles_per_km2(vehicle_count: int, fov_meters: float = 100.0) -> float:
    area_km2 = (fov_meters ** 2) / 1_000_000
    return round(vehicle_count / area_km2, 1) if area_km2 > 0 else 0.0
