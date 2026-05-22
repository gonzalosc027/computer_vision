from collections import deque
import numpy as np


VEHICLE_REAL_HEIGHT_M: dict[str, float] = {
    "car": 1.5, "truck": 3.5, "bus": 3.2,
    "motorcycle": 1.2, "bicycle": 1.1, "person": 1.7,
}

# Vehículo debe ocupar al menos este % de la altura del frame para reportar velocidad
MIN_BOX_HEIGHT_FRAC = 0.05   # 5 % del alto del frame
MIN_PIXEL_DISPLACEMENT = 12  # píxeles mínimos de movimiento total para no reportar ruido


class VehicleTracker:

    def __init__(self, fps: float, scale_m_per_px: float = 0.10):
        self.fps = max(fps, 1.0)
        self.scale = scale_m_per_px          # escala global de respaldo
        self.positions: dict[int, deque] = {}  # id → deque[(frame, cx, cy)]
        self.scales:    dict[int, deque] = {}  # id → deque[m_per_px] por frame
        self.speeds:    dict[int, float] = {}  # id → km/h
        self._smoothed: dict[int, list]  = {}  # para suavizado de señal

    def update(self, track_id: int, cx: float, cy: float, frame_idx: int,
               scale_override: float | None = None) -> float | None:
        if track_id not in self.positions:
            self.positions[track_id] = deque(maxlen=12)
            self.scales[track_id]    = deque(maxlen=12)
        self.positions[track_id].append((frame_idx, cx, cy))
        self.scales[track_id].append(scale_override if scale_override is not None else self.scale)
        return self._estimate(track_id)

    def _estimate(self, track_id: int) -> float | None:
        pos = self.positions[track_id]
        if len(pos) < 4:
            return None
        f1, x1, y1 = pos[0]
        f2, x2, y2 = pos[-1]
        dt = (f2 - f1) / self.fps
        if dt < 0.12:
            return None

        pixel_dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if pixel_dist < MIN_PIXEL_DISPLACEMENT:
            return None

        scales = list(self.scales[track_id])
        scales.sort()
        effective_scale = scales[len(scales) // 2]

        dist_m = pixel_dist * effective_scale
        raw_kmh = (dist_m / dt) * 3.6

        buf = self._smoothed.setdefault(track_id, [])
        buf.append(raw_kmh)
        if len(buf) > 5:
            buf.pop(0)
        smoothed = sum(buf) / len(buf)
        self.speeds[track_id] = round(smoothed, 1)
        return self.speeds[track_id]

    def get_speed(self, track_id: int) -> float | None:
        return self.speeds.get(track_id)

    def reset(self) -> None:
        self.positions.clear()
        self.speeds.clear()
        self._smoothed.clear()


class CountingLine:

    def __init__(self, y_fraction: float = 0.5):
        self.y_fraction = y_fraction
        self.y_px: int = 0
        self.count_down: int = 0   # arriba → abajo
        self.count_up: int = 0     # abajo → arriba
        self._last_y: dict[int, float] = {}
        self._crossed: set[int] = set()

    def set_frame_height(self, h: int) -> None:
        self.y_px = int(h * self.y_fraction)

    def check(self, track_id: int, cy: float) -> str | None:
        direction = None
        if track_id in self._last_y and track_id not in self._crossed:
            prev = self._last_y[track_id]
            if prev < self.y_px <= cy:
                self.count_down += 1
                self._crossed.add(track_id)
                direction = "down"
            elif prev > self.y_px >= cy:
                self.count_up += 1
                self._crossed.add(track_id)
                direction = "up"
        self._last_y[track_id] = cy
        return direction

    @property
    def total(self) -> int:
        return self.count_down + self.count_up

    def reset(self) -> None:
        self.count_down = 0
        self.count_up = 0
        self._last_y.clear()
        self._crossed.clear()
