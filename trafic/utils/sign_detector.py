import cv2
import numpy as np

# Velocidades válidas en señales europeas (km/h)
SPEED_LIMITS = frozenset({5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130})

SIGN_TYPES = {
    "blue": {
        "emoji": "🔵",
        "label": "Señal de autopista/autovía",
        "color_hex": "#4A90D9",
        "color_bgr": (200, 120, 40),
        "meaning": "Señal de dirección en autopista o autovía. "
                   "Indica destinos, distancias y numeración de vías rápidas.",
    },
    "green": {
        "emoji": "🟢",
        "label": "Señal de carretera nacional/comarcal",
        "color_hex": "#27AE60",
        "color_bgr": (50, 180, 50),
        "meaning": "Señal de dirección en carretera convencional. "
                   "Indica destinos y numeración de carreteras secundarias.",
    },
    "stop": {
        "emoji": "🛑",
        "label": "Señal STOP",
        "color_hex": "#E74C3C",
        "color_bgr": (40, 40, 220),
        "meaning": "Detención obligatoria antes de la línea de stop. "
                   "Ceder el paso a todos los vehículos que circulen por la vía.",
    },
    "traffic_light": {
        "emoji": "🚦",
        "label": "Semáforo",
        "color_hex": "#F3C800",
        "color_bgr": (0, 200, 240),
        "meaning": "Control de tráfico regulado por semáforo. "
                   "Rojo = stop · Ámbar = precaución · Verde = paso.",
    },
}

COCO_SIGN_CLASSES = {9: "traffic_light"}


class SignDetector:

    def __init__(self):
        self._reader = None
        self._sign_model = None   # modelo aislado para stop/semáforo

    @property
    def sign_model(self):
        if self._sign_model is None:
            from ultralytics import YOLO
            self._sign_model = YOLO("yolov8n.pt")
        return self._sign_model

    @property
    def reader(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(["es", "en"], gpu=False, verbose=False)
        return self._reader

    def _color_mask(self, hsv: np.ndarray, sign_type: str) -> np.ndarray:
        if sign_type == "blue":
            m = cv2.inRange(hsv, np.array([95, 80, 60]), np.array([135, 255, 255]))
        else:  # green
            m = cv2.inRange(hsv, np.array([42, 60, 50]), np.array([88, 255, 255]))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=3)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  kernel, iterations=1)
        return m

    @staticmethod
    def _is_rectangular(cnt) -> bool:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
        return len(approx) in (4, 5, 6)

    def _find_sign_regions(self, mask: np.ndarray, img_shape: tuple) -> list[tuple]:
        h, w = img_shape[:2]
        min_area = w * h * 0.008   # mínimo 0.8 % de la imagen (filtra objetos pequeños)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        seen: list[tuple[int, int]] = []

        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / max(bh, 1)

            # Señales de dirección: rectangulares y más anchas que altas
            if not (0.40 < aspect < 7.0) or bw < 60 or bh < 20:
                continue

            # Solidez: la región debe estar bien rellena (no es cielo disperso)
            solidity = area / max(bw * bh, 1)
            if solidity < 0.45:
                continue

            # Evitar duplicados cercanos
            cx_c, cy_c = x + bw // 2, y + bh // 2
            if any(abs(cx_c - sx) < 80 and abs(cy_c - sy) < 50 for sx, sy in seen):
                continue
            seen.append((cx_c, cy_c))

            pad = max(4, int(min(bw, bh) * 0.04))
            boxes.append((
                max(0, x - pad), max(0, y - pad),
                min(w, x + bw + pad), min(h, y + bh + pad),
            ))

        return boxes

    def _preprocess_for_ocr(self, crop_bgr: np.ndarray) -> np.ndarray:
        h, w = crop_bgr.shape[:2]
        # Escalar a al menos 400 px de ancho
        target_w = max(w, 400)
        scale = target_w / w
        crop = cv2.resize(crop_bgr, (target_w, int(h * scale)),
                          interpolation=cv2.INTER_LANCZOS4)

        # Extraer canal de luminancia (texto blanco = alto valor)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Ecualización adaptativa para mejorar contraste local
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        gray = clahe.apply(gray)

        # Umbral de Otsu para separar texto del fondo
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Devolver versión en escala de grises ampliada (EasyOCR la acepta)
        return gray

    def _ocr_region(self, img_bgr: np.ndarray, box: tuple) -> tuple[str, float, str]:
        x1, y1, x2, y2 = box
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0 or min(crop.shape[:2]) < 10:
            return "", 0.0, ""

        proc = self._preprocess_for_ocr(crop)

        try:
            raw = self.reader.readtext(proc, detail=1, paragraph=False,
                                       text_threshold=0.4, low_text=0.3)
            if not raw:
                return "", 0.0, ""

            confident: list[tuple[str, float]] = []   # ≥ 0.50 → mostrar directo
            possible:  list[tuple[str, float]] = []   # 0.25–0.50 → mostrar con aviso

            for r in raw:
                text, conf = r[1].strip(), float(r[2])
                clean = "".join(c for c in text if c.isalnum() or c in " .-/")
                if len(clean.strip()) < 2:
                    continue
                if conf >= 0.50:
                    confident.append((text, conf))
                elif conf >= 0.25:
                    possible.append((text, conf))

            def _dedup(pairs: list[tuple[str, float]]) -> list[tuple[str, float]]:
                seen: set[str] = set()
                out = []
                for t, c in pairs:
                    k = t.lower().replace(" ", "")
                    if k not in seen:
                        seen.add(k)
                        out.append((t, c))
                return out

            # Texto con confianza suficiente para mostrar
            unique_conf = _dedup(confident)
            if unique_conf:
                ocr_text = "\n".join(t for t, _ in unique_conf)
                avg_conf  = sum(c for _, c in unique_conf) / len(unique_conf)
            else:
                ocr_text, avg_conf = "", 0.0

            # Texto de apoyo (confianza baja pero algo hay)
            unique_all = _dedup(confident + possible)
            raw_text = "\n".join(t for t, _ in unique_all) if unique_all else ""

            return ocr_text, avg_conf, raw_text

        except Exception:
            return "", 0.0, ""

    def detect_direction_signs(self, img_bgr: np.ndarray) -> list[dict]:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        results = []

        mask_blue = self._color_mask(hsv, "blue")
        for box in self._find_sign_regions(mask_blue, img_bgr.shape):
            text, conf, raw_text = self._ocr_region(img_bgr, box)
            if not raw_text:   # sin texto = no es un cartel real
                continue
            results.append({
                "type": "blue",
                "box": {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]},
                "ocr_text": text,
                "raw_ocr_text": raw_text,
                "ocr_conf": conf,
                "reliable": conf >= 0.45,
                **SIGN_TYPES["blue"],
            })

        mask_green = self._color_mask(hsv, "green")
        mask_green = cv2.bitwise_and(mask_green, cv2.bitwise_not(mask_blue))
        for box in self._find_sign_regions(mask_green, img_bgr.shape):
            text, conf, raw_text = self._ocr_region(img_bgr, box)
            if not raw_text:   # sin texto = no es un cartel real
                continue
            results.append({
                "type": "green",
                "box": {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]},
                "ocr_text": text,
                "raw_ocr_text": raw_text,
                "ocr_conf": conf,
                "reliable": conf >= 0.45,
                **SIGN_TYPES["green"],
            })

        return results

    def detect_standard_signs(self, model, img_bgr: np.ndarray,
                               conf_thresh: float = 0.30) -> list[dict]:
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            cv2.imwrite(tmp.name, img_bgr)
            tmp_path = tmp.name
        # Umbral mínimo de 0.78 para stop/semáforo — evita falsos positivos
        sign_conf = max(conf_thresh, 0.78)
        try:
            res = self.sign_model.predict(tmp_path, conf=sign_conf, verbose=False,
                                          classes=list(COCO_SIGN_CLASSES.keys()))[0]
        finally:
            os.unlink(tmp_path)

        signs = []
        for box in res.boxes:
            cls_id = int(box.cls[0])
            sname = COCO_SIGN_CLASSES.get(cls_id)
            if not sname:
                continue
            stype = "stop" if sname == "stop" else "traffic_light"
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            signs.append({
                "type": stype,
                "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "ocr_text": "",
                "ocr_conf": float(box.conf[0]),
                "reliable": True,
                **SIGN_TYPES[stype],
            })
        return signs

    def _read_speed_number(self, img_bgr: np.ndarray, box: tuple) -> int | None:
        x1, y1, x2, y2 = box
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0 or min(crop.shape[:2]) < 10:
            return None

        # Escalar a buen tamaño (mínimo 90px alto)
        target_h = max(crop.shape[0], 90)
        scale = target_h / crop.shape[0]
        new_w = max(int(crop.shape[1] * scale), 50)
        scaled = cv2.resize(crop, (new_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)

        # EasyOCR funciona mejor con imagen a color o gris sin binarizar
        for img_ocr in [scaled, gray]:
            try:
                raw = self.reader.readtext(
                    img_ocr, detail=1, paragraph=False,
                    allowlist="0123456789",
                    text_threshold=0.35, low_text=0.2,
                )
                for r in raw:
                    text, conf = r[1].strip(), float(r[2])
                    if conf >= 0.35 and text.isdigit():
                        num = int(text)
                        if num in SPEED_LIMITS:
                            return num
            except Exception:
                continue
        return None

    def _is_ring(self, red_mask: np.ndarray, cx: int, cy: int, r: int) -> bool:
        h, w = red_mask.shape[:2]
        r_inner = int(r * 0.68)

        # Máscara exterior (corona roja)
        mask_outer = np.zeros((h, w), np.uint8)
        cv2.circle(mask_outer, (cx, cy), r,       255, -1)
        cv2.circle(mask_outer, (cx, cy), r_inner,   0, -1)
        outer_pixels = cv2.countNonZero(mask_outer)
        outer_red    = cv2.countNonZero(cv2.bitwise_and(red_mask, mask_outer))
        if outer_pixels == 0 or outer_red / outer_pixels < 0.20:
            return False   # borde no suficientemente rojo

        # Máscara interior (zona blanca)
        mask_inner = np.zeros((h, w), np.uint8)
        cv2.circle(mask_inner, (cx, cy), r_inner, 255, -1)
        inner_pixels = cv2.countNonZero(mask_inner)
        inner_red    = cv2.countNonZero(cv2.bitwise_and(red_mask, mask_inner))
        if inner_pixels == 0 or inner_red / inner_pixels > 0.18:
            return False   # interior demasiado rojo (no es blanco)

        return True

    def detect_speed_limit_signs(self, img_bgr: np.ndarray) -> list[dict]:
        h, w = img_bgr.shape[:2]
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # Máscara roja amplia (cubre condiciones de poca luz / atardecer)
        m1 = cv2.inRange(hsv, np.array([0,   50, 40]), np.array([20,  255, 255]))
        m2 = cv2.inRange(hsv, np.array([155, 50, 40]), np.array([180, 255, 255]))
        red = cv2.bitwise_or(m1, m2)

        # Limpieza morfológica ligera
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, k5, iterations=2)
        red = cv2.morphologyEx(red, cv2.MORPH_OPEN,  k5, iterations=1)

        min_side = max(30, int(min(h, w) * 0.025))
        max_side = int(min(h, w) * 0.60)

        candidates: list[tuple[int, int, int]] = []  # (cx, cy, r)

        k_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        red_filled = cv2.morphologyEx(red, cv2.MORPH_CLOSE, k_fill, iterations=5)
        red_filled = cv2.morphologyEx(red_filled, cv2.MORPH_OPEN, k5, iterations=1)

        contours, _ = cv2.findContours(red_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / max(bh, 1)
            if not (0.45 <= aspect <= 2.2):
                continue
            side = min(bw, bh)
            if side < min_side or max(bw, bh) > max_side:
                continue
            candidates.append((x + bw // 2, y + bh // 2, max(bw, bh) // 2))

        red_blur = cv2.GaussianBlur(red, (9, 9), 2)
        circles = cv2.HoughCircles(
            red_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
            param1=60, param2=12,
            minRadius=min_side // 2, maxRadius=max_side // 2,
        )
        if circles is not None:
            for cx_h, cy_h, r_h in np.round(circles[0]).astype(int):
                candidates.append((int(cx_h), int(cy_h), int(r_h)))

        deduped: list[tuple[int, int, int]] = []
        for cx, cy, r in candidates:
            if not any((cx - ex)**2 + (cy - ey)**2 < (max(r, er) * 1.3)**2
                       for ex, ey, er in deduped):
                deduped.append((cx, cy, r))

        results: list[dict] = []
        confirmed: list[tuple[int, int, int]] = []

        for cx, cy, r in deduped:
            if any((cx - cc[0])**2 + (cy - cc[1])**2 < (max(r, cc[2]) * 1.3)**2
                   for cc in confirmed):
                continue

            # Recortar interior evitando el borde rojo
            shrink = max(int(r * 0.20), 5)
            ix1 = max(0, cx - r + shrink)
            iy1 = max(0, cy - r + shrink)
            ix2 = min(w, cx + r - shrink)
            iy2 = min(h, cy + r - shrink)
            if ix2 <= ix1 or iy2 <= iy1:
                continue

            # OCR — único validador
            number = self._read_speed_number(img_bgr, (ix1, iy1, ix2, iy2))
            if number is None:
                continue

            confirmed.append((cx, cy, r))
            pad = int(r * 0.08)
            results.append({
                "type": "speed_limit",
                "speed_value": number,
                "box": {
                    "x1": max(0, cx - r - pad), "y1": max(0, cy - r - pad),
                    "x2": min(w, cx + r + pad), "y2": min(h, cy + r + pad),
                },
                "emoji": "🚫",
                "label": f"Límite de velocidad {number} km/h",
                "color_hex": "#E74C3C",
                "color_bgr": (40, 40, 220),
                "ocr_text": str(number),
                "raw_ocr_text": str(number),
                "ocr_conf": 0.9,
                "reliable": True,
                "meaning": (f"Velocidad máxima permitida: {number} km/h. "
                            "No sobrepasar este límite bajo sanción."),
            })

        # Eliminar lecturas parciales ("10" si también hay "100")
        all_nums = [r["speed_value"] for r in results]
        results = [
            r for r in results
            if not any(
                str(other) != str(r["speed_value"])
                and str(other).startswith(str(r["speed_value"]))
                for other in all_nums
            )
        ]
        return results

    def detect_regulatory_signs(self, model, img_bgr: np.ndarray,
                                 conf_thresh: float = 0.30) -> list[dict]:
        signs = []
        try:
            signs += self.detect_standard_signs(model, img_bgr, conf_thresh)
        except Exception:
            pass
        try:
            signs += self.detect_speed_limit_signs(img_bgr)
        except Exception:
            pass
        return signs

    def build_debug_images(self, img_bgr: np.ndarray) -> dict:
        h, w = img_bgr.shape[:2]
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        out = {}

        h_chan, s_chan, v_chan = cv2.split(hsv)
        out["HSV — Canal H (tono)"]  = cv2.applyColorMap(h_chan, cv2.COLORMAP_HSV)
        out["HSV — Canal S (saturación)"] = cv2.applyColorMap(s_chan, cv2.COLORMAP_BONE)
        out["HSV — Canal V (brillo)"]     = cv2.applyColorMap(v_chan, cv2.COLORMAP_BONE)

        m1 = cv2.inRange(hsv, np.array([0,   110, 70]), np.array([13,  255, 255]))
        m2 = cv2.inRange(hsv, np.array([160, 110, 70]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(m1, m2)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

        blue_mask  = self._color_mask(hsv, "blue")
        green_mask = self._color_mask(hsv, "green")
        green_mask = cv2.bitwise_and(green_mask, cv2.bitwise_not(blue_mask))

        def mask_to_bgr(mask, color_bgr):
            vis = np.zeros((h, w, 3), dtype=np.uint8)
            vis[mask > 0] = color_bgr
            return cv2.addWeighted(img_bgr, 0.45, vis, 0.55, 0)

        out["Máscara roja (señales velocidad)"]  = mask_to_bgr(red_mask,   (40,  40, 220))
        out["Máscara azul (carteles autopista)"] = mask_to_bgr(blue_mask,  (200, 80,  30))
        out["Máscara verde (carretera)"]         = mask_to_bgr(green_mask, (40, 180,  40))

        red_blur = cv2.GaussianBlur(red_mask, (7, 7), 2)
        min_r = max(15, int(min(h, w) * 0.015))
        max_r = int(min(h, w) * 0.28)
        circles_img = img_bgr.copy()
        circles = cv2.HoughCircles(red_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=70,
                                   param1=120, param2=22,
                                   minRadius=min_r, maxRadius=max_r)
        ring_img = img_bgr.copy()
        if circles is not None:
            for cx_h, cy_h, r in np.round(circles[0]).astype(int):
                cv2.circle(circles_img, (cx_h, cy_h), r, (0, 200, 255), 2)
                cv2.circle(circles_img, (cx_h, cy_h), 3, (0, 200, 255), -1)
                # Verde = pasa _is_ring, rojo = no pasa
                color = (0, 220, 60) if self._is_ring(red_mask, cx_h, cy_h, r) else (40, 40, 220)
                cv2.circle(ring_img, (cx_h, cy_h), r, color, 2)
                r_inner = int(r * 0.68)
                cv2.circle(ring_img, (cx_h, cy_h), r_inner, color, 1)

        out["Hough — círculos candidatos"] = circles_img
        out["Test corona (_is_ring) — verde=OK rojo=rechazado"] = ring_img

        contour_img = img_bgr.copy()
        for mask, clr in [(blue_mask, (200, 80, 30)), (green_mask, (40, 180, 40))]:
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(contour_img, cnts, -1, clr, 2)
            for box in self._find_sign_regions(mask, img_bgr.shape):
                x1, y1, x2, y2 = box
                cv2.rectangle(contour_img, (x1, y1), (x2, y2), clr, 3)
        out["Contornos carteles dirección"] = contour_img

        gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, 50, 150)
        out["Bordes Canny"] = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)

        return out

    def detect_all(self, model, img_bgr: np.ndarray,
                   conf_thresh: float = 0.30) -> list[dict]:
        signs = self.detect_direction_signs(img_bgr)
        signs += self.detect_regulatory_signs(model, img_bgr, conf_thresh)
        signs.sort(
            key=lambda s: (s["box"]["x2"] - s["box"]["x1"]) * (s["box"]["y2"] - s["box"]["y1"]),
            reverse=True,
        )
        return signs



def draw_signs(img_bgr: np.ndarray, signs: list[dict]) -> np.ndarray:
    out = img_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, sign in enumerate(signs, 1):
        box = sign["box"]
        x1, y1 = int(box["x1"]), int(box["y1"])
        x2, y2 = int(box["x2"]), int(box["y2"])
        color = sign.get("color_bgr", (200, 200, 0))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)

        if sign["type"] == "speed_limit":
            tag = f"#{i} {sign['speed_value']} km/h"
        else:
            tag = f"#{i} {sign['label'][:22]}"

        (tw, th), _ = cv2.getTextSize(tag, font, 0.52, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, tag, (x1 + 3, y1 - 4), font, 0.52, (0, 0, 0), 1, cv2.LINE_AA)
        if sign.get("ocr_text") and sign.get("reliable") and sign["type"] not in ("speed_limit",):
            short = sign["ocr_text"][:45]
            cv2.putText(out, short, (x1 + 4, y2 - 6), font, 0.42, color, 1, cv2.LINE_AA)
    return out
