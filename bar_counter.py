import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

BLUR = (5, 5)
SQUARE_BBOX = (0.7, 1.45)
RING = (1.05, 1.4)
INNER = 0.88


@dataclass
class Params:
    min_disc_area: int = 800
    min_circularity: float = 0.70
    max_circularity: float = 1.05
    # Brightness checks are ratios, (bright - dark) / bright, so they hold up
    # whether the scene is sunlit, overcast or indoors. min_gap is only a
    # floor so sensor noise in near-black frames can't pass as contrast.
    min_contrast: float = 0.35
    min_gap: int = 12
    # Used only on the disc interior after it's stretched to 0-255, so it's
    # already independent of the overall lighting.
    core_gap: int = 50
    min_ink_fraction: float = 0.02
    min_bar_area_fraction: float = 0.004
    bar_aspect_min: float = 1.2
    max_bar_width_frac: float = 0.22
    bar_width_ratio: float = 0.23
    ink_tolerance: float = 0.55
    max_lock_drift: float = 3.0


@dataclass
class Disc:
    cx: float
    cy: float
    r: float
    contour: np.ndarray
    ok: bool


@dataclass
class Reading:
    label: Optional[str]
    confidence: float
    bbox: Optional[Tuple[float, float, float, float]]
    note: str
    center: Optional[Tuple[float, float, float]] = None


_CLAHE = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))


def prepare(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = _CLAHE.apply(gray)
    gray = cv2.GaussianBlur(gray, BLUR, 0)
    b, g, r = cv2.split(bgr)
    value = cv2.GaussianBlur(cv2.max(cv2.max(b, g), r), BLUR, 0)
    return gray, value


class BarCounter:
    def __init__(self, params: Params = None):
        self.p = params or Params()

    def contrasts(self, bright, dark):
        bright, dark = float(bright), float(dark)
        return bright - dark >= max(self.p.min_gap, self.p.min_contrast * bright)

    def dark(self, gray):
        thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return (gray < thr).astype(np.uint8)

    def discs(self, gray, value):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(self.dark(gray))
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < self.p.min_disc_area:
                continue
            core = self.black_core(gray[y:y + h, x:x + w], labels[y:y + h, x:x + w] == i)
            contours, _ = cv2.findContours(core, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE, offset=(int(x), int(y)))
            for c in contours:
                disc = self.judge(c, value)
                if disc is not None:
                    yield disc

    def black_core(self, gray, blob):
        vals = gray[blob]
        t, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        lo, hi = vals[vals <= t], vals[vals > t]
        if lo.size and hi.size and self.contrasts(hi.mean(), lo.mean()):
            blob = blob & (gray <= t)
        return blob.astype(np.uint8)

    def judge(self, contour, value):
        bx, by, w, h = cv2.boundingRect(contour)
        filled = np.zeros((h, w), np.uint8)
        cv2.drawContours(filled, [contour], -1, 1, -1, offset=(-bx, -by))
        m = cv2.moments(filled, binaryImage=True)
        if m["m00"] < self.p.min_disc_area:
            return None
        cx, cy = bx + m["m10"] / m["m00"], by + m["m01"] / m["m00"]
        r = max(w, h) / 2.0
        circularity = m["m00"] / (math.pi * r * r)
        ok = (SQUARE_BBOX[0] < w / h < SQUARE_BBOX[1]
              and self.p.min_circularity <= circularity <= self.p.max_circularity
              and self.stands_out(value, value[by:by + h, bx:bx + w][filled > 0], cx, cy, r))
        return Disc(cx, cy, r, contour, ok)

    def stands_out(self, value, inside, cx, cy, r):
        R = int(r * RING[1])
        x0, y0 = max(0, int(cx) - R), max(0, int(cy) - R)
        window = value[y0:int(cy) + R + 1, x0:int(cx) + R + 1]
        ring = np.zeros(window.shape, np.uint8)
        center = (int(cx) - x0, int(cy) - y0)
        cv2.circle(ring, center, R, 1, -1)
        cv2.circle(ring, center, int(r * RING[0]), 0, -1)
        around = window[ring > 0]
        return around.size > 0 and self.contrasts(np.median(around), np.median(inside))

    def nearest_disc(self, gray, value, prev_center=None):
        H, W = gray.shape
        ok = [d for d in self.discs(gray, value) if d.ok]
        if not ok:
            return None
        if prev_center is not None:
            pcx, pcy, pr = prev_center
            within = [d for d in ok
                      if math.hypot(d.cx - pcx, d.cy - pcy) <= self.p.max_lock_drift * max(pr, 1.0)]
            if within:
                return min(within, key=lambda d: math.hypot(d.cx - pcx, d.cy - pcy))
        return min(ok, key=lambda d: math.hypot(d.cx - W / 2, d.cy - H / 2))

    def bar_pixels(self, gray, cx, cy, r):
        inner = np.zeros(gray.shape, np.uint8)
        cv2.circle(inner, (cx, cy), int(r * INNER), 1, -1)
        inside = gray[inner > 0]

        ring = np.zeros(gray.shape, np.uint8)
        cv2.circle(ring, (cx, cy), int(r * RING[1]), 1, -1)
        cv2.circle(ring, (cx, cy), int(r * RING[0]), 0, -1)
        paper = float(np.median(gray[ring > 0]))

        lo, hi = np.percentile(inside, [2, 98])
        if hi - lo < self.p.min_contrast * max(paper - lo, self.p.min_gap):
            # Nothing inside gets even part way from disc-black to paper-white;
            # stretching would only blow up noise/smudges into fake bars.
            return np.zeros(gray.shape, np.uint8), inside.size
        alpha = 255.0 / (hi - lo)
        stretched = cv2.convertScaleAbs(gray, alpha=alpha, beta=-lo * alpha)
        inside = stretched[inner > 0]

        t, _ = cv2.threshold(inside, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t = max(t, float(np.percentile(inside, 10)) + self.p.core_gap)
        return ((stretched > t) & (inner > 0)).astype(np.uint8), inside.size

    def bar_shapes(self, bars, searched):
        contours, _ = cv2.findContours(bars, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [cv2.minAreaRect(c) for c in contours
                if cv2.contourArea(c) >= self.p.min_bar_area_fraction * searched]

    def read(self, bgr, prev_center=None):
        H, W = bgr.shape[:2]
        gray, value = prepare(bgr)
        disc = self.nearest_disc(gray, value, prev_center)
        if disc is None:
            return Reading(None, 0.0, None, "no disc")
        cx, cy, r = int(disc.cx), int(disc.cy), int(disc.r)
        center = (disc.cx, disc.cy, disc.r)
        bbox = (max(0.0, (cx - r) / W), max(0.0, (cy - r) / H),
                min(1.0, (cx + r) / W), min(1.0, (cy + r) / H))

        bars, searched = self.bar_pixels(gray, cx, cy, r)
        if searched < 50:
            return Reading(None, 0.0, None, "disc too small", center)
        ink = bars.sum() / searched
        if ink < self.p.min_ink_fraction:
            return Reading("bars_0", 0.90, bbox, f"ink {ink:.1%}, blank disc", center)

        widths, lengths = [], []
        max_width = 2 * r * self.p.max_bar_width_frac
        for _center, (d1, d2), _angle in self.bar_shapes(bars, searched):
            width, length = min(d1, d2), max(d1, d2)
            if length < width * self.p.bar_aspect_min:
                continue
            if width > max_width:
                return Reading(None, 0.0, bbox, f"shape {width:.0f}px wide, bars merged", center)
            widths.append(round(width))
            lengths.append(length)

        k = len(widths)
        if k not in (1, 2, 3):
            return Reading(None, 0.0, bbox, f"{k} bar shapes, ink {ink:.0%}", center)

        longest = max(lengths)
        k_area = bars.sum() / (self.p.bar_width_ratio * longest * longest)
        if abs(k_area - k) > self.p.ink_tolerance:
            return Reading(None, 0.0, bbox, f"{k} shapes but area says {k_area:.1f} bars", center)

        conf = 0.85
        if k > 1:
            spread = float(np.std(widths)) / max(1.0, float(np.mean(widths)))
            conf = max(0.40, min(0.95, 0.95 - spread))
        return Reading(f"bars_{k}", conf, bbox,
                        f"{k} bars, widths {sorted(widths)}, r {r}px", center)
