"""Cheap image checks that run on every frame before any model call.

Each check returns pass / degraded / fail. Any fail means the frame publishes as
"No read" with a reason and never reaches the vision model.

Visibility follows the idea behind MIT Lincoln Lab's VEIA: compare edge strength in
the current frame with the same cam's clear-day reference. Until a cam has a
reference, absolute thresholds are used and the result is at most "degraded".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import cv2
import imagehash
import numpy as np
from astral import Observer
from astral.sun import elevation
from PIL import Image

from .config import Cam

WORK_W = 640

# Thresholds are first guesses; tune them on the labeled set.
SUN_DARK_DEG = -6.0          # below civil twilight
IR_SAT_MAX = 6.0             # mean HSV saturation below this = infrared/greyscale mode
MIN_BRIGHTNESS = 18.0        # mean V (0-255) below this = too dark even if lit
STALE_HASH_DIST = 2          # perceptual-hash distance counted as "same image"
STALE_RUN = 3                # identical frames in a row (daylight) = frozen cam
ABS_LOW_CONTRAST = 18.0      # grey std-dev with no reference
ABS_LOW_EDGES = 6.0          # mean Sobel magnitude with no reference
REF_EDGE_FAIL = 0.35         # edge ratio vs clear-day reference
REF_EDGE_DEGRADED = 0.60
MOVED_FRAC = 0.06            # median feature shift as a fraction of width


@dataclass
class Check:
    status: str  # pass | degraded | fail
    value: float | str | None = None
    note: str = ""


@dataclass
class GateResult:
    status: str
    checks: dict[str, Check] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def reasons(self) -> list[str]:
        return [f"{k}: {c.note}" for k, c in self.checks.items() if c.status != "pass" and c.note]

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "checks": {k: {"status": c.status, "value": c.value, "note": c.note} for k, c in self.checks.items()},
            "metrics": self.metrics,
        }


def _work(im: Image.Image) -> np.ndarray:
    w, h = im.size
    scale = WORK_W / w
    small = im.resize((WORK_W, max(1, int(h * scale))))
    return cv2.cvtColor(np.asarray(small), cv2.COLOR_RGB2BGR)


def metrics(im: Image.Image) -> dict[str, float]:
    bgr = _work(im)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return {
        "saturation": float(hsv[..., 1].mean()),
        "brightness": float(hsv[..., 2].mean()),
        "contrast": float(gray.std()),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "edges": float(np.sqrt(gx**2 + gy**2).mean()),
    }


def phash(im: Image.Image) -> str:
    return str(imagehash.phash(im))


def sun_elevation(cam: Cam, when: datetime) -> float:
    return float(elevation(Observer(latitude=cam.lat, longitude=cam.lon), when))


def _view_shift(im: Image.Image, ref: Image.Image) -> float | None:
    a = cv2.cvtColor(_work(im), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(_work(ref), cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(800)
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 30 or len(kb) < 30:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    if len(matches) < 20:
        return None
    shifts = [np.hypot(ka[m.queryIdx].pt[0] - kb[m.trainIdx].pt[0], ka[m.queryIdx].pt[1] - kb[m.trainIdx].pt[1]) for m in matches]
    return float(np.median(shifts) / WORK_W)


def run_gates(
    cam: Cam,
    im: Image.Image,
    when: datetime,
    frame_hash: str,
    recent_hashes: list[str],
    offline_hashes: list[str],
    reference: Image.Image | None = None,
    reference_metrics: dict[str, float] | None = None,
) -> GateResult:
    m = metrics(im)
    elev = sun_elevation(cam, when)
    m["sun_elevation"] = elev
    checks: dict[str, Check] = {}
    h = imagehash.hex_to_hash(frame_hash)

    # 1. Offline placeholder
    hit = any(h - imagehash.hex_to_hash(x) <= STALE_HASH_DIST for x in offline_hashes)
    checks["offline"] = Check("fail" if hit else "pass", None, "matches a known offline placeholder" if hit else "")

    # 2. Darkness (sun position first, then the pixels)
    ir = m["saturation"] < IR_SAT_MAX
    if elev < SUN_DARK_DEG and not cam.night_ok:
        checks["dark"] = Check("fail", round(elev, 1), f"sun {elev:.0f} deg below horizon" + (", infrared mode" if ir else ""))
    elif m["brightness"] < MIN_BRIGHTNESS:
        checks["dark"] = Check("fail", round(m["brightness"], 1), "frame too dark")
    elif ir and elev >= SUN_DARK_DEG:
        checks["dark"] = Check("degraded", round(elev, 1), "greyscale/infrared frame in twilight")
    else:
        checks["dark"] = Check("pass", round(elev, 1))
    daylight = elev >= SUN_DARK_DEG

    # 3. Stale (only meaningful in daylight; night scenes barely change)
    run = 0
    for x in recent_hashes:
        if h - imagehash.hex_to_hash(x) <= STALE_HASH_DIST:
            run += 1
        else:
            break
    stale = daylight and run >= STALE_RUN - 1
    checks["stale"] = Check("fail" if stale else "pass", run, f"identical to previous {run} frames" if stale else "")

    # 4. Visibility / blocked lens
    if reference_metrics:
        ratio = m["edges"] / max(reference_metrics.get("edges", 1e-6), 1e-6)
        m["edge_ratio"] = ratio
        if ratio < REF_EDGE_FAIL:
            checks["visibility"] = Check("fail", round(ratio, 2), "fog, whiteout or blocked lens (edges far below clear-day reference)")
        elif ratio < REF_EDGE_DEGRADED:
            checks["visibility"] = Check("degraded", round(ratio, 2), "reduced visibility vs clear-day reference")
        else:
            checks["visibility"] = Check("pass", round(ratio, 2))
    else:
        low = m["contrast"] < ABS_LOW_CONTRAST and m["edges"] < ABS_LOW_EDGES
        checks["visibility"] = Check("degraded" if low else "pass", round(m["edges"], 1),
                                     "low contrast; possible fog or wet lens (no reference yet)" if low else "")

    # 5. View moved
    if reference is not None:
        shift = _view_shift(im, reference)
        if shift is None:
            checks["moved"] = Check("degraded", None, "could not match features to reference")
        elif shift > MOVED_FRAC:
            checks["moved"] = Check("fail", round(shift, 3), "view differs from reference (moved or different preset)")
        else:
            checks["moved"] = Check("pass", round(shift, 3))

    statuses = [c.status for c in checks.values()]
    overall = "fail" if "fail" in statuses else "degraded" if "degraded" in statuses else "pass"
    return GateResult(overall, checks, {k: round(v, 3) for k, v in m.items()})
