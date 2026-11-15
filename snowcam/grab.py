"""Fetch one frame per cam.

Stills (including Brownrice snapshots) are a single polite HTTP GET.
HLS feeds take a short 3-frame burst with ffmpeg and disconnect.
"""
from __future__ import annotations

import io
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import requests
from PIL import Image

from .config import Cam

TIMEOUT = 25


@dataclass
class Grab:
    ok: bool
    images: list[Image.Image]
    error: str | None = None
    http_status: int | None = None


def _get_still(url: str, user_agent: str) -> Grab:
    try:
        r = requests.get(url, headers={"User-Agent": user_agent}, timeout=TIMEOUT)
    except requests.RequestException as e:
        return Grab(False, [], f"request failed: {e.__class__.__name__}")
    if r.status_code != 200:
        return Grab(False, [], f"http {r.status_code}", r.status_code)
    ctype = r.headers.get("content-type", "")
    # Some cam hosts send no or a generic type; accept those only if the bytes open as an image.
    if not (ctype.startswith("image/") or ctype in ("", "application/octet-stream", "binary/octet-stream")):
        return Grab(False, [], f"not an image ({ctype})", r.status_code)
    try:
        im = Image.open(io.BytesIO(r.content))
        im.load()
    except Exception as e:  # corrupt image
        return Grab(False, [], f"bad image: {e}", r.status_code)
    return Grab(True, [im.convert("RGB")], None, r.status_code)


def _get_hls_burst(url: str, user_agent: str, frames: int = 3, spacing_s: float = 5.0) -> Grab:
    with tempfile.TemporaryDirectory() as d:
        pattern = str(Path(d) / "f%02d.jpg")
        cmd = [
            "ffmpeg", "-loglevel", "error", "-y", "-user_agent", user_agent,
            "-i", url, "-vf", f"fps=1/{spacing_s}", "-frames:v", str(frames), "-q:v", "3", pattern,
        ]
        try:
            subprocess.run(cmd, check=True, timeout=frames * spacing_s + 40, capture_output=True)
        except subprocess.CalledProcessError as e:
            return Grab(False, [], f"ffmpeg: {e.stderr.decode(errors='ignore')[:200]}")
        except subprocess.TimeoutExpired:
            return Grab(False, [], "ffmpeg timeout")
        ims = []
        for p in sorted(Path(d).glob("f*.jpg")):
            im = Image.open(p)
            im.load()
            ims.append(im.convert("RGB"))
    return Grab(bool(ims), ims, None if ims else "no frames")


def grab(cam: Cam, user_agent: str) -> Grab:
    if cam.feed in ("brownrice", "still"):
        return _get_still(cam.source_url, user_agent)
    if cam.feed == "hls":
        return _get_hls_burst(cam.source_url, user_agent)
    return Grab(False, [], f"unknown feed type {cam.feed}")
