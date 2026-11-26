"""Cost experiment (owner, 2026-11-26): can reads get cheaper without changing what they say?

Re-reads a sample of recent frames, all with full-resolution frames from collect run artifacts. It
builds each request with the production code (reader.build_request) and tries one change at a time
against a rerun of the unchanged request, which gives the noise floor (the same model asked twice
does not always agree with itself).
  base       production request, same model as production
  no_prev    without the previous-frame image (one image fewer)
  small      every image sent at 768 px wide instead of 1092 (about half the image tokens)
  cached     production request with prompt caching on the fixed tools + system prefix
  haiku      stake cams only: Haiku instead of Sonnet
Agreement is per published field, against the production reading of the same frame, on the scales
the pages show. Writes results.json and a summary; frames are never uploaded anywhere.

Usage (in the lab-cost workflow): python -m lab.cost_experiment FRAMES_DIR --n 80 --stakes 25
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from snowcam import reader, store
from snowcam.cli import _local_label, _ref
from snowcam.config import load_registry

PRICE = {"claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125), "claude-sonnet-5-5": (2.0, 10.0, 0.20, 2.5)}  # in, out, cache read, cache write per 1M
SCALE = {
    "snow_cover": {"full": "good", "mostly": "good", "patchy": "thin", "bare": "thin"},
    "fresh_snow": {"none": "none", "dusting": "some", "light": "some", "moderate": "some", "heavy": "some"},
    "snowmaking": {"on": "on", "off": "off"},
    "precip_now": {"none": "none", "snow": "snow", "rain": "rain", "mixed": "rain"},
    "visibility": {"clear": "clear", "hazy": "hazy", "fog": "poor", "whiteout": "poor"},
    "base_busyness": {"empty": "quiet", "light": "quiet", "normal": "normal", "busy": "busy", "packed": "busy"},
    "lifts_moving": {"yes": "yes", "no": "no"},
}
MIN_CONF = 0.6


def scaled(fields: dict) -> dict:
    out = {}
    for k, v in (fields or {}).items():
        if not isinstance(v, dict) or (v.get("conf") or 0) < MIN_CONF:
            continue
        if k == "stake_inches":
            out[k] = v.get("value")
        elif k in SCALE and v.get("value") in SCALE[k]:
            out[k] = SCALE[k][v["value"]]
    return out


def cost(model: str, usage: dict) -> float:
    i, o, cr, cw = PRICE[model]
    return (usage.get("input_tokens", 0) * i + usage.get("output_tokens", 0) * o
            + usage.get("cache_read_input_tokens", 0) * cr + usage.get("cache_creation_input_tokens", 0) * cw) / 1e6


def with_cache(req: dict) -> dict:
    if not isinstance(req["system"], str):
        return req  # production requests are cached since 2026-11-26
    tools = [dict(t) for t in req["tools"]]
    tools[-1]["cache_control"] = {"type": "ephemeral"}
    return {**req, "tools": tools, "system": [{"type": "text", "text": req["system"], "cache_control": {"type": "ephemeral"}}]}


def frame_index(root: Path) -> dict[str, Path]:
    idx = {}
    for p in root.rglob("*.jpg"):
        m = re.search(r"frames/([^/]+)/(\d{8})/(\d{6})\.jpg$", p.as_posix())
        if m:
            idx["/".join(m.groups())] = p
    return idx


def key_of(path: str | None) -> str | None:
    m = re.search(r"frames/([^/]+)/(\d{8})/(\d{6})\.jpg$", path or "")
    return "/".join(m.groups()) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("frames")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--stakes", type=int, default=25)
    ap.add_argument("--out", default="lab-out")
    a = ap.parse_args()
    reg = load_registry()
    con = sqlite3.connect(store.DATA / "snowcam.db")
    con.row_factory = sqlite3.Row
    idx = frame_index(Path(a.frames))
    rows = [r for r in con.execute(
        "select r.id, r.cam, r.ts, r.model, r.fields, f.path from readings r join frames f on f.id=r.frame_id "
        "order by r.ts desc limit 4000") if key_of(r["path"]) in idx]
    rng = random.Random(7)
    rng.shuffle(rows)
    stake, other = [], []
    for r in rows:
        try:
            cam = reg.cam(r["cam"])
        except KeyError:
            continue
        (stake if cam.role == "stake" else other).append((r, cam))
    sample = other[: a.n] + stake[: a.stakes]
    print(f"{len(rows)} production reads have their frame here; sampling {len(other[:a.n])} non-stake, {len(stake[:a.stakes])} stake")

    def prev_for(r, cam):
        p = con.execute("select f.path, r.ts from readings r join frames f on f.id=r.frame_id where r.cam=? and r.ts<? "
                        "order by r.ts desc limit 1", (cam.id, r["ts"])).fetchone()
        k = key_of(p["path"]) if p else None
        return (Image.open(idx[k]).convert("RGB"), _local_label(p["ts"], cam.tz)) if k in idx else None

    jobs = []
    for r, cam in sample:
        cur = [Image.open(idx[key_of(r["path"])]).convert("RGB")]
        ref, _ = _ref(cam.id)
        prev = prev_for(r, cam)
        label = _local_label(r["ts"], cam.tz)
        build = lambda **kw: reader.build_request(cam, kw.get("cur", cur), label, kw.get("ref", ref), kw.get("prev", prev), kw.get("model", r["model"]))
        variants = {"base": build(), "cached": with_cache(build())}
        if prev is not None:
            variants["no_prev"] = build(prev=None)
        small = lambda im: im.resize((768, int(im.height * 768 / im.width))) if im and im.width > 768 else im
        variants["small"] = build(cur=[small(c) for c in cur], ref=small(ref) if ref else None,
                                  prev=(small(prev[0]), prev[1]) if prev else None)
        if cam.role == "stake":
            variants["haiku"] = build(model=reader.DEFAULT_MODEL)
        if variants["small"]["messages"] and cam.crop:
            pass  # the close-up crop is cut from the (now smaller) current frame: part of what "small" tests
        for name, req in variants.items():
            jobs.append((r, cam, name, req))

    def run(job):
        r, cam, name, req = job
        try:
            out = reader.call(req)
            return job, out, None
        except Exception as e:
            return job, None, e

    with ThreadPoolExecutor(6) as ex:
        results = list(ex.map(run, jobs))

    agg = defaultdict(lambda: {"calls": 0, "errors": 0, "cost": 0.0, "agree": defaultdict(lambda: [0, 0]), "in": 0, "out": 0, "cache_read": 0})
    rows_out = []
    for (r, cam, name, req), out, err in results:
        group = f"{'stake' if cam.role == 'stake' else 'other'}:{name}"
        g = agg[group]
        g["calls"] += 1
        if err is not None:
            g["errors"] += 1
            continue
        u = out.raw.get("usage") or {}
        g["cost"] += cost(req["model"], u)
        g["in"] += u.get("input_tokens", 0); g["out"] += u.get("output_tokens", 0); g["cache_read"] += u.get("cache_read_input_tokens", 0)
        prod, got = scaled(json.loads(r["fields"])), scaled(out.fields)
        for k in set(prod) & set(got):
            ok = abs((prod[k] or 0) - (got[k] or 0)) <= 1 if k == "stake_inches" else prod[k] == got[k]
            g["agree"][k][0] += int(ok); g["agree"][k][1] += 1
        rows_out.append({"reading_id": r["id"], "cam": cam.id, "variant": name, "model": req["model"], "usage": u,
                         "fields": scaled(out.fields), "production": prod})
    summary = {}
    for group, g in sorted(agg.items()):
        ok = g["calls"] - g["errors"]
        summary[group] = {"calls": g["calls"], "errors": g["errors"], "cost_per_read": round(g["cost"] / max(1, ok), 5),
                          "in_tokens": round(g["in"] / max(1, ok)), "out_tokens": round(g["out"] / max(1, ok)),
                          "cache_read_tokens": round(g["cache_read"] / max(1, ok)),
                          "agreement": {k: f"{v[0]}/{v[1]} ({v[0] / v[1]:.0%})" for k, v in sorted(g["agree"].items()) if v[1]}}
    Path(a.out).mkdir(exist_ok=True)
    (Path(a.out) / "results.json").write_text(json.dumps({"summary": summary, "rows": rows_out}, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"total spend ${sum(g['cost'] for g in agg.values()):.3f}")


if __name__ == "__main__":
    sys.exit(main())
