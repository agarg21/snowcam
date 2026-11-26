"""Prompt experiment (2026-11-26): do shared field definitions (prompt v0.6) make the two readers agree
more, on the same frames, than the production pair (v0.3-v0.5) did?

Samples recent non-stake reads that already have a Sonnet second opinion (initial or tie-broken
spot), about half from the resorts that keep failing the check. For each frame it runs the
current code's first read (Haiku, production request with reference and previous frame) and its
second-opinion check (Sonnet, verify.build_request), then compares on the published scales:
  old pair: stored first read vs stored Sonnet check
  new pair: new first read vs new Sonnet check
plus how often each new read matches its old counterpart (stability). Writes results.json.

Usage (lab-prompt workflow, run on the branch that carries the new prompt):
  python -m lab.prompt_experiment FRAMES_DIR --n 100 --out lab-out
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from lab.cost_experiment import frame_index, key_of
from snowcam import reader, store, verify
from snowcam.cli import _local_label, _ref
from snowcam.config import load_registry

FIELDS = ["snow_cover", "fresh_snow", "visibility", "snowmaking", "precip_now"]
HARD = {"park-city", "stevens-pass", "deer-valley", "crested-butte", "schweitzer", "keystone", "copper"}
SCALE = {"fresh_snow": {"dusting": "some", "light": "some", "heavy": "some"},
         "visibility": {"fog": "poor", "whiteout": "poor"}}


def val(f, d):
    x = (d or {}).get(f)
    if not isinstance(x, dict) or (x.get("conf") or 0) < verify.MIN_CONF or x.get("value") in (None, "unknown"):
        return None
    v = verify._val(f, x)
    return SCALE.get(f, {}).get(v, v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("frames")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="lab-out")
    a = ap.parse_args()
    reg = load_registry()
    con = sqlite3.connect(store.DATA / "snowcam.db")
    con.row_factory = sqlite3.Row
    idx = frame_index(Path(a.frames))
    rows = [r for r in con.execute(
        "select r.id, r.cam, r.ts, r.model, r.fields first, v.fields chk, v.resort from verifications v "
        "join readings r on r.id=v.reading_id join frames f on f.id=r.frame_id "
        "where v.model='claude-sonnet-5-5' order by v.id desc limit 3000") if True]
    paths = {r["id"]: con.execute("select f.path from readings r join frames f on f.id=r.frame_id where r.id=?",
                                  (r["id"],)).fetchone()[0] for r in rows}
    rows = [r for r in rows if key_of(paths[r["id"]]) in idx]
    rng = random.Random(11)
    rng.shuffle(rows)
    hard = [r for r in rows if r["resort"] in HARD]
    rest = [r for r in rows if r["resort"] not in HARD]
    sample = []
    for r in hard[: a.n // 2] + rest[: a.n - min(len(hard), a.n // 2)]:
        try:
            cam = reg.cam(r["cam"])
        except KeyError:
            continue
        if cam.role != "stake":
            sample.append((r, cam))
    print(f"{len(rows)} checked reads have their frame here; sampling {len(sample)} "
          f"({sum(r['resort'] in HARD for r, _ in sample)} from the struggling resorts)")

    def prev_for(r, cam):
        p = con.execute("select f.path, r.ts from readings r join frames f on f.id=r.frame_id where r.cam=? and r.ts<? "
                        "order by r.ts desc limit 1", (cam.id, r["ts"])).fetchone()
        k = key_of(p["path"]) if p else None
        return (Image.open(idx[k]).convert("RGB"), _local_label(p["ts"], cam.tz)) if k in idx else None

    prevs = {r["id"]: prev_for(r, cam) for r, cam in sample}  # sqlite stays on this thread

    def run(item):
        r, cam = item
        frame = Image.open(idx[key_of(paths[r["id"]])]).convert("RGB")
        ref, _ = _ref(cam.id)
        try:
            first = reader.call(reader.build_request(cam, [frame], _local_label(r["ts"], cam.tz), ref, prevs[r["id"]],
                                                     reader.DEFAULT_MODEL)).fields
            check = verify.call(verify.build_request(cam, frame, ref, verify.INITIAL_MODEL))
            return r, cam, first, check, None
        except Exception as e:
            return r, cam, None, None, f"{type(e).__name__}: {str(e)[:300]}"

    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(run, sample))

    agg = {k: defaultdict(lambda: [0, 0]) for k in ("old_pair", "new_pair", "first_stable", "check_stable")}
    hard_agg = {k: defaultdict(lambda: [0, 0]) for k in ("old_pair", "new_pair")}
    rows_out, errors = [], 0
    for r, cam, first, check, err in res:
        if err:
            errors += 1
            if errors <= 3:
                print("ERROR", cam.id, err)
            continue
        of, oc = json.loads(r["first"]), json.loads(r["chk"])
        for f in FIELDS:
            pairs = {"old_pair": (val(f, of), val(f, oc)), "new_pair": (val(f, first), val(f, check)),
                     "first_stable": (val(f, of), val(f, first)), "check_stable": (val(f, oc), val(f, check))}
            for k, (x, y) in pairs.items():
                if x is not None and y is not None:  # both confident: a real comparison
                    agg[k][f][0] += x == y
                    agg[k][f][1] += 1
                    if r["resort"] in HARD and k in hard_agg:
                        hard_agg[k][f][0] += x == y
                        hard_agg[k][f][1] += 1
            # how often each pair gives a usable (confident) answer at all
            for k, (x, y) in (("old_pair", pairs["old_pair"]), ("new_pair", pairs["new_pair"])):
                agg[k][f + "_both_confident"][1] += 1
                agg[k][f + "_both_confident"][0] += x is not None and y is not None
        rows_out.append({"resort": r["resort"], "cam": cam.id, "ts": r["ts"], "old_first": of, "old_check": oc,
                         "new_first": first, "new_check": check})
    fmt = lambda d: {f: f"{v[0]}/{v[1]} ({v[0] / v[1]:.0%})" if v[1] else "-" for f, v in sorted(d.items())}
    summary = {"frames": len(sample), "errors": errors, **{k: fmt(v) for k, v in agg.items()},
               "hard_resorts": {k: fmt(v) for k, v in hard_agg.items()}}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"summary": summary, "rows": rows_out}, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    if reader.CALLS:
        print(f"spend: ${sum(c[-1] for c in reader.CALLS):.2f} over {len(reader.CALLS)} calls")


if __name__ == "__main__":
    main()
