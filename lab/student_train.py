"""Train and test a cheap student reader on frozen CLIP embeddings (lab/student_embed.py).

Labels per frame and field, on the scales the pages publish: the Sonnet second opinion where the
frame has one (conf >= 0.6), else the production first read (conf >= 0.6). Features: the frame's
embedding plus its difference from the cam's own average (training days only). One logistic
regression per field.

Test: train on days before --test-from, test on the later days (cams the student has seen). The
honest yardstick is the Sonnet check: on test frames that have one, how often does the student agree
with Sonnet, compared with the production first read (Haiku) on the same frames? Then a gate: keep
only frames where the student is confident (threshold picked on the last training days for a target
agreement with the first read) and report coverage and agreement with Sonnet there.

Usage: python -m lab.student_train DATASET_DIR EMB_DIR --test-from 2026-11-25 --out results.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

SCALE = {
    "snow_cover": {"full": "good", "mostly": "good", "patchy": "thin", "bare": "thin"},
    "fresh_snow": {"none": "none", "dusting": "some", "light": "some", "heavy": "some"},
    "visibility": {"clear": "clear", "hazy": "hazy", "fog": "poor", "whiteout": "poor"},
    "snowmaking": {"on": "on", "off": "off"},
    "precip_now": {"none": "none", "snow": "snow", "rain_or_mix": "wet"},
}
FIELDS = list(SCALE)
# Fields trained on Sonnet labels only: the first read's labels disagreed with Sonnet often enough to
# hurt (lab sweep 11-27, two splits: snow cover 78% -> 92% and 78% -> 84%; new cams 90% -> 95%).
# Fresh snow gained 2 frames on seen cams but fell 98 -> 78 of 101 on new cams, so it keeps mixed labels.
SONNET_ONLY = {"snow_cover"}


def train_label(r: dict, f: str):
    if f in SONNET_ONLY:
        return lab(r["sonnet_any"], f)
    return lab(r["sonnet_any"], f) or lab(r["first"], f)


def lab(fields: dict | None, f: str):
    v = (fields or {}).get(f)
    if not isinstance(v, dict) or (v.get("conf") or 0) < 0.6:
        return None
    return SCALE[f].get(v.get("value"))


def load(dataset: Path, emb_dir: Path):
    E, K = [], []
    for c in sorted(emb_dir.glob("chunk-*.npz")):
        z = np.load(c, allow_pickle=True)
        E.append(z["emb"]); K += z["key"].tolist()
    E = np.concatenate(E)
    rows = []
    for i, k in enumerate(K):
        j = json.loads((dataset / f"{k}.json").read_text())
        if j.get("role") == "stake":
            continue
        chk = j.get("check") or {}
        # Only random-sample checks count as the yardstick (spot tie-breaks happen because the
        # first read was disputed, which would bias the comparison against it).
        is_sonnet = "sonnet" in (chk.get("model") or "")
        sonnet = chk.get("fields") if is_sonnet and chk.get("phase") == "initial" else None
        rows.append({"i": i, "key": k, "cam": j["cam"], "resort": j["resort"], "day": j["time_local"][:10],
                     "first": j["teacher"]["fields"], "sonnet": sonnet,
                     # any Sonnet answer is a good training label (tie-breaks included)
                     "sonnet_any": chk.get("fields") if is_sonnet else None})
    return E, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("emb")
    ap.add_argument("--test-from", default="2026-11-25")
    ap.add_argument("--target", type=float, default=0.95)
    ap.add_argument("--out", default="student_results.json")
    a = ap.parse_args()
    E, rows = load(Path(a.dataset), Path(a.emb))
    train = [r for r in rows if r["day"] < a.test_from]
    seen = {r["cam"] for r in train}
    test_all = [r for r in rows if r["day"] >= a.test_from]
    test = [r for r in test_all if r["cam"] in seen]          # cams the student has seen
    test_new = [r for r in test_all if r["cam"] not in seen]  # resorts added on 11-26
    days = sorted({r["day"] for r in train})
    val_days = set(days[-2:])
    # per-cam mean embedding from training days only
    cam_mean = {}
    by_cam = defaultdict(list)
    for r in train:
        by_cam[r["cam"]].append(E[r["i"]])
    for c, v in by_cam.items():
        cam_mean[c] = np.mean(v, 0)
    glob = np.mean([E[r["i"]] for r in train], 0)
    X = lambda rs: np.stack([np.concatenate([E[r["i"]], E[r["i"]] - cam_mean.get(r["cam"], glob)]) for r in rs])
    out = {"frames": {"train": len(train), "test": len(test)}, "fields": {}}
    print(f"train {len(train)} frames ({days[0]}..{days[-1]}), test {len(test)} frames on seen cams + "
          f"{len(test_new)} on new cams (from {a.test_from})")
    out["frames"]["test_new_cams"] = len(test_new)
    for f in FIELDS:
        tl = lambda r: train_label(r, f)
        tr = [r for r in train if tl(r) is not None]
        classes = sorted({tl(r) for r in tr})
        if len(classes) < 2 or len(tr) < 50:
            continue
        # threshold on the last two training days (fit on the rest)
        fit = [r for r in tr if r["day"] not in val_days]
        val = [r for r in tr if r["day"] in val_days]
        m0 = LogisticRegression(max_iter=3000, C=1.0).fit(X(fit), [tl(r) for r in fit])
        pv = m0.predict_proba(X(val)) if val else None
        thr = 0.5
        if val:
            conf, pred = pv.max(1), m0.classes_[pv.argmax(1)]
            ok = np.array([p == tl(r) for p, r in zip(pred, val)])
            for t in np.arange(0.5, 0.995, 0.01):
                sel = conf >= t
                if sel.sum() >= 20 and ok[sel].mean() >= a.target:
                    thr = float(t)
                    break
            else:
                thr = 0.99
        m = LogisticRegression(max_iter=3000, C=1.0).fit(X(tr), [tl(r) for r in tr])
        P = m.predict_proba(X(test))
        pred, conf = m.classes_[P.argmax(1)], P.max(1)
        res = {"classes": classes, "train_labels": len(tr), "threshold": round(thr, 2)}
        # vs the first read, all test frames where it is confident
        fr = [(p, lab(r["first"], f), c) for p, r, c in zip(pred, test, conf) if lab(r["first"], f) is not None]
        res["student_vs_first"] = f"{sum(p == y for p, y, _ in fr)}/{len(fr)}"
        # the yardstick: frames with a Sonnet check
        g = [(p, lab(r["first"], f), lab(r["sonnet"], f), c) for p, r, c in zip(pred, test, conf) if lab(r["sonnet"], f) is not None]
        both = [x for x in g if x[1] is not None]
        res["sonnet_frames"] = len(g)
        res["student_vs_sonnet"] = f"{sum(p == s for p, _, s, _ in g)}/{len(g)}"
        res["first_vs_sonnet"] = f"{sum(y == s for _, y, s, _ in both)}/{len(both)}"
        res["student_vs_sonnet_same_frames"] = f"{sum(p == s for p, _, s, _ in both)}/{len(both)}"
        gated = [x for x in g if x[3] >= thr]
        res["gated_coverage_all_test"] = f"{(conf >= thr).mean():.0%}"
        res["gated_student_vs_sonnet"] = f"{sum(p == s for p, _, s, _ in gated)}/{len(gated)}"
        if test_new:
            Pn = m.predict_proba(X(test_new))
            pn = m.classes_[Pn.argmax(1)]
            gn = [(p, lab(r["sonnet"], f)) for p, r in zip(pn, test_new) if lab(r["sonnet"], f) is not None]
            fn = [(p, lab(r["first"], f)) for p, r in zip(pn, test_new) if lab(r["first"], f) is not None]
            res["new_cams_student_vs_first"] = f"{sum(p == y for p, y in fn)}/{len(fn)}"
            res["new_cams_student_vs_sonnet"] = f"{sum(p == y for p, y in gn)}/{len(gn)}"
        out["fields"][f] = res
        print(f, json.dumps(res))
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
