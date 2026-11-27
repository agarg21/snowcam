"""Compare frozen encoders for the student on the same frames: those with a Sonnet answer. Each
encoder gets the same features (frame + 4-tile max, plus difference from the cam mean), one logistic
regression per field trained on Sonnet labels only, scored against the random Sonnet checks on the
test days (cams seen in training, and new cams). Concatenations are tried too ("a+b").

Usage: python -m lab.student_encoders DATASET_DIR clip=EMB_DIR siglip=EMB_DIR ... [--combo clip+dino]
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from lab.student_train import FIELDS, lab, load


def score(E, rows, keys, test_from):
    rows = [r for r in rows if r["key"] in keys]
    train = [r for r in rows if r["day"] < test_from]
    seen = {r["cam"] for r in train}
    test = [r for r in rows if r["day"] >= test_from]
    by_cam = defaultdict(list)
    for r in train:
        by_cam[r["cam"]].append(E[r["i"]])
    cm = {c: np.mean(v, 0) for c, v in by_cam.items()}
    zero = np.zeros(E.shape[1], dtype=E.dtype)
    # cams never seen in training have no mean: their difference part is zero
    X = lambda rs: np.stack([np.concatenate([E[r["i"]], E[r["i"]] - cm[r["cam"]] if r["cam"] in cm else zero])
                             for r in rs])
    out = {}
    for f in FIELDS:
        tr = [r for r in train if lab(r["sonnet_any"], f) is not None]
        y = [lab(r["sonnet_any"], f) for r in tr]
        if len(set(y)) < 2:
            continue
        m = LogisticRegression(max_iter=4000, C=1.0).fit(X(tr), y)
        res = []
        for name, group in (("seen", [r for r in test if r["cam"] in seen]), ("new", [r for r in test if r["cam"] not in seen])):
            g = [r for r in group if lab(r["sonnet"], f) is not None]
            if not g:
                res.append(f"{name} -"); continue
            p = m.predict(X(g))
            ok = sum(a == lab(r["sonnet"], f) for a, r in zip(p, g))
            res.append(f"{name} {ok}/{len(g)}")
        out[f] = "  ".join(res)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("encoders", nargs="+", help="name=EMB_DIR")
    ap.add_argument("--combo", action="append", default=[], help="e.g. clip+dino")
    ap.add_argument("--test-from", action="append", default=[])
    a = ap.parse_args()
    loaded = {}
    for spec in a.encoders:
        name, d = spec.split("=", 1)
        E, rows = load(Path(a.dataset), Path(d))
        loaded[name] = (E, rows)
    # common frames across all encoders
    keys = set.intersection(*[{r["key"] for r in rows} for _, rows in loaded.values()])
    print(f"{len(keys)} frames in common")
    for combo in a.combo:
        parts = combo.split("+")
        base_rows = loaded[parts[0]][1]
        idx = [{r["key"]: r["i"] for r in loaded[p][1]} for p in parts]
        ks = [r["key"] for r in base_rows if r["key"] in keys]
        E = np.concatenate([np.stack([loaded[p][0][ix[k]] for k in ks]) for p, ix in zip(parts, idx)], 1)
        pos = {k: i for i, k in enumerate(ks)}
        rows = [dict(r, i=pos[r["key"]]) for r in base_rows if r["key"] in keys]
        loaded[combo] = (E, rows)
    for tf in a.test_from or ["2026-11-25"]:
        print(f"\n### test from {tf}")
        for name, (E, rows) in loaded.items():
            print(f"## {name}")
            for f, s in score(E, rows, keys, tf).items():
                print(f"   {f:11s} {s}")


if __name__ == "__main__":
    main()
