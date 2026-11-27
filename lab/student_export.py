"""Train the student on every labelled day and export it for production (snowcam/student.py).

Confidence thresholds come from the last two days held out (heads fit on the rest), as in
lab/student_train.py; the exported heads are then refit on all days.

Usage: python -m lab.student_export DATASET_DIR EMB_DIR models/student-v1.npz --version v1
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from lab.student_train import FIELDS, lab, load


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("emb")
    ap.add_argument("out")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--target", type=float, default=0.95)
    a = ap.parse_args()
    E, rows = load(Path(a.dataset), Path(a.emb))
    days = sorted({r["day"] for r in rows})
    val_days = set(days[-2:])
    by_cam = defaultdict(list)
    for r in rows:
        by_cam[r["cam"]].append(E[r["i"]])
    cam_means = {c: np.mean(v, 0) for c, v in by_cam.items()}
    glob = np.mean([E[r["i"]] for r in rows], 0)
    X = lambda rs: np.stack([np.concatenate([E[r["i"]], E[r["i"]] - cam_means[r["cam"]]]) for r in rs])
    save = {"version": np.array(a.version), "cams": np.array(list(cam_means)),
            "cam_means": np.stack(list(cam_means.values())).astype(np.float32), "global_mean": glob.astype(np.float32)}
    fields = []
    for f in FIELDS:
        tl = lambda r: lab(r["sonnet"], f) or lab(r["first"], f)
        tr = [r for r in rows if tl(r) is not None]
        if len({tl(r) for r in tr}) < 2:
            continue
        fit = [r for r in tr if r["day"] not in val_days]
        val = [r for r in tr if r["day"] in val_days]
        m0 = LogisticRegression(max_iter=3000).fit(X(fit), [tl(r) for r in fit])
        pv = m0.predict_proba(X(val))
        conf, pred = pv.max(1), m0.classes_[pv.argmax(1)]
        ok = np.array([p == tl(r) for p, r in zip(pred, val)])
        thr = next((float(t) for t in np.arange(0.5, 0.995, 0.01) if (conf >= t).sum() >= 20 and ok[conf >= t].mean() >= a.target), 0.99)
        m = LogisticRegression(max_iter=3000).fit(X(tr), [tl(r) for r in tr])
        save[f"{f}_coef"] = m.coef_.astype(np.float32)
        save[f"{f}_intercept"] = m.intercept_.astype(np.float32)
        save[f"{f}_classes"] = np.array(m.classes_)
        save[f"{f}_threshold"] = np.array(thr)
        fields.append(f)
        print(f"{f}: {len(tr)} labels, classes {list(m.classes_)}, sure at p >= {thr:.2f}")
    save["fields"] = np.array(fields)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **save)
    print(f"wrote {a.out} ({Path(a.out).stat().st_size / 1e6:.2f} MB), {len(cam_means)} cams")


if __name__ == "__main__":
    main()
