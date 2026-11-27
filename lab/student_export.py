"""Train the student on every labelled day and export it for production (snowcam/student.py).

Each field can use its own frozen image model (encoder); the model file stores, per encoder, its
spec and per-cam mean embeddings, and per field which encoder its head reads. Confidence thresholds
come from the last two days held out (heads fit on the rest), as in lab/student_train.py; the
exported heads are then refit on all days.

Usage:
  python -m lab.student_export DATASET_DIR models/student-v3.npz --version v3 \\
      --enc clip=EMB_DIR=open_clip:ViT-B-32-quickgelu:openai \\
      --enc dino=EMB_DIR=timm:vit_small_patch14_reg4_dinov2.lvd142m \\
      --use snow_cover=dino --use snowmaking=dino        (other fields use the first --enc)
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from lab.student_train import FIELDS, load, train_label


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("out")
    ap.add_argument("--version", required=True)
    ap.add_argument("--target", type=float, default=0.95)
    ap.add_argument("--enc", action="append", required=True, help="name=EMB_DIR=SPEC")
    ap.add_argument("--use", action="append", default=[], help="field=encoder name")
    a = ap.parse_args()
    encs = []
    for spec in a.enc:
        name, d, bb = spec.split("=", 2)
        E, rows = load(Path(a.dataset), Path(d))
        encs.append((name, bb, E, {r["key"]: r for r in rows}))
    names = [e[0] for e in encs]
    use = dict(u.split("=", 1) for u in a.use)
    keys = sorted(set.intersection(*[set(e[3]) for e in encs]))
    rows = [encs[0][3][k] for k in keys]  # labels are the same in every encoder's rows
    days = sorted({r["day"] for r in rows})
    val_days = set(days[-2:])
    cams = sorted({r["cam"] for r in rows})
    save = {"version": np.array(a.version), "encoders": np.array(names), "cams": np.array(cams)}
    feats = {}
    for i, (name, bb, E, by_key) in enumerate(encs):
        emb = {k: E[by_key[k]["i"]] for k in keys}
        by_cam = defaultdict(list)
        for r in rows:
            by_cam[r["cam"]].append(emb[r["key"]])
        cm = {c: np.mean(v, 0) for c, v in by_cam.items()}
        save[f"enc{i}_backbone"] = np.array(bb)
        save[f"enc{i}_cam_means"] = np.stack([cm[c] for c in cams]).astype(np.float32)
        save[f"enc{i}_global_mean"] = np.mean(list(emb.values()), 0).astype(np.float32)
        feats[name] = lambda rs, emb=emb, cm=cm: np.stack(
            [np.concatenate([emb[r["key"]], emb[r["key"]] - cm[r["cam"]]]) for r in rs])
    fields = []
    for f in FIELDS:
        enc = use.get(f, names[0])
        X = feats[enc]
        tl = lambda r: train_label(r, f)
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
        save[f"{f}_encoder"] = np.array(names.index(enc))
        save[f"{f}_coef"] = m.coef_.astype(np.float32)
        save[f"{f}_intercept"] = m.intercept_.astype(np.float32)
        save[f"{f}_classes"] = np.array(m.classes_)
        save[f"{f}_threshold"] = np.array(thr)
        fields.append(f)
        print(f"{f}: {enc}, {len(tr)} labels, classes {list(m.classes_)}, sure at p >= {thr:.2f}")
    save["fields"] = np.array(fields)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **save)
    print(f"wrote {a.out} ({Path(a.out).stat().st_size / 1e6:.2f} MB), {len(cams)} cams")


if __name__ == "__main__":
    main()
