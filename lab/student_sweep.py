"""Cheap variants of the student on existing embeddings, scored on the test days against the random
Sonnet checks (same yardstick as lab/student_train.py). Variants: regularisation C, a small MLP head,
and training labels (Sonnet + confident first reads vs Sonnet only).

Usage: python -m lab.student_sweep DATASET_DIR EMB_DIR [--test-from 2026-11-25]
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from lab.student_train import FIELDS, lab, load


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("emb")
    ap.add_argument("--test-from", default="2026-11-25")
    ap.add_argument("--fields", default=",".join(FIELDS))
    a = ap.parse_args()
    E, rows = load(Path(a.dataset), Path(a.emb))
    train = [r for r in rows if r["day"] < a.test_from]
    seen = {r["cam"] for r in train}
    test = [r for r in rows if r["day"] >= a.test_from and r["cam"] in seen]
    by_cam = defaultdict(list)
    for r in train:
        by_cam[r["cam"]].append(E[r["i"]])
    cm = {c: np.mean(v, 0) for c, v in by_cam.items()}
    X = lambda rs: np.stack([np.concatenate([E[r["i"]], E[r["i"]] - cm[r["cam"]]]) for r in rs])
    Xte = X(test)
    for f in a.fields.split(","):
        gold = [(k, lab(r["sonnet"], f)) for k, r in enumerate(test) if lab(r["sonnet"], f) is not None]
        idx = [k for k, _ in gold]
        y = [g for _, g in gold]
        print(f"## {f}  (test frames with a random Sonnet check: {len(gold)})")
        variants = {}
        mixed = [r for r in train if (lab(r["sonnet"], f) or lab(r["first"], f)) is not None]
        son = [r for r in train if lab(r["sonnet"], f) is not None]
        ym = [lab(r["sonnet"], f) or lab(r["first"], f) for r in mixed]
        ys = [lab(r["sonnet"], f) for r in son]
        Xm, Xs = X(mixed), X(son)
        for C in (0.1, 0.3, 1.0, 3.0):
            variants[f"logreg C={C} mixed labels"] = LogisticRegression(max_iter=4000, C=C).fit(Xm, ym)
        if len(set(ys)) > 1:
            variants["logreg C=1 Sonnet-only labels"] = LogisticRegression(max_iter=4000, C=1.0).fit(Xs, ys)
        variants["MLP 256 mixed labels"] = MLPClassifier((256,), alpha=1e-3, max_iter=300, early_stopping=True,
                                                         random_state=0).fit(Xm, ym)
        for name, m in variants.items():
            p = m.predict(Xte[idx])
            print(f"   {name:32s} {sum(a_ == b for a_, b in zip(p, y))}/{len(y)}")


if __name__ == "__main__":
    main()
