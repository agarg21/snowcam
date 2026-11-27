"""Collect the embeddings the shadow pass saved into the archive ({stem}.emb.npz beside each frame)
into an EMB_DIR per encoder, the format lab/student_train.py and lab/student_export.py read. With
these, retraining is a download plus a few seconds of fitting: no frame is embedded twice.

Usage: python -m lab.student_embs_from_archive DATASET_DIR OUT_ROOT
   -> OUT_ROOT/<encoder slug>/chunk-0000.npz ... for every encoder found
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("out")
    a = ap.parse_args()
    root = Path(a.dataset)
    by_enc: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for p in sorted(root.rglob("*.emb.npz")):
        z = np.load(p, allow_pickle=True)
        key = str(p.relative_to(root))[:-len(".emb.npz")]
        for i, spec in enumerate(z["backbones"]):
            keys, embs = by_enc[str(spec)]
            keys.append(key)
            embs.append(z[f"enc{i}"].astype(np.float32))
    for spec, (keys, embs) in by_enc.items():
        d = Path(a.out) / re.sub(r"[^A-Za-z0-9.-]+", "_", spec)
        d.mkdir(parents=True, exist_ok=True)
        for c in range(0, len(keys), 2000):
            np.savez(d / f"chunk-{c // 2000:04d}.npz", emb=np.stack(embs[c:c + 2000]), key=np.array(keys[c:c + 2000]))
        print(f"{spec}: {len(keys)} frames -> {d}")


if __name__ == "__main__":
    main()
