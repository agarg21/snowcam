"""Embed every frame of the private dataset (snowcam-lab day zips, extracted) with a frozen CLIP
model on CPU: the whole frame plus the max over its four quadrant tiles (small things like snow-gun
plumes still register). Resumable: writes chunks and skips frames already embedded.

Usage: python -m lab.student_embed DATASET_DIR OUT_DIR [--model ViT-B-32 --pretrained openai]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("out")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--chunk", type=int, default=512)
    a = ap.parse_args()
    import torch
    import open_clip
    from PIL import Image

    torch.set_num_threads(max(1, torch.get_num_threads()))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    done = set()
    for f in out.glob("chunk-*.npz"):
        done.update(np.load(f, allow_pickle=True)["key"].tolist())
    jpgs = sorted(p for p in Path(a.dataset).rglob("*.jpg"))
    todo = [p for p in jpgs if str(p.relative_to(a.dataset))[:-4] not in done]
    print(f"{len(jpgs)} frames, {len(done)} already embedded, {len(todo)} to go", flush=True)
    model, _, prep = open_clip.create_model_and_transforms(a.model, pretrained=a.pretrained)
    model.eval()
    t0, n = time.time(), 0
    for c in range(0, len(todo), a.chunk):
        part = todo[c:c + a.chunk]
        keys, embs = [], []
        with torch.no_grad():
            for i in range(0, len(part), 8):
                batch, ks = [], []
                for p in part[i:i + 8]:
                    im = Image.open(p).convert("RGB")
                    w, h = im.size
                    tiles = [im.crop((x * w // 2, y * h // 2, (x + 1) * w // 2, (y + 1) * h // 2)) for y in (0, 1) for x in (0, 1)]
                    batch += [prep(t) for t in [im] + tiles]
                    ks.append(str(p.relative_to(a.dataset))[:-4])
                e = model.encode_image(torch.stack(batch))
                e = (e / e.norm(dim=-1, keepdim=True)).numpy().astype(np.float32).reshape(len(ks), 5, -1)
                embs.append(np.concatenate([e[:, 0], e[:, 1:].max(1)], 1))
                keys += ks
        np.savez(out / f"chunk-{len(list(out.glob('chunk-*.npz'))):04d}.npz", emb=np.concatenate(embs), key=np.array(keys))
        n += len(part)
        print(f"{len(done) + n}/{len(jpgs)}  {(time.time() - t0) / n:.3f} s/frame", flush=True)
    meta = {"model": a.model, "pretrained": a.pretrained, "features": "whole + max over 4 tiles"}
    (out / "meta.json").write_text(json.dumps(meta))


if __name__ == "__main__":
    main()
