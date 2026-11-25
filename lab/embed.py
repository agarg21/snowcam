"""Embed every archived frame that has a teacher reading. Usage: python -I embed.py ART_DIR DB OUT.npz MODEL PRETRAINED"""
import sys, json, sqlite3, time, os, re
from pathlib import Path
import numpy as np, torch, open_clip
from PIL import Image
art, db, out, model_name, pre = sys.argv[1:6]
torch.set_num_threads(2)
TILES = os.environ.get("TILES") == "1"
idx = {}
for p in Path(art).rglob("*.jpg"):
    parts = p.parts
    i = parts.index("frames")
    cam, day, t = parts[i+1], parts[i+2], parts[i+3][:-4]
    if not re.fullmatch(r"\d{6}", t): continue  # skip burst extras
    idx[f"{cam}/{day}/{t}"] = p
con = sqlite3.connect(db)
rows = []
for rid, path, fields, cam, ts in con.execute("select r.id, f.path, r.fields, r.cam, r.ts from readings r join frames f on f.id=r.frame_id"):
    m = re.search(r"frames/([^/]+)/(\d{8})/(\d{6})\.jpg$", path or "")
    if m and (k := "/".join(m.groups())) in idx:
        rows.append((rid, k, cam, ts, fields))
print("matched", len(rows), "of", len(idx), "frames", flush=True)
model, _, prep = open_clip.create_model_and_transforms(model_name, pretrained=pre)
model.eval()
embs, t0 = [], time.time()
B = 8 if TILES else 32
with torch.no_grad():
    for s in range(0, len(rows), B):
        batch = []
        for r in rows[s:s+B]:
            im = Image.open(idx[r[1]]).convert("RGB"); w, h = im.size
            crops = [im] + ([im.crop((x0*w//2, y0*h//2, (x0+1)*w//2, (y0+1)*h//2)) for y0 in (0, 1) for x0 in (0, 1)] if TILES else [])
            batch += [prep(c) for c in crops]
        e = model.encode_image(torch.stack(batch)); e = e / e.norm(dim=-1, keepdim=True)
        k = 5 if TILES else 1
        e = e.numpy().astype(np.float32).reshape(-1, k, e.shape[-1])
        embs.append(np.concatenate([e[:, 0], e[:, 1:].max(1)], 1) if TILES else e[:, 0])
        if s % 640 == 0: print(s, f"{(time.time()-t0)/(s+B):.3f}s/img", flush=True)
dt = (time.time() - t0) / max(1, len(rows))
np.savez(out, emb=np.concatenate(embs), rid=np.array([r[0] for r in rows]), key=np.array([r[1] for r in rows]),
         cam=np.array([r[2] for r in rows]), ts=np.array([r[3] for r in rows]), fields=np.array([r[4] for r in rows]),
         sec_per_img=dt)
print("done", len(rows), f"{dt:.3f}s/img")
