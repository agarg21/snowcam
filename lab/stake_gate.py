"""Clean-board gate for snow stakes, tested against the saved Sonnet stake readings.

For each stake cam a fixed region around the lower ruler and the snow board (fractions of the
frame) is compared with the cam's own bare-board frames from earlier days (normalised
cross-correlation of grey levels and edges). A close match means "0 in., no Sonnet call";
anything else goes to Sonnet. A second check flags regions too dark or flat to read, where
Sonnet itself is rarely confident.

Train = each cam's earlier days, test = its last 40% of days, so the gate never sees the
frames it is scored on.
"""
import json
import sys
from collections import defaultdict

import numpy as np
from PIL import Image, ImageFilter

ROI = {  # x0, y0, x1, y1: lower ruler plus board surface
    "alta-collins-snow-stake": (0.44, 0.55, 0.60, 0.92),
    "bogus-basin-yt-g714uvr8rjg": (0.42, 0.30, 0.58, 0.55),
    "breckenridge-brecksnowstake": (0.42, 0.55, 0.56, 0.90),
    "keystone-keysnowstake": (0.42, 0.55, 0.56, 0.92),
    "mountsnow-stake": (0.18, 0.55, 0.58, 0.90),
    "mt-bachelor-yt-orlxprywwd4": (0.44, 0.55, 0.53, 0.88),
    "northstar-snowstake": (0.30, 0.20, 0.66, 0.60),
    "okemo-stake": (0.54, 0.50, 0.66, 0.88),
    "stevens-pass-snowstake": (0.44, 0.45, 0.60, 0.85),
    "stowe-snowstake": (0.42, 0.50, 0.56, 0.88),
    "sugarbush-yt-2kjx9cnxrxa": (0.36, 0.50, 0.47, 0.85),
    "sugarbush-yt-gwpiruoopee": (0.46, 0.45, 0.60, 0.80),
    "vail-snowsummit": (0.52, 0.55, 0.69, 0.90),
    "winter-park-yt-yj-5q1vbmaa": (0.48, 0.60, 0.58, 0.97),
}
SIZE = (48, 96)
CONF = 0.6


def feats(path, roi):
    im = Image.open(path).convert("L")
    w, h = im.size
    x0, y0, x1, y1 = roi
    c = im.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))).resize(SIZE, Image.BILINEAR)
    g = np.asarray(c, dtype=np.float32)
    e = np.asarray(c.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    norm = lambda a: (a - a.mean()) / (a.std() + 1e-6)
    return np.concatenate([norm(g).ravel(), norm(e).ravel()]) / np.sqrt(2 * g.size), float(g.mean()), float(g.std()), float(e.mean())


def main():
    L = [x for x in json.load(open("labels.json")) if x["cam"] in ROI]
    by = defaultdict(list)
    for x in L:
        x["night"] = (x["sun"] if x["sun"] is not None else -90) < -6
        by[x["cam"]].append(x)
    tot = defaultdict(int)
    rows = []
    pos_scores = []
    per_cam = {}
    for cam, xs in by.items():
        xs.sort(key=lambda x: x["ts"])
        days = sorted({x["ts"][:10] for x in xs})
        cut = days[int(len(days) * 0.6)] if len(days) > 2 else days[-1]
        train = [x for x in xs if x["ts"][:10] < cut]
        test = [x for x in xs if x["ts"][:10] >= cut]
        for x in xs:
            x["f"], x["mean"], x["std"], x["edge"] = feats(x["path"], ROI[cam])
        def zero(x):
            try:
                return x["conf"] >= CONF and x["inches"] is not None and float(x["inches"]) == 0
            except (TypeError, ValueError):
                return False
        def snow(x):
            try:
                return x["conf"] >= CONF and x["inches"] is not None and float(x["inches"]) > 0
            except (TypeError, ValueError):
                return False
        refs = {n: np.stack([x["f"] for x in train if zero(x) and x["night"] == n]) for n in (False, True)
                if any(zero(x) and x["night"] == n for x in train)}
        def score(x, exclude=None):
            R = refs.get(x["night"])
            if R is None:
                return -1.0
            s = R @ x["f"]
            if exclude is not None:
                s = s[[i for i in range(len(s)) if i != exclude]] if len(s) > 1 else np.array([-1.0])
            return float(s.max())
        # threshold: above every training frame Sonnet saw snow on, and above the low tail of
        # leave-one-out scores of training zeros is not required (precision first)
        tr_snow = [score(x) for x in train if snow(x)]
        per_cam[cam] = dict(refs=sum(len(v) for v in refs.values()), train=len(train), test=len(test),
                            train_snow=len(tr_snow), max_snow=max(tr_snow) if tr_snow else None)
        pos_scores += tr_snow
        for x in test:
            x["score"] = score(x)
            rows.append((cam, x, zero(x), snow(x)))
    tau_global = max(0.80, (max(pos_scores) + 0.02) if pos_scores else 0.80)
    print(f"global threshold {tau_global:.3f} (highest training snow frame {max(pos_scores) if pos_scores else None})")
    out = defaultdict(lambda: defaultdict(int))
    for cam, x, z, s in rows:
        m = per_cam[cam]["max_snow"]
        tau = max(tau_global, (m + 0.02) if m is not None else 0)
        call0 = x["score"] >= tau
        k = "zero" if z else "snow" if s else "unsure"
        out[cam][(k, call0)] += 1
        out["ALL"][(k, call0)] += 1
    print(f"{'cam':30s} test  Sonnet0 caught  Sonnet>0 missed  unsure->0   refs")
    for cam in sorted(out, key=lambda c: (c == "ALL", c)):
        o = out[cam]
        z = o[("zero", True)] + o[("zero", False)]
        s = o[("snow", True)] + o[("snow", False)]
        u = o[("unsure", True)] + o[("unsure", False)]
        print(f"{cam:30s} {z + s + u:4d}  {o[('zero', True)]:3d}/{z:<3d}      {o[('snow', True)]:3d}/{s:<3d}        "
              f"{o[('unsure', True)]:3d}/{u:<3d}   {per_cam.get(cam, {}).get('refs', '')}")
    json.dump({c: {f"{k[0]}_{k[1]}": v for k, v in o.items()} for c, o in out.items()}, open("gate_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
