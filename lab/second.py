"""Does student-vs-teacher disagreement find the frames where Sonnet disagrees with the teacher?"""
import sys, json, sqlite3
import numpy as np, yaml
from sklearn.linear_model import LogisticRegression
d = np.load(sys.argv[1]); db = sys.argv[2]; reg = yaml.safe_load(open(sys.argv[3]))
E, cams, F, rids = d["emb"], d["cam"], [json.loads(x) for x in d["fields"]], d["rid"]
cam2res = {c["id"]: rid for rid, r in reg["resorts"].items() for c in r["cams"]}
res = np.array([cam2res.get(c, "?") for c in cams])
mean = {c: E[cams == c].mean(0) for c in set(cams)}
X = np.hstack([E, E - np.stack([mean[c] for c in cams])])
con = sqlite3.connect(db)
second = {r: json.loads(f) for r, f in con.execute("select reading_id, fields from verifications where model like '%sonnet%'")}
MAPS = {"snow_cover": {"full": "good", "mostly": "good", "patchy": "thin", "bare": "thin"},
        "fresh_snow": {"none": "none", "dusting": "some", "light": "some", "heavy": "some"},
        "snowmaking": {"on": "on", "off": "off"}, "precip_now": {"none": "none", "snow": "snow"},
        "visibility": {"clear": "clear", "hazy": "hazy", "fog": "poor", "whiteout": "poor"}}
def lab(fd, f):
    v = fd.get(f, {})
    return MAPS[f].get(v.get("value")) if isinstance(v, dict) and v.get("conf", 0) >= .6 else None
tot = dict(frames=0, flagged=0, caught=0, errors=0)
for f in MAPS:
    y = np.array([lab(fd, f) or "" for fd in F]); ok = y != ""
    pred = np.empty(len(y), dtype=object)
    for r in sorted(set(res[ok])):  # leave-resort-out predictions, so no frame is judged by a probe trained on it
        te, tr = ok & (res == r), ok & (res != r)
        if len(set(y[tr])) > 1: pred[te] = LogisticRegression(max_iter=3000, class_weight="balanced").fit(X[tr], y[tr]).predict(X[te])
    idx = [i for i in range(len(y)) if ok[i] and rids[i] in second and lab(second[rids[i]], f) and pred[i] is not None]
    t = np.array([y[i] for i in idx]); s = np.array([lab(second[rids[i]], f) for i in idx]); p = np.array([pred[i] for i in idx])
    err = t != s; flag = p != t
    print(f"{f:12s} n={len(idx):5d} teacher-vs-sonnet disagree={err.sum():4d}  student flags={flag.sum():4d} ({flag.mean():.0%})  "
          f"caught={ (err & flag).sum():3d} ({(err & flag).sum() / max(1, err.sum()):.0%} of disagreements)")
