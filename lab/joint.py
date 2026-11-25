"""Hybrid: how many model calls a cheap student could skip at >=95% per-field agreement with the teacher.
Train on early days, pick per-field confidence thresholds on the next days (validation), measure on the last 2 days."""
import sys, json
import numpy as np
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, "/tmp/labscripts")
emb_path = sys.argv[1]
d = np.load(emb_path)
E, cams, ts, F = d["emb"], d["cam"], d["ts"], [json.loads(x) for x in d["fields"]]
mean = {c: E[cams == c].mean(0) for c in set(cams)}
X = np.hstack([E, E - np.stack([mean[c] for c in cams])])
day = np.array([t[:10] for t in ts]); days = sorted(set(day))
test = day >= days[-2]; val = (day >= days[-4]) & ~test; train = day < days[-4]
MAPS = {
    "snow_cover": {"full": "good", "mostly": "good", "patchy": "thin", "bare": "thin"},
    "fresh_snow": {"none": "none", "dusting": "some", "light": "some", "heavy": "some"},
    "snowmaking": {"on": "on", "off": "off"},
    "precip_now": {"none": "none", "snow": "snow"},
    "visibility": {"clear": "clear", "hazy": "hazy", "fog": "poor", "whiteout": "poor"},
}
def lab(fd, f):
    v = fd.get(f, {})
    return MAPS[f].get(v.get("value")) if isinstance(v, dict) and v.get("conf", 0) >= 0.6 else None
TARGET = float(sys.argv[2]) if len(sys.argv) > 2 else 0.95
confident = np.ones(len(F), bool); hits = {}; report = {}
for f in MAPS:
    y = np.array([lab(fd, f) or "" for fd in F]); ok = y != ""
    clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(X[train & ok], y[train & ok])
    p = clf.predict_proba(X); pred = clf.classes_[p.argmax(1)]; conf = p.max(1)
    # threshold: smallest confidence whose answers on validation agree >= TARGET
    v = val & ok; order = np.argsort(-conf[v]); hit = (pred[v] == y[v])[order]
    cum = np.cumsum(hit) / np.arange(1, len(hit) + 1)
    good = np.where(cum >= TARGET)[0]
    thr = conf[v][order][good.max()] if len(good) else 1.01
    # frames where the teacher had no usable answer still need the student to be confident-or-skip:
    # count the student as confident on them only above the threshold too
    sure = conf >= thr
    confident &= sure
    t = test & ok
    report[f] = dict(threshold=round(float(thr), 3), test_coverage=round(float(sure[t].mean()), 3),
                     test_agreement_when_sure=round(float((pred[t & sure] == y[t & sure]).mean()), 3) if (t & sure).any() else None,
                     test_n=int(t.sum()))
    hits[f] = (pred, y, ok)
skip = confident & test
report["_joint"] = dict(test_frames=int(test.sum()), all_fields_sure=round(float(skip.sum() / test.sum()), 3),
                        agreement_on_skipped={f: round(float((hits[f][0][skip & hits[f][2]] == hits[f][1][skip & hits[f][2]]).mean()), 3)
                                              for f in MAPS if (skip & hits[f][2]).any()})
print(json.dumps(report, indent=1))
