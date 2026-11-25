"""Linear probes on frame embeddings vs the teacher reader. Usage: python -I probe.py EMB.npz DB REGISTRY"""
import sys, json, sqlite3, re
from collections import Counter, defaultdict
import numpy as np, yaml
from sklearn.linear_model import LogisticRegression
emb_path, db, regp = sys.argv[1:4]
d = np.load(emb_path, allow_pickle=False)
E, cams, ts, F = d["emb"], d["cam"], d["ts"], [json.loads(x) for x in d["fields"]]
reg = yaml.safe_load(open(regp))
cam2res = {c["id"]: rid for rid, r in reg["resorts"].items() for c in r["cams"]}
cam2role = {c["id"]: c["role"] for r in reg["resorts"].values() for c in r["cams"]}
res = np.array([cam2res.get(c, "?") for c in cams])
day = np.array([t[:10] for t in ts])
# cam-centred features: subtract each cam's mean embedding (unsupervised; available at deploy time)
mean = {c: E[cams == c].mean(0) for c in set(cams)}
X = np.hstack([E, E - np.stack([mean[c] for c in cams])])
MAPS = {
    "snow_cover": {"full": "good", "mostly": "good", "patchy": "thin", "bare": "thin"},
    "fresh_snow": {"none": "none", "dusting": "some", "light": "some", "heavy": "some"},
    "snowmaking": {"on": "on", "off": "off"},
    "precip_now": {"none": "none", "snow": "snow"},
    "visibility": {"clear": "clear", "hazy": "hazy", "fog": "poor", "whiteout": "poor"},
    "base_busyness": {"empty": "quiet", "light": "quiet", "normal": "normal+", "busy": "normal+", "packed": "normal+"},
}
# Sonnet second opinions for the same reads, mapped the same way
con = sqlite3.connect(db)
second = {rid: json.loads(f) for rid, f in con.execute("select reading_id, fields from verifications where model like '%sonnet%'")}
rids = d["rid"]
def lab(fd, f):
    v = fd.get(f, {}); val = v.get("value") if isinstance(v, dict) else None
    conf = v.get("conf", 0) if isinstance(v, dict) else 0
    return MAPS[f].get(val) if conf >= 0.6 else None
out = {}
for f in MAPS:
    y = np.array([lab(fd, f) or "" for fd in F]); ok = y != ""
    if f == "base_busyness": ok &= np.array([cam2role.get(c) == "base" for c in cams])
    classes = Counter(y[ok]); 
    if len(classes) < 2: continue
    major = classes.most_common(1)[0][1] / ok.sum()
    def fit_eval(tr, te):
        clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(X[tr], y[tr])
        p = clf.predict_proba(X[te]); pred = clf.classes_[p.argmax(1)]; conf = p.max(1)
        return pred, conf
    # A) leave-resort-out: a new resort the student has never seen
    preds = np.empty(len(y), dtype=object); confs = np.zeros(len(y))
    for r in sorted(set(res[ok])):
        te = ok & (res == r); tr = ok & (res != r)
        if len(set(y[tr])) < 2: continue
        p, c = fit_eval(tr, te); preds[te] = p; confs[te] = c
    m = ok & (preds != None)
    lro = (preds[m] == y[m]).mean()
    rec = {str(k): round(float((preds[m & (y == k)] == k).mean()), 3) for k in classes}
    bal = float(np.mean(list(rec.values())))
    # B) time split: train on reads before the last 2 days, test on the last 2 days (known cams)
    days = sorted(set(day)); cut = days[-2]
    tr, te = ok & (day < cut), ok & (day >= cut)
    p, c = fit_eval(tr, te); ts_acc = (p == y[te]).mean()
    # coverage at 95% agreement: confident student answers, the rest go to the model
    order = np.argsort(-c); hit = (p == y[te])[order]
    cum = np.cumsum(hit) / np.arange(1, len(hit) + 1)
    cov95 = (np.where(cum >= 0.95)[0].max() + 1) / len(hit) if (cum >= 0.95).any() else 0
    # teacher vs Sonnet second opinion on the same reads (the bar the teacher itself meets)
    pairs = [(lab(F[i], f), lab(second[rids[i]], f)) for i in range(len(F)) if rids[i] in second]
    pairs = [(a, b) for a, b in pairs if a and b]
    tvs = np.mean([a == b for a, b in pairs]) if pairs else float("nan")
    out[f] = dict(n=int(ok.sum()), classes={str(k): int(v) for k, v in classes.items()}, majority=round(float(major), 3), leave_resort_out=round(float(lro), 3), lro_recall=rec, lro_balanced=round(bal, 3),
                  time_split=round(float(ts_acc), 3), time_split_n=int(te.sum()), coverage_at_95=round(float(cov95), 3),
                  teacher_vs_sonnet=round(float(tvs), 3), sonnet_pairs=len(pairs))
    print(f, "bal", out[f]["lro_balanced"], out[f]["lro_recall"], "acc", out[f]["leave_resort_out"], "time", out[f]["time_split"], "cov95", out[f]["coverage_at_95"], "maj", out[f]["majority"], "t-v-s", out[f]["teacher_vs_sonnet"], flush=True)
json.dump(dict(results=out, n_frames=int(len(E)), n_resorts=len(set(res)), n_cams=len(set(cams)),
               days=[min(day.tolist()), max(day.tolist())], sec_per_img=float(d["sec_per_img"])), open(emb_path + ".results.json", "w"), indent=1)
