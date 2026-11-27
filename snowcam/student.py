"""Student reader (shadow mode, 2026-11-27): a frozen image model (CLIP or DINOv2) plus small per-field
layers trained on our own labelled frames (labels: Sonnet checks, else confident first reads).

v3 (11-27): snow cover and snowmaking read DINOv2-small features, the other fields CLIP B/32;
snow cover is trained on Sonnet labels only. Lab, random Sonnet checks on cams it has seen, two
test splits: snow cover 97% and 93% (CLIP-only v1: 78% and 78%; Haiku on the same frames: 53%).

It runs on the runner's CPU beside the Haiku first read and logs its answers in student_preds;
`snowcam student-report` scores both against the same random Sonnet checks. It publishes
nothing until that report shows it holding up (lab test 11-27: on identical random-check frames
vs Sonnet it beat Haiku on snow cover 73% vs 53%, fresh snow 92% vs 80%, visibility 91% vs 84%).

Model file: models/student-<version>.npz (heads, per-cam mean embeddings, confidence thresholds),
written by lab/student_export.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "student-v3.npz"
# Frozen image model, stored in the model file as "open_clip:<name>:<pretrained>" or "timm:<name>".
DEFAULT_BACKBONE = "open_clip:ViT-B-32-quickgelu:openai"


def load_backbone(spec: str):
    """(encode, preprocess) for a backbone spec; lab/student_embed.py uses the same function so
    training and production embeddings match exactly."""
    kind, _, rest = spec.partition(":")
    if kind == "timm":
        import timm
        model = timm.create_model(rest, pretrained=True, num_classes=0, img_size=224)
        cfg = timm.data.resolve_data_config({}, model=model)
        cfg.update(input_size=(3, 224, 224), crop_pct=1.0)
        prep = timm.data.create_transform(**cfg)
        encode = model.forward
    elif kind == "open_clip":
        import open_clip
        name, _, pretrained = rest.partition(":")
        model, _, prep = open_clip.create_model_and_transforms(name, pretrained=pretrained)
        encode = model.encode_image
    else:
        raise ValueError(f"unknown backbone {spec!r}")
    model.eval()
    return encode, prep

SCHEMA = """
create table if not exists student_preds (
  id integer primary key, reading_id integer unique, frame_id integer, cam text not null, ts text not null,
  version text not null, fields text not null, sure text not null
);
"""


class Student:
    def __init__(self, path: Path = MODEL):
        z = np.load(path, allow_pickle=True)
        self.version = str(z["version"])
        self.fields = [str(f) for f in z["fields"]]
        import torch
        self.torch = torch
        cams = [str(c) for c in z["cams"]]
        if "encoders" in z:  # v3+: one or more encoders, each field reads one of them
            n = len(z["encoders"])
            specs = [str(z[f"enc{i}_backbone"]) for i in range(n)]
            means = [(dict(zip(cams, z[f"enc{i}_cam_means"])), z[f"enc{i}_global_mean"]) for i in range(n)]
            enc_of = {f: int(z[f"{f}_encoder"]) for f in self.fields}
        else:  # v1/v2: a single encoder
            specs = [str(z["backbone"]) if "backbone" in z else DEFAULT_BACKBONE]
            means = [(dict(zip(cams, z["cam_means"])), z["global_mean"])]
            enc_of = {f: 0 for f in self.fields}
        self.backbones = specs
        self.encoders = [load_backbone(s) for s in specs]
        self.means = means
        self.heads = {f: (enc_of[f], z[f"{f}_coef"], z[f"{f}_intercept"], [str(c) for c in z[f"{f}_classes"]],
                          float(z[f"{f}_threshold"])) for f in self.fields}

    def embed(self, im) -> list[np.ndarray]:
        """One feature vector per encoder: the whole frame plus the max over its four tiles."""
        w, h = im.size
        tiles = [im.crop((x * w // 2, y * h // 2, (x + 1) * w // 2, (y + 1) * h // 2)) for y in (0, 1) for x in (0, 1)]
        out = []
        with self.torch.no_grad():
            for encode, prep in self.encoders:
                e = encode(self.torch.stack([prep(t) for t in [im] + tiles]))
                e = (e / e.norm(dim=-1, keepdim=True)).numpy().astype(np.float32)
                out.append(np.concatenate([e[0], e[1:].max(0)]))
        return out

    def predict(self, cam_id: str, embs: list[np.ndarray]) -> tuple[dict, dict]:
        out, sure = {}, {}
        for f, (k, coef, b, classes, thr) in self.heads.items():
            cam_means, glob = self.means[k]
            x = np.concatenate([embs[k], embs[k] - cam_means.get(cam_id, glob)])
            z = coef @ x + b
            if len(classes) == 2:  # sklearn binary: one row, positive class is classes[1]
                p1 = 1 / (1 + np.exp(-z[0]))
                p = np.array([1 - p1, p1])
            else:
                p = np.exp(z - z.max())
                p = p / p.sum()
            i = int(p.argmax())
            out[f] = {"value": classes[i], "conf": round(float(p[i]), 3)}
            sure[f] = bool(p[i] >= thr)
        return out, sure


def shadow(reg, con, since_iso: str) -> int:
    """Run the student on every daytime frame read since since_iso and log its answers."""
    from PIL import Image
    con.executescript(SCHEMA)
    rows = con.execute(
        "select r.id, r.frame_id, r.cam, r.ts, f.path, f.gates from readings r join frames f on f.id=r.frame_id "
        "where r.ts>=? and f.path is not null and r.id not in (select reading_id from student_preds)", (since_iso,)).fetchall()
    todo = []
    for r in rows:
        try:
            cam = reg.cam(r["cam"])
        except KeyError:
            continue
        sun = (json.loads(r["gates"] or "{}").get("metrics", {}) or {}).get("sun_elevation")
        if cam.role != "stake" and Path(r["path"]).exists() and (sun if sun is not None else -90) >= -6:
            todo.append(r)
    if not todo:
        print("student: nothing to read")
        return 0
    if not MODEL.exists():
        print(f"student: no model at {MODEL}")
        return 0
    s = Student()
    for r in todo:
        im = Image.open(r["path"]).convert("RGB")
        if im.width > 640:  # trained on the 640 px archive copies
            im = im.resize((640, round(im.height * 640 / im.width)), Image.LANCZOS)
        fields, sure = s.predict(r["cam"], s.embed(im))
        con.execute("insert or ignore into student_preds(reading_id, frame_id, cam, ts, version, fields, sure) "
                    "values (?,?,?,?,?,?,?)", (r["id"], r["frame_id"], r["cam"], r["ts"], s.version,
                                               json.dumps(fields), json.dumps(sure)))
    con.commit()
    print(f"student {s.version}: read {len(todo)} frames in shadow mode")
    return len(todo)


SCALE = {
    "snow_cover": {"full": "good", "mostly": "good", "patchy": "thin", "bare": "thin"},
    "fresh_snow": {"none": "none", "dusting": "some", "light": "some", "heavy": "some"},
    "visibility": {"clear": "clear", "hazy": "hazy", "fog": "poor", "whiteout": "poor"},
    "snowmaking": {"on": "on", "off": "off"},
    "precip_now": {"none": "none", "snow": "snow", "rain_or_mix": "wet"},
}


def _lab(fields: dict | None, f: str):
    v = (fields or {}).get(f)
    if not isinstance(v, dict) or (v.get("conf") or 0) < 0.6:
        return None
    return SCALE[f].get(v.get("value"))


def report(con, days: int = 14) -> str:
    """Student vs Haiku first read, both against the random Sonnet checks on the same frames."""
    con.executescript(SCHEMA)
    rows = con.execute(
        "select p.fields sf, p.sure, r.fields ff, v.fields vf from student_preds p join readings r on r.id=p.reading_id "
        "join verifications v on v.reading_id=p.reading_id where v.phase='initial' and v.model like 'claude-sonnet%' "
        "and p.ts >= datetime('now', ?)", (f"-{days} days",)).fetchall()
    lines = [f"student vs Haiku against random Sonnet checks, last {days} days ({len(rows)} checked frames)",
             "field         student   Haiku    | student when sure (share of frames)"]
    for f in SCALE:
        s_ok = h_ok = n = g_ok = g_n = 0
        for r in rows:
            sv = json.loads(r["sf"]).get(f, {}).get("value")
            gold = _lab(json.loads(r["vf"]), f)
            hv = _lab(json.loads(r["ff"]), f)
            if gold is None or hv is None or sv is None:
                continue
            n += 1
            s_ok += sv == gold
            h_ok += hv == gold
            if json.loads(r["sure"]).get(f):
                g_n += 1
                g_ok += sv == gold
        if n:
            lines.append(f"{f:12s}  {s_ok / n:6.0%}   {h_ok / n:6.0%}    | {g_ok}/{g_n} ({g_n / n:.0%})   n={n}")
    return "\n".join(lines)
