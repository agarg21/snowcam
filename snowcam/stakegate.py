"""Clean-board gate for snow stakes (2026-11-26), in shadow mode.

A fixed region around each stake's lower ruler and board is compared with that cam's own
bare-board frames (normalised cross-correlation of grey levels and edges). A close match means
the board is clean: 0 in. of snow. Bare-board references are added whenever Sonnet reads the
stake as 0 in. with confidence.

Lab test on the saved Sonnet stake readings (stakelab, Nov 15-26, train on each cam's earlier
days, test on its last 40%): 0 of 14 frames with snow on the board were called clean, and 50 of
79 confident 0 in. frames were caught. Shadow mode logs the gate next to every Sonnet stake read;
SKIP stays off until the log shows at least 30 snow-on-board frames and none called clean.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageFilter

ROIS = yaml.safe_load((Path(__file__).parent / "stake_rois.yaml").read_text()) or {}
SIZE = (48, 96)
TAU = 0.80          # lab: highest score of a snow-on-board frame was 0.754
MARGIN = 0.02       # per cam: also above every logged snow-on-board frame
KEEP_REFS = 30
SKIP = False        # shadow mode; see docstring for the switch-on rule
MIN_SNOW_SEEN = 30

SCHEMA = """
create table if not exists stake_refs (id integer primary key, cam text not null, night integer not null,
  ts text not null, vec blob not null);
create table if not exists stake_gate_log (id integer primary key, cam text not null, ts text not null,
  frame_id integer, reading_id integer, score real, tau real, clean integer,
  sonnet_inches real, sonnet_conf real);
"""


def ensure(con) -> None:
    con.executescript(SCHEMA)


def vec(path: str | Path, roi) -> np.ndarray:
    im = Image.open(path).convert("L")
    w, h = im.size
    x0, y0, x1, y1 = roi
    c = im.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))).resize(SIZE, Image.BILINEAR)
    g = np.asarray(c, dtype=np.float32)
    e = np.asarray(c.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    norm = lambda a: (a - a.mean()) / (a.std() + 1e-6)
    return (np.concatenate([norm(g).ravel(), norm(e).ravel()]) / np.sqrt(2 * g.size)).astype(np.float32)


def tau_for(con, cam: str) -> float:
    m = con.execute("select max(score) from stake_gate_log where cam=? and sonnet_conf>=0.6 and sonnet_inches>0",
                    (cam,)).fetchone()[0]
    return max(TAU, (m + MARGIN) if m is not None else TAU)


def score(con, cam: str, path: str, night: bool) -> tuple[float | None, float]:
    """Best match against the cam's bare-board references, and the threshold for this cam."""
    if cam not in ROIS:
        return None, TAU
    refs = [np.frombuffer(r[0], dtype=np.float32) for r in con.execute(
        "select vec from stake_refs where cam=? and night=? order by id desc limit ?", (cam, int(night), KEEP_REFS))]
    if not refs:
        return None, tau_for(con, cam)
    v = vec(path, ROIS[cam])
    return float(np.max(np.stack(refs) @ v)), tau_for(con, cam)


def record(con, cam: str, ts: str, frame_id, reading_id, path: str, night: bool, fields: dict) -> None:
    """Log the gate next to a Sonnet stake read; keep confident 0 in. frames as references."""
    if cam not in ROIS:
        return
    s, tau = score(con, cam, path, night)
    st = fields.get("stake_inches") or {}
    val, conf = st.get("value"), st.get("conf") or 0
    con.execute("insert into stake_gate_log(cam, ts, frame_id, reading_id, score, tau, clean, sonnet_inches, sonnet_conf) "
                "values (?,?,?,?,?,?,?,?,?)", (cam, ts, frame_id, reading_id, s, tau,
                                               None if s is None else int(s >= tau), val, conf))
    if conf >= 0.6 and val is not None and float(val) == 0:
        con.execute("insert into stake_refs(cam, night, ts, vec) values (?,?,?,?)",
                    (cam, int(night), ts, vec(path, ROIS[cam]).tobytes()))
        con.execute("delete from stake_refs where cam=? and night=? and id not in (select id from stake_refs "
                    "where cam=? and night=? order by id desc limit ?)", (cam, int(night), cam, int(night), KEEP_REFS))


def report(con) -> str:
    ensure(con)
    lines = ["cam                            logged  sonnet0 caught  snow called clean  unsure called clean  refs"]
    tot = [0, 0, 0, 0, 0, 0]
    for (cam,) in con.execute("select distinct cam from stake_gate_log order by cam"):
        rows = con.execute("select clean, sonnet_inches, sonnet_conf from stake_gate_log where cam=?", (cam,)).fetchall()
        z = [r for r in rows if (r[2] or 0) >= 0.6 and r[1] == 0]
        sn = [r for r in rows if (r[2] or 0) >= 0.6 and (r[1] or 0) > 0]
        un = [r for r in rows if (r[2] or 0) < 0.6]
        refs = con.execute("select count(*) from stake_refs where cam=?", (cam,)).fetchone()[0]
        c = [len(rows), sum(r[0] == 1 for r in z), len(z), sum(r[0] == 1 for r in sn), len(sn), sum(r[0] == 1 for r in un)]
        tot = [a + b for a, b in zip(tot, c)]
        lines.append(f"{cam:30s} {c[0]:6d}  {c[1]:4d}/{c[2]:<4d}       {c[3]:4d}/{c[4]:<4d}          {c[5]:4d}/{len(un):<4d}       {refs}")
    lines.append(f"{'ALL':30s} {tot[0]:6d}  {tot[1]:4d}/{tot[2]:<4d}       {tot[3]:4d}/{tot[4]:<4d}          {tot[5]:4d}")
    ready = tot[4] >= MIN_SNOW_SEEN and tot[3] == 0
    lines.append(f"skip switch: {'ready to turn on' if ready else 'stay in shadow'} "
                 f"(needs {MIN_SNOW_SEEN}+ snow-on-board frames, none called clean; SKIP={SKIP})")
    return "\n".join(lines)
