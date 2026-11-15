"""SQLite + frame files. Frames are private working data, never served publicly."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from PIL import Image

from .config import ROOT

DATA = ROOT / "data"
FRAMES = DATA / "frames"
REFS = DATA / "refs"
DB_PATH = DATA / "snowcam.db"
STORE_MAX_W = 1280  # downscale on save; enough for the reader, keeps storage small

SCHEMA = """
create table if not exists frames (
  id integer primary key,
  cam text not null,
  ts text not null,
  path text,
  phash text,
  width integer, height integer,
  gate_status text,
  gates json,
  error text
);
create index if not exists frames_cam_ts on frames(cam, ts);
create table if not exists readings (
  id integer primary key,
  frame_id integer references frames(id),
  cam text, ts text,
  model text, prompt_version text,
  fields json, notes text, raw json,
  created text
);
create index if not exists readings_cam_ts on readings(cam, ts);
create table if not exists resort_state (
  id integer primary key,
  resort text, ts text,
  verdict text, line text, fused json
);
create table if not exists cam_profiles (
  cam text primary key, profile json, model text, created text
);
create table if not exists labels (
  id integer primary key,
  frame_id integer references frames(id),
  field text, value text, labeler text, created text
);
"""


def connect() -> sqlite3.Connection:
    DATA.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def save_frame_image(cam_id: str, ts: datetime, im: Image.Image, suffix: str = "") -> Path:
    d = FRAMES / cam_id / ts.strftime("%Y%m%d")
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{ts.strftime('%H%M%S')}{suffix}.jpg"
    if im.width > STORE_MAX_W:
        im = im.resize((STORE_MAX_W, int(im.height * STORE_MAX_W / im.width)))
    im.save(p, quality=85)
    return p


def insert_frame(con, cam_id, ts, path, phash, size, gate_status, gates, error=None) -> int:
    cur = con.execute(
        "insert into frames(cam, ts, path, phash, width, height, gate_status, gates, error) values (?,?,?,?,?,?,?,?,?)",
        (cam_id, ts.isoformat(), str(path) if path else None, phash, size[0] if size else None,
         size[1] if size else None, gate_status, json.dumps(gates) if gates else None, error),
    )
    con.commit()
    return cur.lastrowid


def recent_hashes(con, cam_id: str, n: int = 5) -> list[str]:
    rows = con.execute(
        "select phash from frames where cam=? and phash is not null order by ts desc limit ?", (cam_id, n)
    ).fetchall()
    return [r["phash"] for r in rows]


def last_read_frame(con, cam_id: str):
    return con.execute(
        "select f.* , r.fields, r.ts as read_ts from readings r join frames f on f.id=r.frame_id "
        "where r.cam=? order by r.ts desc limit 1", (cam_id,)
    ).fetchone()


def insert_reading(con, frame_id, cam_id, ts, model, prompt_version, fields, notes, raw) -> int:
    cur = con.execute(
        "insert into readings(frame_id, cam, ts, model, prompt_version, fields, notes, raw, created) values (?,?,?,?,?,?,?,?,?)",
        (frame_id, cam_id, ts, model, prompt_version, json.dumps(fields), notes, json.dumps(raw), datetime.now().isoformat()),
    )
    con.commit()
    return cur.lastrowid


def reference_path(cam_id: str) -> Path:
    return REFS / cam_id / "clear.jpg"


def offline_hashes() -> list[str]:
    p = DATA / "offline_hashes.txt"
    return [l.strip() for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
