"""Daytime frame archive for the cheaper-reader dataset (owner, 2026-11-25).

Every daytime frame the reader read is saved as a ~640 px JPEG (snow stakes at full size) next to a label JSON: resort, cam,
time, every field with its confidence, and the second reader's check when there is one. Each pass
writes one part zip; the workflow attaches it to that day's release in the private repo
agarg21/snowcam-lab and, the next morning, merges a day's parts into one zip. Nothing here goes to
the public feed.

The day is the US Eastern date, so a whole US ski day (7am ET to dusk on the West Coast) lands in
one zip.
"""
from __future__ import annotations

import csv
import io
import json
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from PIL import Image

from .gates import SUN_DARK_DEG

WIDTH = 640
QUALITY = 82
DAY_TZ = ZoneInfo("America/New_York")
LABEL_VERSION = 1


def day_of(ts_iso: str) -> str:
    return datetime.fromisoformat(ts_iso).astimezone(DAY_TZ).date().isoformat()


def _small(path: Path, full: bool = False) -> bytes:
    im = Image.open(path).convert("RGB")
    if im.width > WIDTH and not full:
        im = im.resize((WIDTH, round(im.height * WIDTH / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=QUALITY, optimize=True)
    return buf.getvalue()


def label(reg, con, r, check) -> dict:
    cam = reg.cam(r["cam"])
    gates = json.loads(r["gates"] or "{}")
    ts = datetime.fromisoformat(r["ts"])
    out = {
        "label_version": LABEL_VERSION,
        "resort": cam.resort_id,
        "resort_name": cam.resort_name,
        "cam": cam.id,
        "cam_name": cam.name,
        "role": cam.role,
        "view": cam.describe,
        "time_utc": ts.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "time_local": ts.astimezone(ZoneInfo(cam.tz)).isoformat(timespec="seconds"),
        "frame_id": r["frame_id"],
        "reading_id": r["id"],
        "teacher": {"model": r["model"], "prompt_version": r["prompt_version"],
                    "fields": json.loads(r["fields"] or "{}")},
        "gates": {"status": gates.get("status"), "metrics": gates.get("metrics", {})},
        "check": None,
    }
    if check is not None:
        res = json.loads(check["tiebreak"] or "{}")
        out["check"] = {
            "phase": check["phase"],
            "model": check["model"],
            "fields": json.loads(check["fields"] or "{}"),
            # True/False per field; stake_inches holds the difference in inches instead
            "agrees": res.get("compare", {}),
            "false_fresh": bool(res.get("false_fresh")),
            # spot checks: the cheap checker disagreed, so "fields" is the Sonnet tie-break and
            # this is what the cheap checker said
            "first_check_fields": res.get("spot_fields"),
        }
    return out


def build_part(reg, con, since_iso: str, out_dir: Path, tag: str) -> list[Path]:
    """Write one part zip per Eastern day for this pass's daytime reads. Returns the zips."""
    rows = con.execute(
        "select r.id, r.frame_id, r.cam, r.ts, r.model, r.prompt_version, r.fields, f.path, f.gates "
        "from readings r join frames f on f.id=r.frame_id where r.ts>=? and f.path is not null order by r.ts",
        (since_iso,)).fetchall()
    zips: dict[str, zipfile.ZipFile] = {}
    paths: dict[str, Path] = {}
    n = 0
    for r in rows:
        m = json.loads(r["gates"] or "{}").get("metrics", {})
        if (m.get("sun_elevation") if m.get("sun_elevation") is not None else -90) < SUN_DARK_DEG:
            continue  # daytime frames only (lit snow stakes at night are left out)
        src = Path(r["path"])
        if not src.exists():
            continue
        try:
            reg.cam(r["cam"])
        except KeyError:
            continue
        check = con.execute("select * from verifications where reading_id=? order by id desc limit 1",
                            (r["id"],)).fetchone()
        lab = label(reg, con, r, check)
        day = day_of(r["ts"])
        if day not in zips:
            out_dir.mkdir(parents=True, exist_ok=True)
            paths[day] = out_dir / f"snowcam-{day}-part-{tag}.zip"
            zips[day] = zipfile.ZipFile(paths[day], "w", zipfile.ZIP_STORED)  # JPEGs do not compress
        stem = f"{lab['resort']}/{lab['cam']}/{lab['time_utc'].replace(':', '').replace('+0000', 'Z')}"
        zips[day].writestr(f"{stem}.jpg", _small(src, full=lab["role"] == "stake"))  # rulers need full size
        zips[day].writestr(f"{stem}.json", json.dumps(lab, indent=1))
        n += 1
    for z in zips.values():
        z.close()
    print(f"archive: {n} daytime frames in {len(zips)} part zip(s)")
    return list(paths.values())


MANIFEST_FIELDS = ["fresh_snow", "snow_cover", "visibility", "precip_now", "snowmaking", "lifts_moving",
                   "base_busyness", "lift_line", "stake_inches", "stake_new_snow"]


def merge(parts: list[Path], out: Path) -> int:
    """Merge a day's part zips into one zip with a manifest.csv (one row per frame)."""
    seen: set[str] = set()
    rows = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for p in sorted(parts):
            with zipfile.ZipFile(p) as src:
                for name in src.namelist():
                    if name in seen or name == "manifest.csv":
                        continue
                    seen.add(name)
                    data = src.read(name)
                    z.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED if name.endswith(".json") else zipfile.ZIP_STORED)
                    if name.endswith(".json"):
                        lab = json.loads(data)
                        f = lab["teacher"]["fields"]
                        row = {"image": name[:-5] + ".jpg", "resort": lab["resort"], "cam": lab["cam"],
                               "role": lab["role"], "time_utc": lab["time_utc"], "teacher_model": lab["teacher"]["model"],
                               "checked": lab["check"] is not None}
                        for k in MANIFEST_FIELDS:
                            v = f.get(k)
                            if not isinstance(v, dict):  # old prompts sometimes gave a bare value (no confidence)
                                v = {"value": v, "conf": None} if v is not None else {}
                            row[k] = v.get("value")
                            row[k + "_conf"] = v.get("conf")
                        rows.append(row)
        buf = io.StringIO()
        if rows:
            w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        z.writestr("manifest.csv", buf.getvalue(), compress_type=zipfile.ZIP_DEFLATED)
    print(f"merged {len(parts)} parts -> {out.name}: {len(rows)} frames")
    return len(rows)


def savings(con, days: int = 7) -> str:
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    lines = ["day         reads  skipped  same  dark  blurry  est. saved"]
    for d, in con.execute("select distinct substr(ts,1,10) from skips where ts>=? order by 1", (since,)):
        reads = con.execute("select count(*) from readings where substr(ts,1,10)=?", (d,)).fetchone()[0]
        by = dict(con.execute("select reason, count(*) from skips where substr(ts,1,10)=? group by reason", (d,)).fetchall())
        cost = con.execute("select coalesce(sum(est_cost),0) from skips where substr(ts,1,10)=?", (d,)).fetchone()[0]
        tot = sum(by.values())
        lines.append(f"{d}  {reads:5d}  {tot:5d} ({tot / max(1, tot + reads):.0%})  {by.get('same', 0):4d}  "
                     f"{by.get('dark', 0):4d}  {by.get('blurry', 0):6d}  ${cost:.2f}")
    return "\n".join(lines)
