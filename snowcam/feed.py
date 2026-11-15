"""Public text feed for the resort conditions pages (Family Tripwise D47/D48).

Only text leaves the pipeline: status, one line, fields that passed the confidence bar, read
times and the resort's own cam page. No images, no frame paths, nothing about people.
A resort whose `publish` flag is false in registry.yaml is shown as "No read" with no claims,
so pages can be built while its reads are still being checked.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

LABELS = {"Go": "Looks good", "OK": "Fair", "Busy": "Busy", "Thin cover": "Thin cover",
          "Lifts idle": "Lifts not moving", "No read": "No read"}
NO_READ = {"offline": "feed offline", "dark": "dark", "stale": "feed frozen",
           "visibility": "fogged in", "moved": "view moved"}
PUBLIC_FIELDS = ["fresh_snow", "stake_new_snow", "snow_cover", "visibility", "snowmaking",
                 "lifts_moving", "base_busyness", "lift_line", "precip_now"]
VALIDATING = "Cam reads for this resort are still being checked before we publish them."


def _iso(ts: str | None) -> str | None:
    return datetime.fromisoformat(ts).astimezone(timezone.utc).isoformat(timespec="seconds") if ts else None


def ps_changed(con, rid):
    r = con.execute("select changed from publish_state where resort=?", (rid,)).fetchone()
    return r["changed"] if r else None


def build(reg, con, window_min: int = 45, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    since = (now - timedelta(minutes=window_min)).isoformat()
    out = {"generated_at": now.isoformat(timespec="seconds"), "window_minutes": window_min, "resorts": {}}
    for rid, r in reg.resorts.items():
        ps = None
        try:
            ps = con.execute("select publish, fields, summary from publish_state where resort=?", (rid,)).fetchone()
        except Exception:
            pass
        # Owner D49: the automated check decides; registry `publish: true` is a manual override.
        publish = bool(r.get("publish", False)) or bool(ps and ps["publish"])
        allowed = set(json.loads(ps["fields"])) if ps and ps["publish"] and ps["fields"] else None
        st = con.execute("select ts, verdict, line, fused from resort_state where resort=? order by id desc limit 1",
                         (rid,)).fetchone()
        cams, used = [], 0
        for cam in [c for c in reg.by_resort(rid) if c.usable]:
            rd = con.execute("select ts from readings where cam=? and ts>=? order by ts desc limit 1",
                             (cam.id, since)).fetchone()
            fr = con.execute("select ts, gate_status, gates from frames where cam=? and ts>=? order by ts desc limit 1",
                             (cam.id, since)).fetchone()
            no_read = None
            if not rd:
                if not fr:
                    no_read = "no recent frame"
                elif fr["gate_status"] == "fail":
                    checks = json.loads(fr["gates"] or "{}").get("checks", {})
                    failed = [k for k, c in checks.items() if c.get("status") == "fail"]
                    no_read = NO_READ.get(failed[0], "unclear view") if failed else "unclear view"
                else:
                    no_read = "not read yet"
            used += bool(rd)
            cams.append({"name": cam.name, "role": cam.role,
                         "read_at": _iso(rd["ts"]) if rd else None, "no_read": no_read})
        entry = {"publish": publish, "name": r["name"], "cam_page": r.get("cam_page"),
                 "cams_used": used, "cams_total": len(cams), "cams": cams}
        fused = json.loads(st["fused"]) if st and st["fused"] else {}
        if publish and st:
            entry.update(status=st["verdict"], label=LABELS.get(st["verdict"], "No read"), line=st["line"],
                         read_at=_iso(st["ts"]),
                         fields={k: fused[k] for k in PUBLIC_FIELDS
                                 if fused.get(k) is not None and (allowed is None or k in allowed)} if used else {},
                         stake_inches=fused.get("stake_inches") if used and (allowed is None or "stake_inches" in allowed) else None)
            if ps and ps["publish"] and ps["summary"]:
                sm = json.loads(ps["summary"])
                entry["check"] = {"reads": sm.get("reads"), "agreement": sm.get("agreement"), "at": ps_changed(con, rid)}
        else:
            entry.update(status="No read", label="No read",
                         line=None if publish else VALIDATING,
                         read_at=None, fields={}, stake_inches=None)
        out["resorts"][rid] = entry
    return out
