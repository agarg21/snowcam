"""Command line: collect frames, run gates, read, fuse, report.

  python -m snowcam.cli collect            # one pass over all cams
  python -m snowcam.cli collect --loop 15  # every 15 minutes (jittered)
  python -m snowcam.cli read --dry-run     # write request bundles instead of calling the API
  python -m snowcam.cli read               # call the model on usable unread frames (needs ANTHROPIC_API_KEY)
  python -m snowcam.cli fuse               # per-resort verdicts from recent readings
  python -m snowcam.cli set-reference CAM FRAME_PATH
  python -m snowcam.cli auto-reference     # pick each cam's sharpest daylight frame as reference
  python -m snowcam.cli report             # HTML contact sheet of the latest pass
"""
from __future__ import annotations

import argparse
import html
import json
import os
import random
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from PIL import Image

from . import fusion, gates, grab, reader, store
from .config import ROOT, load_registry


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _ref(cam_id: str):
    p = store.reference_path(cam_id)
    if not p.exists():
        return None, None
    mp = p.with_name("metrics.json")
    m = json.loads(mp.read_text()) if mp.exists() else gates.metrics(Image.open(p).convert("RGB"))
    return Image.open(p).convert("RGB"), m


def collect_once(reg, con, only: list[str] | None = None, quiet: bool = False) -> list[dict]:
    ua = reg.defaults.get("user_agent", "snowcam/0.1")
    offline = store.offline_hashes()
    results = []
    cams = [c for c in reg.cams if c.usable and (not only or c.id in only or c.resort_id in only)]
    # Fetch in parallel (at most 8 at a time, one request per cam); gate and store in order.
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(8) as ex:
        grabbed = list(ex.map(lambda c: (now_utc(), grab.grab(c, ua)), cams))
    for cam, (ts, g) in zip(cams, grabbed):
        if not g.ok:
            store.insert_frame(con, cam.id, ts, None, None, None, "fail",
                               {"status": "fail", "checks": {"fetch": {"status": "fail", "note": g.error}}}, g.error)
            results.append({"cam": cam.id, "status": "fail", "reasons": [f"fetch: {g.error}"]})
            if not quiet:
                print(f"{cam.id:22s} FETCH FAIL  {g.error}")
            continue
        im = g.images[0]
        h = gates.phash(im)
        ref_im, ref_m = _ref(cam.id)
        res = gates.run_gates(cam, im, ts, h, store.recent_hashes(con, cam.id), offline, ref_im, ref_m)
        path = store.save_frame_image(cam.id, ts, im)
        for i, extra in enumerate(g.images[1:], start=2):
            store.save_frame_image(cam.id, ts, extra, suffix=f"_b{i}")
        fid = store.insert_frame(con, cam.id, ts, path, h, im.size, res.status, res.to_dict())
        results.append({"cam": cam.id, "frame_id": fid, "status": res.status, "reasons": res.reasons, "path": str(path)})
        if not quiet:
            print(f"{cam.id:22s} {res.status.upper():9s} sun {res.metrics['sun_elevation']:+5.1f}  "
                  f"edges {res.metrics['edges']:5.1f}  {'; '.join(res.reasons)}")
    return results


def _local_label(ts_iso: str, tz: str) -> str:
    from zoneinfo import ZoneInfo
    return datetime.fromisoformat(ts_iso).astimezone(ZoneInfo(tz)).strftime("%a %b %d, %I:%M %p local time")


def clock_lag_minutes(clock: str | None, ts_iso: str, tz: str) -> float | None:
    """Minutes between a clock read off the frame and the frame time (handles 12/24h)."""
    from zoneinfo import ZoneInfo
    if not clock or ":" not in clock:
        return None
    try:
        hh, mm = (int(x) for x in clock.strip()[:5].split(":"))
    except ValueError:
        return None
    local = datetime.fromisoformat(ts_iso).astimezone(ZoneInfo(tz))
    frame_min = local.hour * 60 + local.minute
    best = None
    candidates = {hh % 24} if hh > 12 else {hh % 12, (hh % 12) + 12}  # analog/12h clocks are ambiguous
    for h in candidates:
        d = abs(frame_min - (h * 60 + mm))
        d = min(d, 1440 - d)
        best = d if best is None else min(best, d)
    return float(best)


def _burst_paths(path: Path) -> list[Path]:
    return [path] + sorted(path.parent.glob(path.stem + "_b*.jpg"))


def read_pending(reg, con, dry_run: bool, samples: int, since_hours: float, limit: int | None) -> None:
    since = (now_utc() - timedelta(hours=since_hours)).isoformat()
    rows = con.execute(
        "select f.* from frames f left join readings r on r.frame_id=f.id "
        "where r.id is null and f.gate_status in ('pass','degraded') and f.ts>=? order by f.ts", (since,)
    ).fetchall()
    if limit:
        rows = rows[:limit]
    if not rows:
        print("no usable unread frames")
        return
    if not dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set; use --dry-run to write request bundles instead")
    jobs = []
    for row in rows:
        try:
            cam = reg.cam(row["cam"])
        except KeyError:
            continue
        if not cam.usable or not cam.describe:
            continue  # not a ski view, or waiting for its one-time profile
        prev = store.last_read_frame(con, cam.id)
        if cam.role == "stake" and prev and prev["ts"] > (now_utc() - timedelta(minutes=110)).isoformat():
            continue  # stakes change slowly and use the larger model: read them every 2 hours
        cur = [Image.open(p).convert("RGB") for p in _burst_paths(Path(row["path"]))]
        ref_im, _ = _ref(cam.id)
        prev_t = None
        if prev and prev["path"]:
            # Frames are not kept between cloud runs; fall back to the saved copy of the last read frame.
            for cand in (Path(prev["path"]), store.DATA / "last" / f"{cam.id}.jpg"):
                if cand.exists():
                    prev_t = (Image.open(cand).convert("RGB"), prev["ts"])
                    break
        req = reader.build_request(cam, cur, _local_label(row["ts"], cam.tz), ref_im,
                                   (prev_t[0], _local_label(prev_t[1], cam.tz)) if prev_t else None,
                                   reader.ESCALATE_MODEL if cam.role == "stake" else reader.DEFAULT_MODEL)
        if dry_run:
            out = store.DATA / "requests" / cam.id / row["ts"].replace(":", "")
            if out.exists():
                continue
            reader.write_bundle(req, out)
            print(f"{cam.id:22s} bundle -> {out}")
            continue
        jobs.append((row, cam, cur, req))

    def run(job):
        row, cam, cur, req = job
        try:
            r1 = reader.call(req)
            fields = r1.fields
            if samples > 1:
                fields = reader.merge_samples(fields, reader.call(req).fields)
            return job, r1, fields, None
        except Exception as e:  # one bad call must not cost the rest of the pass
            return job, None, None, e

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(run, jobs))
    for (row, cam, cur, req), r1, fields, err in results:
        if err is not None:
            print(f"{cam.id:22s} READ ERROR {type(err).__name__}: {str(err)[:200]}")
            continue
        lag = clock_lag_minutes((fields.get("clock_time") or {}).get("value"), row["ts"], cam.tz)
        if lag is not None and lag > 20:  # frame clock disagrees with fetch time: feed is stale
            fields = {k: ({**v, "conf": 0.0} if isinstance(v, dict) else v) for k, v in fields.items()}
            fields["stale_clock_lag_min"] = {"value": lag, "conf": 1.0}
        store.insert_reading(con, row["id"], cam.id, row["ts"], r1.model, reader.PROMPT_VERSION, fields, r1.evidence, r1.raw)
        last = store.DATA / "last" / f"{cam.id}.jpg"
        last.parent.mkdir(parents=True, exist_ok=True)
        cur[0].save(last, format="JPEG", quality=85)
        print(f"{cam.id:22s} " + ", ".join(f"{k}={v.get('value')}({v.get('conf')})" for k, v in fields.items() if isinstance(v, dict)))


def fuse_all(reg, con, window_min: int = 45) -> list[fusion.ResortState]:
    since = (now_utc() - timedelta(minutes=window_min)).isoformat()
    out = []
    for rid, r in reg.resorts.items():
        readings, reasons, awaiting = [], [], 0
        for cam in [c for c in reg.by_resort(rid) if c.usable]:
            rd = con.execute("select fields from readings where cam=? and ts>=? order by ts desc limit 1",
                             (cam.id, since)).fetchone()
            if rd:
                readings.append((cam.role, json.loads(rd["fields"])))
                continue
            fr = con.execute("select gate_status, gates from frames where cam=? and ts>=? order by ts desc limit 1",
                             (cam.id, since)).fetchone()
            if fr and fr["gate_status"] in ("pass", "degraded"):
                awaiting += 1
            elif fr and fr["gates"]:
                g = json.loads(fr["gates"])
                reasons += [f"{cam.name} {k}: {c['note']}" for k, c in g.get("checks", {}).items()
                            if c.get("status") == "fail" and c.get("note")]
        if awaiting and not readings:
            reasons.insert(0, f"{awaiting} cam{'s' if awaiting > 1 else ''} clear but not read yet; reader is off")
        st = fusion.fuse(r["name"], readings, reasons)
        con.execute("insert into resort_state(resort, ts, verdict, line, fused) values (?,?,?,?,?)",
                    (rid, now_utc().isoformat(), st.verdict, st.line, json.dumps(st.fused)))
        con.commit()
        out.append(st)
        print(f"{st.verdict:8s} {st.line}")
    return out


def auto_reference(reg, con) -> None:
    """Pick each cam's sharpest passing daylight frame as its clear-day reference."""
    for cam in reg.cams:
        rows = con.execute("select path, gates from frames where cam=? and gate_status='pass' and path is not null",
                           (cam.id,)).fetchall()
        best = None
        for r in rows:
            m = json.loads(r["gates"])["metrics"]
            if m.get("sun_elevation", -90) < 5:
                continue
            if best is None or m["edges"] > best[1]:
                best = (r["path"], m["edges"])
        if best:
            set_reference(cam.id, best[0])
        else:
            print(f"{cam.id:22s} no daylight pass frame yet")


def describe_new(reg, con) -> None:
    """One-time profile for cams without a description, from this run's clear daylight frame."""
    from . import describe
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set; skipping profiles")
        return
    done = {r["cam"] for r in con.execute("select cam from cam_profiles")}
    since = (now_utc() - timedelta(hours=1)).isoformat()
    for cam in reg.cams:
        if cam.describe or cam.id in done:
            continue
        row = con.execute("select path, gates from frames where cam=? and ts>=? and gate_status in ('pass','degraded') "
                          "and path is not null order by ts desc limit 1", (cam.id, since)).fetchone()
        if not row or not Path(row["path"]).exists():
            continue
        if json.loads(row["gates"] or "{}").get("metrics", {}).get("sun_elevation", -90) < 10:
            continue
        try:
            prof = describe.call(describe.build_request(cam.resort_name, Image.open(row["path"]).convert("RGB")))
        except Exception as e:
            print(f"{cam.id:30s} PROFILE ERROR {type(e).__name__}: {str(e)[:150]}")
            continue
        describe.save(con, cam.id, prof, describe.PROFILE_MODEL)
        print(f"{cam.id:30s} {prof.get('role'):7s} usable={prof.get('usable')} {prof.get('name')}: {prof.get('describe', '')[:90]}")


def verify_run(reg, con) -> None:
    """Owner D49: independent second reads of a sample of this run's reads."""
    from . import verify
    verify.ensure(con)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set; skipping verification")
        return
    since = (now_utc() - timedelta(hours=0.5)).isoformat()
    picks = [(row, cam, phase) for row, cam, phase in verify.pick(con, reg, since, random.Random())
             if row["path"] and Path(row["path"]).exists()]
    if not picks:
        print("nothing to verify")
        return

    def run(item):
        row, cam, phase = item
        frame = Image.open(row["path"]).convert("RGB")
        ref, _ = _ref(cam.id)
        first = json.loads(row["fields"])
        model = verify.INITIAL_MODEL if phase == "initial" else verify.SPOT_MODEL
        try:
            check = verify.call(verify.build_request(cam, frame, ref, model))
            cmp_, ff = verify.compare(first, check), verify.false_fresh(first, check)
            result = {"compare": cmp_, "false_fresh": ff}
            # Spot checks: a cheap verifier that disagrees gets a Sonnet tie-break before it counts.
            if phase == "spot" and (ff or any(v is False for v in cmp_.values())):
                check2 = verify.call(verify.build_request(cam, frame, ref, verify.INITIAL_MODEL))
                result = {"compare": verify.compare(first, check2), "false_fresh": verify.false_fresh(first, check2),
                          "spot_fields": check}
                check, model = check2, verify.INITIAL_MODEL
            return item, model, check, result, None
        except Exception as e:
            return item, model, None, None, e

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(run, picks))
    for (row, cam, phase), model, check, result, err in results:
        if err is not None:
            print(f"{cam.id:30s} VERIFY ERROR {type(err).__name__}: {str(err)[:150]}")
            continue
        con.execute("insert or ignore into verifications(reading_id, cam, resort, ts, phase, model, fields, tiebreak, created) "
                    "values (?,?,?,?,?,?,?,?,?)", (row["id"], cam.id, cam.resort_id, row["ts"], phase, model,
                                                  json.dumps(check), json.dumps(result), verify.now_iso()))
        bad = [f for f, ok in result["compare"].items() if ok is False or (f == "stake_inches" and ok and ok > 1)]
        print(f"{cam.id:30s} {phase:7s} {'FALSE FRESH ' if result['false_fresh'] else ''}disagree={bad}")
    con.commit()


def check_publish(reg, con) -> None:
    """Flip publish on when a resort passes the D49 check; off when spot checks slip."""
    from . import verify
    verify.ensure(con)
    if n := verify.rescore(con):
        print(f"re-scored {n} stored checks")
    for rid in reg.resorts:
        st = con.execute("select publish, fields from publish_state where resort=?", (rid,)).fetchone()
        if st and st["publish"]:
            ok, why = verify.spot_ok(con, rid, json.loads(st["fields"] or "[]"))
            if not ok:
                con.execute("update publish_state set publish=0, summary=?, changed=? where resort=?",
                            (json.dumps({"unpublished": why}), verify.now_iso(), rid))
                print(f"{rid:16s} UNPUBLISHED: {why}")
            continue
        ev = verify.evaluate(con, rid)
        if ev["ready"]:
            con.execute("insert or replace into publish_state(resort, publish, fields, summary, changed) values (?,?,?,?,?)",
                        (rid, 1, json.dumps(ev["fields_ok"]), json.dumps(ev), verify.now_iso()))
            print(f"{rid:16s} PUBLISHED fields={ev['fields_ok']} agreement={ev['agreement']}")
        elif ev["reads"]:
            print(f"{rid:16s} checking: {ev['reads']}/{verify.SAMPLE_TARGET} reads over {ev['days']} days, "
                  f"agreement {ev['agreement']}, false fresh {ev['false_fresh']}")
    con.commit()


def reference_missing(reg, con) -> None:
    """Give cams without a clear-day reference this run's frame, if it is a clean daylight pass."""
    since = (now_utc() - timedelta(hours=1)).isoformat()
    for cam in reg.cams:
        if store.reference_path(cam.id).exists():
            continue
        row = con.execute("select path, gates from frames where cam=? and ts>=? and gate_status='pass' "
                          "and path is not null order by ts desc limit 1", (cam.id, since)).fetchone()
        if not row or not Path(row["path"]).exists():
            continue
        m = json.loads(row["gates"] or "{}").get("metrics", {})
        if m.get("sun_elevation", -90) >= 15:
            set_reference(cam.id, row["path"])


def set_reference(cam_id: str, frame_path: str) -> None:
    dst = store.reference_path(cam_id)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(frame_path, dst)
    m = gates.metrics(Image.open(dst).convert("RGB"))
    dst.with_name("metrics.json").write_text(json.dumps(m))
    print(f"{cam_id:22s} reference <- {frame_path} (edges {m['edges']:.1f})")


def report(reg, con, out: Path) -> Path:
    """Contact sheet of the latest frame per cam with gate results; private, for review."""
    out.parent.mkdir(parents=True, exist_ok=True)
    cards = []
    for cam in reg.cams:
        f = con.execute("select * from frames where cam=? order by ts desc limit 1", (cam.id,)).fetchone()
        if not f:
            continue
        g = json.loads(f["gates"]) if f["gates"] else {}
        rd = con.execute("select fields, notes from readings where frame_id=?", (f["id"],)).fetchone()
        rel = os.path.relpath(f["path"], out.parent) if f["path"] else ""
        notes = "; ".join(f"{k}: {c['note']}" for k, c in g.get("checks", {}).items() if c.get("note"))
        reading = ""
        if rd:
            fl = json.loads(rd["fields"])
            reading = "<br>".join(f"{k}: {v.get('value')} ({v.get('conf')})" for k, v in fl.items() if isinstance(v, dict))
        cards.append(
            f"<figure class='{html.escape(f['gate_status'] or 'fail')}'>"
            + (f"<img src='{html.escape(rel)}' loading='lazy'>" if rel else "<div class='noimg'>no image</div>")
            + f"<figcaption><b>{html.escape(cam.resort_name)} · {html.escape(cam.name)}</b> "
            f"<span class='pill'>{html.escape((f['gate_status'] or 'fail').upper())}</span><br>"
            f"<small>{html.escape(f['ts'][:16].replace('T', ' '))} UTC · role {cam.role}</small><br>"
            f"<small>{html.escape(notes) or 'all checks pass'}</small>"
            + (f"<div class='rd'>{reading}</div>" if reading else "") + "</figcaption></figure>"
        )
    page = f"""<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>
<title>Snow cam pass</title><style>
body{{font:14px system-ui;margin:16px;background:#f6f6f4;color:#222}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}}
figure{{margin:0;background:#fff;border-radius:8px;overflow:hidden;border:2px solid #ddd}}
figure.fail{{border-color:#d9534f}} figure.degraded{{border-color:#e0a800}} figure.pass{{border-color:#4a9a4a}}
img{{width:100%;display:block}} figcaption{{padding:8px}} .pill{{font-size:11px;padding:1px 6px;border-radius:8px;background:#eee}}
.rd{{margin-top:6px;font-size:12px;color:#444}}</style>
<h1>Snow cam pass</h1><div class=grid>{''.join(cards)}</div>"""
    out.write_text(page)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="snowcam")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--loop", type=float, default=0, help="minutes between passes; 0 = once")
    c.add_argument("--hours", type=float, default=0, help="stop looping after this many hours")
    c.add_argument("--only", nargs="*")
    c.add_argument("--read", action="store_true", help="read usable frames after each pass")
    r = sub.add_parser("read")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--samples", type=int, default=1)
    r.add_argument("--since-hours", type=float, default=2)
    r.add_argument("--limit", type=int)
    sub.add_parser("fuse")
    s = sub.add_parser("set-reference")
    s.add_argument("cam")
    s.add_argument("frame")
    sub.add_parser("auto-reference")
    sub.add_parser("describe-new")
    sub.add_parser("verify")
    sub.add_parser("check-publish")
    sub.add_parser("reference-missing")
    fx = sub.add_parser("export-feed")
    fx.add_argument("--out", default=str(ROOT / "feed" / "conditions.json"))
    rp = sub.add_parser("report")
    rp.add_argument("--out", default=str(ROOT / "data" / "report.html"))
    a = ap.parse_args(argv)

    reg = load_registry()
    con = store.connect()
    from . import describe
    describe.apply(reg, con)
    if a.cmd == "collect":
        end = time.time() + a.hours * 3600 if a.hours else None
        while True:
            print(f"--- pass {now_utc().isoformat(timespec='seconds')}")
            collect_once(reg, con, a.only)
            if a.read:
                read_pending(reg, con, dry_run=not os.environ.get("ANTHROPIC_API_KEY"), samples=1, since_hours=0.5, limit=None)
                fuse_all(reg, con)
            if not a.loop or (end and time.time() > end):
                break
            time.sleep(a.loop * 60 + random.uniform(-60, 60))
    elif a.cmd == "read":
        read_pending(reg, con, a.dry_run, a.samples, a.since_hours, a.limit)
    elif a.cmd == "fuse":
        fuse_all(reg, con)
    elif a.cmd == "set-reference":
        set_reference(a.cam, a.frame)
    elif a.cmd == "auto-reference":
        auto_reference(reg, con)
    elif a.cmd == "verify":
        verify_run(reg, con)
    elif a.cmd == "check-publish":
        check_publish(reg, con)
    elif a.cmd == "describe-new":
        describe_new(reg, con)
    elif a.cmd == "reference-missing":
        reference_missing(reg, con)
    elif a.cmd == "export-feed":
        from . import feed
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(feed.build(reg, con), indent=1) + "\n")
        print(out)
    elif a.cmd == "report":
        print(report(reg, con, Path(a.out)))


if __name__ == "__main__":
    main()
