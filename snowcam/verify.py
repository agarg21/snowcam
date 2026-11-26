"""Automated accuracy check that gates publishing (Family Tripwise owner D49).

1. Before a resort is published, a sample of its daytime reads is re-read by an independent
   verifier: a different model (Sonnet) with a differently worded prompt, shown the images in a
   different order and never shown the first read.
2. A resort passes after 100 verified reads over at least 3 days when every published field
   agrees at least 90% of the time, no read claims fresh snow the verifier does not see, and
   stake depth is within 1 inch in 90% of reads and never off by more than 2 inches.
   Fields that fail stay out of the feed; the resort can publish the rest.
3. After publishing, 10% of reads are re-read by Haiku with the verifier prompt; where it
   disagrees, Sonnet breaks the tie. A published field whose agreement over its last 50 checks
   drops below 85%, or any false fresh-snow read, unpublishes the resort automatically.
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone

from PIL import Image

from . import reader
from .fusion import BUSY3 as BUSY, COVER, MIN_CONF

INITIAL_MODEL = reader.ESCALATE_MODEL
SPOT_MODEL = reader.DEFAULT_MODEL
# Owner, 2026-11-16: release early and keep checking. A resort publishes after 20 checks over
# 2+ days (fields with 20+ checks at 90%+); checks continue at full rate to 100, a published
# field drops below 85%, and spot checks take over after 100.
SAMPLE_TARGET = 100
PUBLISH_MIN_READS = 20
FIELD_MIN = 20
MIN_DAYS = 2
PASS_AGREE = 0.90
DROP_AGREE = 0.85
DROP_WINDOW = 50
SPOT_SHARE = 0.10
PER_RESORT_PER_RUN = 4
# Re-checks (owner, 2026-11-26): a resort taken off the pages, or one that used all 100 checks
# without passing, keeps getting 2 checks a pass. It is judged only on checks made after it was
# taken off (the latest 100), at the same bar, so a fixed cam can earn its way back on fresh
# frames and old passing checks never re-publish it. About $0.03 a pass per resort.
RECHECK_PER_RUN = 2

# Fields compared, by role. Busyness (quiet / normal / busy) and snow cover (good / thin)
# are compared on the same three-grade scales the feed publishes.
CHECK_FIELDS = ["fresh_snow", "snow_cover", "snowmaking", "lifts_moving", "base_busyness",
                "visibility", "precip_now", "stake_new_snow"]
SCALES = {"base_busyness": BUSY, "snow_cover": COVER}
SNOWY = {"dusting", "light", "heavy"}

SCHEMA = """
create table if not exists verifications (
  id integer primary key, reading_id integer unique, cam text, resort text, ts text,
  phase text, model text, fields json, tiebreak json, created text
);
create table if not exists publish_state (
  resort text primary key, publish integer, fields json, summary json, changed text
);
create table if not exists recheck (resort text primary key, since text);
"""

VERIFY_SYSTEM = (
    "You are checking a ski-resort webcam image for a second opinion. Report only what you can "
    "see yourself. Do not identify or count people. If the image does not let you tell, answer "
    "'unknown'. Being wrong with confidence is worse than saying unknown."
)
# v0.6 (2026-11-26): the same written definitions as the first read (reader.FIELDS), asked
# independently: a different model, the image order reversed and no sight of the first read.
SHARED = ["fresh_snow", "snow_cover", "snowmaking", "visibility", "precip_now"]
HELP = {
    **{f: reader.FIELDS[f]["help"] for f in SHARED},
    "lifts_moving": "Are the lift chairs evidently running? Unknown unless obvious.",
    "base_busyness": "How busy does the base area look compared with an empty base? Never count people.",
    "stake_new_snow": "How much snow sits on the snow board or at the foot of the stake?",
}


def ensure(con) -> None:
    con.executescript(SCHEMA)
    # Resorts taken off before re-checks existed (2026-11-26) start their window when they were taken off.
    con.execute("insert or ignore into recheck select resort, changed from publish_state where publish=0 and changed is not null")


def _schema(fields: list[str], stake: bool) -> dict:
    props = {f: {"type": "object", "properties": {
        "value": {"type": "string", "enum": reader.FIELDS[f]["enum"]},
        "conf": {"type": "number", "minimum": 0, "maximum": 1}}, "required": ["value", "conf"]} for f in fields}
    if stake:
        props.update(reader.STAKE_PROPS)  # flat, as in the first read (v0.5)
    return {"type": "object", "properties": props, "required": list(props)}


def build_request(cam, frame: Image.Image, ref: Image.Image | None, model: str) -> dict:
    fields = [f for f in reader.ROLE_FIELDS[cam.role] if f in CHECK_FIELDS]
    content: list[dict] = []
    n = 1
    if ref is not None:  # reference first: a different order from the first read
        content += [{"type": "text", "text": f"Image {n} (comparison: a clear day at this camera):"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": reader._b64(ref)}}]
        n += 1
    content += [{"type": "text", "text": f"Image {n} (the image to check):"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": reader._b64(frame)}}]
    lines = [f"Camera at {cam.resort_name}: {cam.describe}", f"Check Image {n}. For each item pick one option:"]
    for f in fields:
        lines.append(f"- {f}: {HELP[f]} Options: {', '.join(reader.FIELDS[f]['enum'])}.")
    if cam.role == "stake":
        lines.append("- stake_inches: snow depth on the stake ruler in inches (0 if bare, null if unreadable). "
                     "The ruler's 0 is at the board; read the number at the snow surface.")
    lines.append("Give a confidence from 0 to 1 for each. Use the record_check tool.")
    content.append({"type": "text", "text": "\n".join(lines)})
    return {"model": model, "max_tokens": 6000 if model == INITIAL_MODEL else 600,
            **reader.cached_prefix(VERIFY_SYSTEM, [{"name": "record_check", "description": "Record the second-opinion check.",
                                                    "input_schema": _schema(fields, cam.role == "stake")}]),
            "tool_choice": {"type": "tool", "name": "record_check"},
            "messages": [{"role": "user", "content": content}]}


def call(request: dict) -> dict:
    r = reader.call(request, kind="check")  # same transport: forced tool, auto fallback, empty-answer check
    return r.fields


def _val(field: str, f: dict | None):
    if not isinstance(f, dict) or f.get("value") in (None, "unknown"):
        return None
    v = f["value"]
    return SCALES.get(field, {}).get(v, v)


def spot_triage(first: dict, check: dict, published: set) -> tuple[bool, dict]:
    """Cheap spot check: does it need a Sonnet tie-break? (2026-11-26)

    Only a confident contradiction of a published field (or a false fresh-snow read, which can
    unpublish a resort) earns the tie-break. Otherwise disagreements are left unscored (None):
    an unsure checker or an unpublished field is no evidence either way, and only agreements
    count. Since 11-20, 55% of tie-breaks touched no published field and most were the cheap
    checker answering 'unknown' on snowmaking."""
    cmp_ = compare(first, check)
    if false_fresh(first, check):
        return True, cmp_
    for f, ok in cmp_.items():
        if f == "stake_inches":
            if ok is not None and ok > 1 and f in published:
                return True, cmp_
            continue
        b = check.get(f)
        confident = isinstance(b, dict) and (b.get("conf") or 0) >= MIN_CONF and _val(f, b) is not None
        if ok is False and confident and f in published:
            return True, cmp_
    scored = {f: (None if (ok is False or (f == "stake_inches" and ok is not None and ok > 1)) else ok)
              for f, ok in cmp_.items()}
    return False, scored


def compare(first: dict, check: dict) -> dict:
    """Per field: True (agree), False (disagree) or None (first read published nothing)."""
    out = {}
    for f in CHECK_FIELDS:
        a = first.get(f)
        if not isinstance(a, dict) or a.get("conf", 0) < MIN_CONF or _val(f, a) is None:
            continue  # not published, so not judged
        out[f] = _val(f, a) == _val(f, check.get(f))
    a, b = first.get("stake_inches"), check.get("stake_inches")
    if isinstance(a, dict) and a.get("conf", 0) >= MIN_CONF and a.get("value") is not None:
        bv = (b or {}).get("value")
        out["stake_inches"] = None if bv is None else abs(float(a["value"]) - float(bv))
    return out


def false_fresh(first: dict, check: dict) -> bool:
    for f in ("fresh_snow", "stake_new_snow"):
        a = first.get(f)
        if isinstance(a, dict) and a.get("conf", 0) >= MIN_CONF and a.get("value") in SNOWY \
                and (check.get(f) or {}).get("value") == "none":
            return True
    return False


def window_start(con, resort: str) -> str | None:
    """When a resort was last taken off the pages: only checks after this count toward its record."""
    row = con.execute("select since from recheck where resort=?", (resort,)).fetchone()
    return row["since"] if row else None


def mark_unpublished(con, resort: str, when: str) -> None:
    con.execute("insert or replace into recheck(resort, since) values (?, ?)", (resort, when))


def checks_in_window(con, resort: str) -> int:
    since = window_start(con, resort)
    return con.execute("select count(*) from verifications where resort=? and phase='initial' and (? is null or ts > ?)",
                       (resort, since, since)).fetchone()[0]


# Early finish (2026-11-27): a published resort leaves the first-100 checks once every published
# field is clearly above the bar: one-sided 95% Wilson lower bound >= 85% on 40+ checks
# (about at most 2 misses in 40). Spot checks then take over, as after 100.
EARLY_MIN = 40
EARLY_LB = 0.85


def wilson_lb(k: int, n: int, z: float = 1.645) -> float:
    if n == 0:
        return 0.0
    p = k / n
    return (p + z * z / (2 * n) - z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)) / (1 + z * z / n)


def early_done(con, resort: str, fields: list | set) -> bool:
    fields = set(fields or [])
    if not fields or "stake_inches" in fields:
        return False
    ev = evaluate(con, resort, keep=fields)
    if ev["reads"] < EARLY_MIN:
        return False
    for f in fields:
        d = ev["detail"].get(f) or {}
        n, a = d.get("n", 0), d.get("agreement", 0)
        if n < EARLY_MIN or wilson_lb(round(a * n), n) < EARLY_LB:
            return False
    return True


# Second-opinion checks that nobody waits for go through the Message Batches API at half price
# (2026-11-27); results are collected by the next pass. Spot checks stay direct, because a
# disagreement there needs a tie-break on the same frame, which is gone after the pass.
PENDING = """
create table if not exists pending_checks (
  custom_id text primary key, batch_id text not null, reading_id integer not null, cam text not null,
  resort text not null, ts text not null, phase text not null, model text not null, created text not null
);
"""
BATCH_EXPIRE_H = 26


def pending_count(con, resort: str) -> int:
    con.executescript(PENDING)
    return con.execute("select count(*) from pending_checks where resort=?", (resort,)).fetchone()[0]


def submit_batch(con, items: list[tuple]) -> str | None:
    """items: (row, cam, phase, request). One batch per pass; returns its id."""
    if not items:
        return None
    import anthropic
    con.executescript(PENDING)
    reqs = []
    for row, cam, phase, req in items:
        req = {**req, "tool_choice": {"type": "auto"}, "max_tokens": max(req.get("max_tokens", 600), 6000)} \
            if req["model"] == INITIAL_MODEL else req  # Sonnet refuses a forced tool call
        reqs.append({"custom_id": f"r{row['id']}-{phase}", "params": req})
    batch = anthropic.Anthropic().messages.batches.create(requests=reqs)
    for (row, cam, phase, req), r in zip(items, reqs):
        con.execute("insert or replace into pending_checks values (?,?,?,?,?,?,?,?,?)",
                    (r["custom_id"], batch.id, row["id"], cam.id, cam.resort_id, row["ts"], phase,
                     req["model"], now_iso()))
    con.commit()
    return batch.id


def collect_batches(con) -> list[tuple]:
    """Finished batch results: (pending row, check fields or None, error or None)."""
    import anthropic
    con.executescript(PENDING)
    client = anthropic.Anthropic()
    out = []
    for (bid,) in con.execute("select distinct batch_id from pending_checks").fetchall():
        b = client.messages.batches.retrieve(bid)
        rows = {r["custom_id"]: r for r in con.execute("select * from pending_checks where batch_id=?", (bid,))}
        if b.processing_status != "ended":
            oldest = min(r["created"] for r in rows.values())
            if datetime.fromisoformat(oldest) < datetime.now(timezone.utc) - timedelta(hours=BATCH_EXPIRE_H):
                con.execute("delete from pending_checks where batch_id=?", (bid,))
                out += [(r, None, "batch expired") for r in rows.values()]
            continue
        for res in client.messages.batches.results(bid):
            r = rows.pop(res.custom_id, None)
            if r is None:
                continue
            if res.result.type == "succeeded":
                try:
                    out.append((r, reader.parse(res.result.message, r["model"], "check-batch", discount=0.5).fields, None))
                except Exception as e:
                    out.append((r, None, f"{type(e).__name__}: {str(e)[:120]}"))
            else:
                out.append((r, None, res.result.type))
        out += [(r, None, "missing from results") for r in rows.values()]
        con.execute("delete from pending_checks where batch_id=?", (bid,))
    con.commit()
    return out


def pick(con, reg, since_iso: str, rng: random.Random) -> list[tuple]:
    """This run's reads to check: up to 4 per unpublished resort, about 10% for published ones,
    and 2 re-checks for a resort that was taken off or used all its checks without passing."""
    published_fields = {r["resort"]: json.loads(r["fields"] or "[]")
                        for r in con.execute("select resort, fields from publish_state where publish=1")}
    published = set(published_fields)
    rows = con.execute(
        "select r.id, r.cam, r.ts, r.fields, f.path from readings r join frames f on f.id=r.frame_id "
        "left join verifications v on v.reading_id=r.id where v.id is null and r.ts>=?", (since_iso,)).fetchall()
    by_resort: dict[str, list] = {}
    for row in rows:
        try:
            cam = reg.cam(row["cam"])
        except KeyError:
            continue
        by_resort.setdefault(cam.resort_id, []).append((row, cam))
    out = []
    for rid, items in by_resort.items():
        rng.shuffle(items)
        n = checks_in_window(con, rid) + pending_count(con, rid)
        if rid in published:
            done = n < SAMPLE_TARGET and early_done(con, rid, published_fields.get(rid, []))
            if n < SAMPLE_TARGET and not done:  # published early (or back early): full checking to 100
                out += [(row, cam, "initial") for row, cam in items[:PER_RESORT_PER_RUN]]
            else:
                out += [(row, cam, "spot") for row, cam in items if rng.random() < SPOT_SHARE]
        elif window_start(con, rid) is None and n < SAMPLE_TARGET:
            out += [(row, cam, "initial") for row, cam in items[:PER_RESORT_PER_RUN]]
        else:  # taken off, or all 100 used without passing: keep re-checking on fresh frames
            out += [(row, cam, "initial") for row, cam in items[:RECHECK_PER_RUN]]
    return out


def rescore(con) -> int:
    """Re-score stored checks with the current compare rules (e.g. after a scale change).
    Both reads are kept, so this costs nothing and changes no reading."""
    n = 0
    for r in con.execute("select v.id, v.fields, v.tiebreak, r.fields first from verifications v "
                         "join readings r on r.id=v.reading_id").fetchall():
        first, check = json.loads(r["first"] or "{}"), json.loads(r["fields"] or "{}")
        res = json.loads(r["tiebreak"] or "{}")
        cmp_ = compare(first, check)
        if res.get("triaged"):  # no tie-break was run: disagreements stay unscored
            cmp_ = {f: (None if (ok is False or (f == "stake_inches" and ok is not None and ok > 1)) else ok)
                    for f, ok in cmp_.items()}
            new = dict(res, compare=cmp_, false_fresh=False)
        else:
            new = dict(res, compare=cmp_, false_fresh=false_fresh(first, check))
        if new != res:
            con.execute("update verifications set tiebreak=? where id=?", (json.dumps(new), r["id"]))
            n += 1
    return n


SNOW_FIELDS = {"snow_cover", "fresh_snow", "stake_inches"}

# A cam whose two readers keep disagreeing on a field (e.g. a lake view read for snow cover, or a
# trail that sits on the patchy/mostly line) stops counting for that field: it is not fused, not
# published and not scored. Computed from all checks, for every resort alike (2026-11-25).
BLOCK_MIN = 10
BLOCK_BELOW = 0.70


def cam_blocks(con) -> dict[str, set]:
    per: dict[tuple, list] = {}
    # Initial and spot checks both count, so a cam that starts slipping after the first 100 is caught.
    for r in con.execute("select cam, tiebreak from verifications where phase in ('initial', 'spot')"):
        for f, ok in json.loads(r["tiebreak"] or "{}").get("compare", {}).items():
            if ok is not None and f != "stake_inches":
                per.setdefault((r["cam"], f), []).append(bool(ok))
    out: dict[str, set] = {}
    for (cam, f), xs in per.items():
        if len(xs) >= BLOCK_MIN and sum(xs) / len(xs) < BLOCK_BELOW:
            out.setdefault(cam, set()).add(f)
    return out


def has_snow(fields) -> bool:
    return bool(SNOW_FIELDS & set(fields))


def evaluate(con, resort: str, keep: set | None = None, blocks: dict | None = None) -> dict:
    """Pass/fail per field for a resort from its latest initial checks. A field already published
    (in keep) stays while it holds 85%; a new field needs 90%."""
    keep = keep or set()
    blocks = cam_blocks(con) if blocks is None else blocks
    # The latest 100 initial checks; for a resort taken off the pages, only those made since.
    since = window_start(con, resort)
    rows = con.execute("select cam, ts, fields, tiebreak from verifications v where resort=? and phase='initial' "
                       "and (? is null or ts > ?) order by id desc limit ?",
                       (resort, since, since, SAMPLE_TARGET)).fetchall()
    days = {r["ts"][:10] for r in rows}
    agree: dict[str, list] = {}
    stake: list[float] = []
    falsefresh = 0
    for r in rows:
        res = json.loads(r["tiebreak"] or "{}")
        blocked = blocks.get(r["cam"], set())
        for f, ok in res.get("compare", {}).items():
            if f in blocked:
                continue
            if f == "stake_inches":
                if ok is not None:
                    stake.append(ok)
            elif ok is not None:
                agree.setdefault(f, []).append(ok)
        falsefresh += bool(res.get("false_fresh")) and "fresh_snow" not in blocked
    fields_ok = []
    detail = {}
    for f, xs in agree.items():
        rate = sum(xs) / len(xs)
        detail[f] = {"n": len(xs), "agreement": round(rate, 3)}
        bar = DROP_AGREE if f in keep else PASS_AGREE
        if len(xs) >= FIELD_MIN and rate >= bar and not (f == "fresh_snow" and falsefresh):
            fields_ok.append(f)
    if stake:
        within1 = sum(d <= 1 for d in stake) / len(stake)
        detail["stake_inches"] = {"n": len(stake), "within_1in": round(within1, 3), "max_off": max(stake)}
        bar = DROP_AGREE if "stake_inches" in keep else PASS_AGREE
        if len(stake) >= FIELD_MIN and within1 >= bar and max(stake) <= 2:
            fields_ok.append("stake_inches")
    # The page is about snow: a resort needs a checked snow field, not only weather or crowds.
    ready = len(rows) >= PUBLISH_MIN_READS and len(days) >= MIN_DAYS and has_snow(fields_ok)
    allv = [x for xs in agree.values() for x in xs]
    return {"reads": len(rows), "days": len(days), "false_fresh": falsefresh, "fields_ok": sorted(fields_ok),
            "agreement": round(sum(allv) / len(allv), 3) if allv else None, "detail": detail, "ready": ready}


def spot_ok(con, resort: str, fields: list[str]) -> tuple[bool, str]:
    """Keep publishing? Last 50 spot checks per published field, after tie-breaks."""
    rows = con.execute("select cam, tiebreak from verifications where resort=? and phase='spot' order by id desc limit ?",
                       (resort, DROP_WINDOW)).fetchall()
    blocks = cam_blocks(con)
    per: dict[str, list] = {}
    for r in rows:
        res = json.loads(r["tiebreak"] or "{}")
        blocked = blocks.get(r["cam"], set())
        if res.get("false_fresh") and "fresh_snow" not in blocked:
            return False, "false fresh-snow read in a spot check"
        for f, ok in res.get("compare", {}).items():
            if f in fields and ok is not None and f != "stake_inches" and f not in blocked:
                per.setdefault(f, []).append(ok)
    for f, xs in per.items():
        if len(xs) >= 20 and sum(xs) / len(xs) < DROP_AGREE:
            return False, f"{f} agreement {sum(xs) / len(xs):.0%} over the last {len(xs)} checks"
    return True, ""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
