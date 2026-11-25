"""Daily health check: no model calls, only the database and the run list.

Each finding has a stable key (so a daily issue can say "new" or "still open since"), a short
line, and the standard fix from docs/operations.md. The check is deliberately dumb: it finds the
recurring failure shapes we have already seen (a resort stuck before publishing, a cam that never
gets read, a run that failed or never started, spend creeping up) and points at the playbook.
"""
from __future__ import annotations

import json
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .cli import EST_COST
from .verify import SAMPLE_TARGET

STUCK_DAYS = 3          # a resort still unpublished this long after its first read
STALL_HOURS = 48        # no new accuracy checks for an unpublished resort in this long
NO_READ_HOURS = 24      # a resort with no usable read in this long
CAM_MIN_FRAMES = 8      # frames a cam must have in the window before "never read" counts
CAM_WINDOW_HOURS = 48
RUNS_EXPECTED = 13      # collect.yml schedules about 13 passes a day
RUNS_MIN = 9            # fewer than this in 24 h means GitHub dropped or delayed many runs
SPEND_RATIO = 1.5       # a day above 1.5x the median of the 7 days before it


@dataclass
class Finding:
    key: str
    kind: str
    line: str
    fix: str


FIX = {
    "stuck": "Playbook 1: look at the per-field agreement in the line; if one field drags it down, "
             "block that field for the weak cam (registry `block:`) rather than lowering the bar.",
    "stalled": "Playbook 2: the resort gets no new checks. Usually every cam is gated out or not read; "
               "see the cam findings for the same resort, then the latest report.html.",
    "done": "Playbook 2b: every check is in. Find the cam and field that fail most in the stored checks and "
            "block that field for that cam; check-publish re-scores the stored checks each run.",
    "unpublished": "Playbook 3: taken off the pages (early-release snow fields below 85%, or spot checks slipped). Read the reason; fix the cam or field, never the threshold.",
    "no_read": "Playbook 4: no usable read in 24 h. Check the cam URLs still serve images (feed moved, "
               "player changed) and whether it is night or fog everywhere.",
    "cam": "Playbook 5: this cam is fetched but never read. If the gate reason is the same every time, "
           "fix the gate input (crop, reference frame) or mark the cam unusable; don't keep paying to fetch it.",
    "runs_failed": "Playbook 6: open the failed run's log; a failure in two runs in a row is a code fix, not a re-run.",
    "runs_missing": "Playbook 6: GitHub delayed or dropped scheduled runs. Nothing to fix unless it repeats 3 days.",
    "spend": "Playbook 7: more model calls than usual. Check for a new resort, a skip rule that stopped "
             "firing (savings line), or verification picking too many reads.",
}


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _parse(ts: str) -> datetime:
    d = datetime.fromisoformat(ts)
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def resort_findings(reg, con, now: datetime) -> list[Finding]:
    out: list[Finding] = []
    for rid, r in reg.resorts.items():
        name = r.get("name", rid)
        ids = [c.id for c in reg.by_resort(rid)]
        if not ids:
            continue
        first, last = con.execute(f"select min(ts), max(ts) from readings where cam in ({','.join('?' * len(ids))})",
                                  ids).fetchone()
        ps = con.execute("select publish, summary, changed from publish_state where resort=?", (rid,)).fetchone()
        published = bool(r.get("publish")) or bool(ps and ps["publish"])
        summary = json.loads(ps["summary"]) if ps and ps["summary"] else {}

        if last is None or now - _parse(last) > timedelta(hours=NO_READ_HOURS):
            when = f"last read {last[:16]}Z" if last else "never read"
            out.append(Finding(f"no_read:{rid}", "no_read", f"{name}: no usable read in {NO_READ_HOURS} h ({when}).",
                               FIX["no_read"]))
        if published or first is None:
            continue
        age = now - _parse(first)
        recent = con.execute("select count(*) from verifications where resort=? and ts>=?",
                             (rid, _iso(now - timedelta(hours=STALL_HOURS)))).fetchone()[0]
        if "unpublished" in summary:
            out.append(Finding(f"unpublished:{rid}", "unpublished",
                               f"{name}: taken off the pages {str(ps['changed'])[:10]}: {summary['unpublished']}.",
                               FIX["unpublished"]))
        elif age > timedelta(days=STUCK_DAYS):
            reads, agr = summary.get("reads"), summary.get("agreement")
            weak = sorted(((k, v.get("agreement")) for k, v in (summary.get("detail") or {}).items()
                           if isinstance(v, dict) and v.get("agreement") is not None), key=lambda kv: kv[1])[:2]
            weak_s = ", ".join(f"{k} {a:.0%}" for k, a in weak)
            out.append(Finding(f"stuck:{rid}", "stuck",
                               f"{name}: not published {age.days} days after its first read "
                               f"({reads} checks, agreement {agr}{'; weakest ' + weak_s if weak_s else ''}).",
                               FIX["stuck"]))
        done = con.execute("select count(*) from verifications where resort=? and phase='initial'", (rid,)).fetchone()[0]
        if done >= SAMPLE_TARGET:
            # All checks are in and it still failed: new checks will not come, so only a fix that
            # changes how the stored checks score (blocking a weak cam's field) can publish it.
            out.append(Finding(f"done:{rid}", "done",
                               f"{name}: all {done} checks are in and it did not pass; no more checks will run.",
                               FIX["done"]))
        elif recent == 0 and age > timedelta(hours=STALL_HOURS):
            out.append(Finding(f"stalled:{rid}", "stalled",
                               f"{name}: no new accuracy checks in {STALL_HOURS} h, so it can't publish.",
                               FIX["stalled"]))
    return out


def cam_findings(reg, con, now: datetime) -> list[Finding]:
    out: list[Finding] = []
    since = _iso(now - timedelta(hours=CAM_WINDOW_HOURS))
    for cam in reg.cams:
        if not cam.usable:
            continue
        frames = con.execute("select gate_status, gates from frames where cam=? and ts>=?", (cam.id, since)).fetchall()
        if len(frames) < CAM_MIN_FRAMES:
            continue
        reads = con.execute("select count(*) from readings where cam=? and ts>=?", (cam.id, since)).fetchone()[0]
        if reads:
            continue
        reasons: Counter = Counter()
        for f in frames:
            checks = json.loads(f["gates"] or "{}").get("checks", {}) if f["gates"] else {}
            failed = [k for k, c in checks.items() if isinstance(c, dict) and c.get("status") == "fail"]
            reasons.update(failed or [f["gate_status"] or "no gate"])
        top = ", ".join(f"{k} {n}" for k, n in reasons.most_common(2))
        out.append(Finding(f"cam:{cam.id}", "cam",
                           f"{cam.resort_name} / {cam.name}: {len(frames)} frames in {CAM_WINDOW_HOURS} h, none read "
                           f"(held by: {top}).", FIX["cam"]))
    return out


def run_findings(runs: list[dict] | None, now: datetime) -> list[Finding]:
    """runs: `gh run list --json conclusion,createdAt,databaseId,event,url` for collect.yml."""
    if runs is None:
        return []
    # Passes come from the cron and from a timed dispatch (GitHub's cron is often late), so count both.
    day = [r for r in runs if now - _parse(r["createdAt"].replace("Z", "+00:00")) <= timedelta(hours=24)]
    out: list[Finding] = []
    failed = [r for r in day if r.get("conclusion") in ("failure", "timed_out", "startup_failure")]
    if failed:
        links = " ".join(r.get("url", str(r.get("databaseId"))) for r in failed[:3])
        out.append(Finding("runs_failed", "runs_failed", f"{len(failed)} of {len(day)} runs failed in 24 h: {links}",
                           FIX["runs_failed"]))
    if len(day) < RUNS_MIN:
        out.append(Finding("runs_missing", "runs_missing",
                           f"Only {len(day)} runs started in 24 h (about {RUNS_EXPECTED} expected).",
                           FIX["runs_missing"]))
    return out


def daily_spend(con) -> dict[str, float]:
    """Estimated $ per UTC day: reads plus second-opinion checks, at the measured per-call costs."""
    spend: Counter = Counter()
    for table in ("readings", "verifications"):
        for d, model, n in con.execute(f"select substr(ts,1,10), model, count(*) from {table} group by 1, 2"):
            spend[d] += n * EST_COST.get(model, 0.0006)
    return dict(sorted(spend.items()))


def spend_findings(con, now: datetime) -> tuple[list[Finding], str]:
    spend = daily_spend(con)
    yday = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    prior = [v for d, v in spend.items() if d < yday][-7:]
    y = spend.get(yday, 0.0)
    line = f"Estimated model spend {yday}: ${y:.2f}"
    if prior:
        med = statistics.median(prior)
        line += f" (median of the 7 days before: ${med:.2f})"
        if med and y > SPEND_RATIO * med:
            return [Finding("spend", "spend", line + f", {y / med:.1f}x.", FIX["spend"])], line
    return [], line


def check(reg, con, runs: list[dict] | None, now: datetime | None = None) -> tuple[list[Finding], str]:
    now = now or datetime.now(timezone.utc)
    findings = resort_findings(reg, con, now) + cam_findings(reg, con, now) + run_findings(runs, now)
    sp, spend_line = spend_findings(con, now)
    return findings + sp, spend_line


def render(findings: list[Finding], spend_line: str, now: datetime, previous: set[str] | None = None) -> str:
    previous = previous or set()
    lines = [f"Health check {now.strftime('%Y-%m-%d %H:%MZ')}: {len(findings)} finding(s). No model calls were made.", "",
             spend_line, ""]
    order = ["runs_failed", "no_read", "unpublished", "done", "stalled", "stuck", "cam", "spend", "runs_missing"]
    for kind in order:
        group = [f for f in findings if f.kind == kind]
        if not group:
            continue
        lines.append(f"### {kind.replace('_', ' ')}")
        for f in group:
            tag = "" if f.key in previous else " **(new)**"
            lines.append(f"- {f.line}{tag} `{f.key}`")
        lines.append(f"  - Fix: {group[0].fix}")
        lines.append("")
    if not findings:
        lines.append("All clear.")
    lines.append("<!-- keys: " + " ".join(sorted(f.key for f in findings)) + " -->")
    return "\n".join(lines) + "\n"


def previous_keys(body: str | None) -> set[str]:
    if not body or "<!-- keys:" not in body:
        return set()
    return set(body.split("<!-- keys:", 1)[1].split("-->", 1)[0].split())
