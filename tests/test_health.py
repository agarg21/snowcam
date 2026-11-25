import json
import sqlite3
from datetime import datetime, timedelta, timezone

from snowcam import health, store, verify
from snowcam.config import Cam, Registry

NOW = datetime(2026, 11, 25, 22, 0, tzinfo=timezone.utc)


def _reg():
    cams = [Cam(id=f"{r}-{i}", name=f"Cam {i}", feed="still", role="summit", resort_id=r, resort_name=r.title(),
                lat=0, lon=0, tz="America/New_York", url="x") for r in ("alpha", "beta") for i in (1, 2)]
    return Registry(cams=cams, resorts={"alpha": {"name": "Alpha"}, "beta": {"name": "Beta"}})


def _con():
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.executescript(store.SCHEMA)
    con.executescript(verify.SCHEMA)
    return con


def _t(hours_ago):
    return (NOW - timedelta(hours=hours_ago)).isoformat()


def test_stuck_resort_and_no_read_and_dead_cam():
    con, reg = _con(), _reg()
    # alpha: published, read an hour ago -> quiet. beta: first read 5 days ago, none in 30 h, not published.
    con.execute("insert into publish_state values ('alpha', 1, '[]', '{}', ?)", (_t(100),))
    con.execute("insert into readings(cam, ts, model) values ('alpha-1', ?, 'claude-haiku-5-5')", (_t(1),))
    con.execute("insert into readings(cam, ts, model) values ('beta-1', ?, 'claude-haiku-5-5')", (_t(120),))
    con.execute("insert into readings(cam, ts, model) values ('beta-1', ?, 'claude-haiku-5-5')", (_t(30),))
    gates = json.dumps({"checks": {"dark": {"status": "fail"}}})
    for h in range(10):  # alpha-2 fetched every few hours, never read
        con.execute("insert into frames(cam, ts, gate_status, gates) values ('alpha-2', ?, 'fail', ?)", (_t(h * 3), gates))
    found, spend_line = health.check(reg, con, runs=None, now=NOW)
    keys = {f.key for f in found}
    assert "no_read:beta" in keys and "no_read:alpha" not in keys
    assert "stuck:beta" in keys and "stalled:beta" in keys and "stuck:alpha" not in keys
    assert "cam:alpha-2" in keys and "dark 10" in next(f.line for f in found if f.key == "cam:alpha-2")
    assert "Estimated model spend" in spend_line


def test_done_resort_is_named_not_stalled():
    con, reg = _con(), _reg()
    con.execute("insert into readings(cam, ts, model) values ('beta-1', ?, 'claude-haiku-5-5')", (_t(200),))
    con.execute("insert into readings(cam, ts, model) values ('beta-1', ?, 'claude-haiku-5-5')", (_t(1),))
    for i in range(verify.SAMPLE_TARGET):
        con.execute("insert into verifications(reading_id, resort, ts, phase, model) values (?, 'beta', ?, 'initial', 'm')",
                    (i, _t(100)))
    keys = {f.key for f in health.check(reg, con, runs=None, now=NOW)[0]}
    assert "done:beta" in keys and "stalled:beta" not in keys


def test_runs_failed_and_missing():
    runs = [{"createdAt": (NOW - timedelta(hours=h)).isoformat().replace("+00:00", "Z"), "event": "workflow_dispatch",
             "conclusion": "failure" if h == 2 else "success", "url": f"u{h}"} for h in range(1, 6)]
    keys = {f.key for f in health.run_findings(runs, NOW)}
    assert keys == {"runs_failed", "runs_missing"}
    ok = [dict(r, conclusion="success") for r in runs] * 3
    assert health.run_findings(ok, NOW) == []


def test_spend_jump_flagged():
    con = _con()
    for d in range(2, 9):  # a steady week, then double yesterday
        for _ in range(100):
            con.execute("insert into readings(cam, ts, model) values ('a', ?, 'claude-haiku-5-5')", (_t(24 * d),))
    for _ in range(300):
        con.execute("insert into readings(cam, ts, model) values ('a', ?, 'claude-haiku-5-5')", (_t(24),))
    found, line = health.spend_findings(con, NOW)
    assert [f.key for f in found] == ["spend"] and "3.0x" in found[0].line


def test_render_marks_new_and_keeps_keys():
    f = [health.Finding("cam:x", "cam", "X never read", "fix"), health.Finding("stuck:y", "stuck", "Y stuck", "fix")]
    body = health.render(f, "spend", NOW, previous={"cam:x"})
    assert "Y stuck **(new)**" in body and "X never read `cam:x`" in body
    assert health.previous_keys(body) == {"cam:x", "stuck:y"}
    assert "All clear." in health.render([], "spend", NOW)
