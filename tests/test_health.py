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


def test_taken_off_resort_shows_recheck_progress():
    con, reg = _con(), _reg()
    con.execute("insert into readings(cam, ts, model) values ('beta-1', ?, 'claude-haiku-5-5')", (_t(200),))
    con.execute("insert into readings(cam, ts, model) values ('beta-1', ?, 'claude-haiku-5-5')", (_t(1),))
    con.execute("insert into publish_state values ('beta', 0, '[]', '{\"unpublished\": \"x\"}', ?)", (_t(150),))
    verify.ensure(con)
    for i in range(verify.SAMPLE_TARGET):  # old checks, before it was taken off
        con.execute("insert into verifications(reading_id, resort, ts, phase, model) values (?, 'beta', ?, 'initial', 'm')",
                    (i, _t(170)))
    con.execute("insert into verifications(reading_id, resort, ts, phase, model) values (999, 'beta', ?, 'initial', 'm')",
                (_t(5),))
    found = {f.key: f for f in health.check(reg, con, runs=None, now=NOW)[0]}
    assert "1/100 fresh checks" in found["recheck:beta"].line


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


def _check(con, rid, ts_h, ok, reading_id, cam=1):
    tb = json.dumps({"compare": {"snow_cover": ok}})
    con.execute("insert into verifications(reading_id, cam, resort, ts, phase, model, tiebreak) "
                "values (?, ?, ?, ?, 'initial', 'm', ?)", (reading_id, f"{rid}-{cam}", rid, _t(ts_h), tb))


def test_recheck_window_ignores_checks_before_it_was_taken_off():
    con = _con(); verify.ensure(con)
    for i in range(100):  # a passing record from before
        _check(con, "beta", 200 - i, True, i)
    con.execute("insert into publish_state values ('beta', 0, '[]', '{}', ?)", (_t(90),))
    verify.ensure(con)  # backfills the re-check window from `changed`
    assert verify.window_start(con, "beta") == _t(90)
    ev = verify.evaluate(con, "beta")
    assert ev["reads"] == 0 and not ev["ready"]  # old passing checks don't put it back
    for i in range(30):  # fresh checks over two days, mostly agreeing
        _check(con, "beta", 60 - i * 2, i % 15 != 0, 500 + i)
    ev = verify.evaluate(con, "beta")
    assert ev["reads"] == 30 and ev["detail"]["snow_cover"]["agreement"] == round(28 / 30, 3)
    assert verify.checks_in_window(con, "beta") == 30


def test_evaluate_uses_latest_100_checks():
    con = _con(); verify.ensure(con)
    for i in range(100):  # 100 old failures, then 100 newer agreements
        _check(con, "alpha", 300 - i, False, i)
    for i in range(100):  # on another cam: per-cam blocks count every check ever made
        _check(con, "alpha", 150 - i, True, 1000 + i, cam=2)
    assert verify.evaluate(con, "alpha")["detail"]["snow_cover"]["agreement"] == 1.0
