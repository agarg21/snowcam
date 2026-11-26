from datetime import datetime, timezone

from PIL import Image

from snowcam import fusion, gates, reader
from snowcam.cli import clock_lag_minutes
from snowcam.config import load_registry


def test_registry_loads_and_ids_are_unique():
    reg = load_registry()
    ids = [c.id for c in reg.cams]
    assert len(ids) == len(set(ids)) and len(reg.resorts) >= 32
    assert reg.cam("hunter-base").source_url.endswith("/snapshot/hunter1")
    # cams added without a description wait for their one-time profile before being read
    assert all(c.describe or c.resort_id not in ("hunter", "okemo", "mount-snow", "windham") for c in reg.cams)


def test_clock_lag_handles_12h_and_24h():
    ts = "2026-11-11T09:14:20+00:00"  # 04:14 EST
    assert clock_lag_minutes("04:12", ts, "America/New_York") == 2
    assert clock_lag_minutes("16:14", ts, "America/New_York") == 720
    assert clock_lag_minutes(None, ts, "America/New_York") is None


def test_fusion_no_read_when_nothing_usable():
    st = fusion.fuse("Hunter", [], ["Base dark: sun below horizon"])
    assert st.verdict == "No read" and "sun below horizon" in st.line


def test_fusion_go_on_fresh_snow_quiet_base():
    st = fusion.fuse("Hunter", [
        ("summit", {"fresh_snow": {"value": "light", "conf": 0.8}, "visibility": {"value": "clear", "conf": 0.9}}),
        ("base", {"base_busyness": {"value": "light", "conf": 0.7}, "lifts_moving": {"value": "yes", "conf": 0.9}}),
    ], [])
    assert st.verdict == "Go"
    assert "fresh snow" in st.line and "base quiet" in st.line


def test_fusion_ignores_low_confidence_and_flags_busy():
    st = fusion.fuse("Hunter", [
        ("summit", {"fresh_snow": {"value": "heavy", "conf": 0.3}}),
        ("base", {"base_busyness": {"value": "packed", "conf": 0.8}}),
    ], [])
    assert st.verdict == "Busy" and "fresh snow" not in st.line


def test_fusion_stake_reading():
    st = fusion.fuse("Okemo", [("stake", {"stake_new_snow": {"value": "none", "conf": 0.9},
                                          "stake_inches": {"value": 0, "conf": 0.9}})], [])
    assert st.fused["stake_inches"] == 0


def test_request_has_role_fields_and_forced_tool():
    reg = load_registry()
    im = Image.new("RGB", (1920, 1080), "white")
    req = reader.build_request(reg.cam("okemo-stake"), [im], "04:14 AM local time", None, None, "claude-haiku-5-5")
    props = req["tools"][0]["input_schema"]["properties"]
    assert {"stake_new_snow", "stake_inches", "clock_time", "evidence"} <= set(props)
    assert req["tool_choice"] == {"type": "tool", "name": "record_reading"}
    assert req["messages"][0]["content"][1]["type"] == "image"  # images before instructions


def test_dark_gate_fails_at_night_unless_night_ok():
    reg = load_registry()
    night = datetime(2026, 11, 11, 9, 0, tzinfo=timezone.utc)  # 4am ET
    im = Image.new("RGB", (640, 360), (120, 120, 120))
    h = gates.phash(im)
    assert gates.run_gates(reg.cam("hunter-base"), im, night, h, [], []).checks["dark"].status == "fail"
    assert gates.run_gates(reg.cam("okemo-stake"), im, night, h, [], []).checks["dark"].status == "pass"


def test_stale_gate_in_daylight():
    reg = load_registry()
    noon = datetime(2026, 11, 11, 17, 0, tzinfo=timezone.utc)
    im = Image.effect_noise((640, 360), 60).convert("RGB")
    h = gates.phash(im)
    res = gates.run_gates(reg.cam("hunter-base"), im, noon, h, [h, h], [])
    assert res.checks["stale"].status == "fail"


def test_fusion_thin_cover_on_bare_slopes_without_lift_motion():
    st = fusion.fuse("Okemo", [
        ("summit", {"snow_cover": {"value": "bare", "conf": 0.95}, "fresh_snow": {"value": "none", "conf": 0.9},
                    "lifts_moving": {"value": "unknown", "conf": 0.2}}),
    ], [])
    assert st.verdict == "Thin cover" and "thin cover" in st.line


def test_view_shift_ignores_scene_change_but_catches_pan():
    import numpy as np
    from PIL import Image, ImageEnhance
    from snowcam.gates import _view_shift, MOVED_FRAC
    rng = np.random.default_rng(0)
    base = (rng.random((90, 160)) > 0.5).astype("uint8") * 255
    ref = Image.fromarray(np.kron(base, np.ones((8, 8), dtype="uint8"))).convert("RGB")
    w, h = ref.size
    relit = ImageEnhance.Brightness(ref).enhance(0.7)          # same view, different light
    assert _view_shift(relit, ref) < MOVED_FRAC
    panned = ref.crop((int(w * 0.15), 0, w, h)).resize((w, h))  # camera panned and zoomed
    assert _view_shift(panned, ref) > MOVED_FRAC


def test_stake_only_gives_snow_line_but_no_verdict():
    from snowcam.fusion import fuse
    st = fuse("Okemo", [("stake", {"stake_new_snow": {"value": "light", "conf": 0.8},
                                   "precip_now": {"value": "none", "conf": 0.8}})], [])
    assert st.verdict == "No read"
    assert "fresh snow" in st.line and "snow stake only" in st.line


def test_feed_hides_unpublished_resorts_and_never_leaks_paths(tmp_path):
    import sqlite3, json
    from datetime import datetime, timezone
    from snowcam import store, feed
    from snowcam.config import load_registry
    reg = load_registry()
    con = sqlite3.connect(tmp_path / "t.db"); con.row_factory = sqlite3.Row
    con.executescript(store.SCHEMA)
    ts = "2026-11-15T15:00:00+00:00"
    con.execute("insert into frames(cam, ts, path, phash, gate_status, gates) values (?,?,?,?,?,?)",
                ("okemo-summit", ts, "/secret/frame.jpg", "0", "pass", "{}"))
    con.execute("insert into readings(frame_id, cam, ts, model, prompt_version, fields, notes, raw) values (1,?,?,?,?,?,?,?)",
                ("okemo-summit", ts, "m", "v", json.dumps({"fresh_snow": {"value": "heavy", "conf": 0.9}}), "", "{}"))
    con.execute("insert into resort_state(resort, ts, verdict, line, fused) values (?,?,?,?,?)",
                ("okemo", ts, "Go", "Okemo: fresh snow.", json.dumps({"fresh_snow": "heavy", "stake_inches": None})))
    now = datetime(2026, 11, 15, 15, 10, tzinfo=timezone.utc)
    hidden = feed.build(reg, con, now=now)["resorts"]["okemo"]
    assert hidden["status"] == "No read" and hidden["fields"] == {}
    reg.resorts["okemo"]["publish"] = True
    shown = feed.build(reg, con, now=now)
    assert shown["resorts"]["okemo"]["label"] == "Looks good"
    assert shown["resorts"]["okemo"]["status"] == "Looks good"
    assert shown["resorts"]["okemo"]["fields"] == {"fresh_snow": {"value": "heavy", "conf": None}}
    assert "/secret/" not in json.dumps(shown)


def test_stake_snow_alone_never_gives_go():
    # Nov 15 morning: stake read "heavy" while slope cams showed a dusting on patchy cover.
    st = fusion.fuse("Okemo", [
        ("stake", {"stake_new_snow": {"value": "heavy", "conf": 0.8}}),
        ("trail", {"fresh_snow": {"value": "dusting", "conf": 0.7}, "snow_cover": {"value": "patchy", "conf": 0.8}}),
        ("base", {"fresh_snow": {"value": "none", "conf": 0.9}, "snow_cover": {"value": "bare", "conf": 0.9},
                  "base_busyness": {"value": "empty", "conf": 0.8}}),
    ], [])
    assert st.verdict == "Thin cover"


def test_verify_compare_and_publish_gate(tmp_path):
    import sqlite3, json
    from snowcam import verify, store
    first = {"fresh_snow": {"value": "light", "conf": 0.8}, "snow_cover": {"value": "patchy", "conf": 0.9},
             "base_busyness": {"value": "light", "conf": 0.7}, "visibility": {"value": "fog", "conf": 0.4},
             "stake_inches": {"value": 4.5, "conf": 0.8}}
    check = {"fresh_snow": {"value": "none", "conf": 0.7}, "snow_cover": {"value": "patchy", "conf": 0.8},
             "base_busyness": {"value": "empty", "conf": 0.6}, "stake_inches": {"value": 4.0, "conf": 0.7}}
    c = verify.compare(first, check)
    assert c["snow_cover"] is True and c["base_busyness"] is True   # light ~ empty: both "quiet"
    assert c["fresh_snow"] is False and "visibility" not in c       # low-confidence fields are not judged
    assert c["stake_inches"] == 0.5 and verify.false_fresh(first, check)
    con = sqlite3.connect(tmp_path / "v.db"); con.row_factory = sqlite3.Row
    verify.ensure(con)
    good = json.dumps({"compare": {"snow_cover": True, "stake_inches": 0.5}, "false_fresh": False})
    for i in range(100):
        con.execute("insert into verifications(reading_id, resort, ts, phase, tiebreak) values (?,?,?,?,?)",
                    (i, "okemo", f"2026-11-{15 + i % 3}T15:00:00", "initial", good))
    ev = verify.evaluate(con, "okemo")
    assert ev["ready"] and ev["fields_ok"] == ["snow_cover", "stake_inches"]
    con.execute("insert into verifications(reading_id, resort, ts, phase, tiebreak) values (?,?,?,?,?)",
                (999, "okemo", "2026-11-18T15:00:00", "spot", json.dumps({"compare": {}, "false_fresh": True})))
    assert verify.spot_ok(con, "okemo", ["snow_cover"])[0] is False


def test_cover_and_busyness_use_public_grades_in_check_and_feed(tmp_path):
    import sqlite3, json
    from datetime import datetime, timezone
    from snowcam import store, feed, verify
    from snowcam.config import load_registry
    # Neighbouring raw grades agree on the public scale; different grades still disagree.
    f = lambda v: {"value": v, "conf": 0.9}
    assert verify.compare({"snow_cover": f("full")}, {"snow_cover": f("mostly")})["snow_cover"] is True
    assert verify.compare({"snow_cover": f("mostly")}, {"snow_cover": f("patchy")})["snow_cover"] is False
    assert verify.compare({"snow_cover": f("patchy")}, {"snow_cover": f("bare")})["snow_cover"] is True
    assert verify.compare({"base_busyness": f("empty")}, {"base_busyness": f("light")})["base_busyness"] is True

    reg = load_registry()
    con = sqlite3.connect(tmp_path / "t.db"); con.row_factory = sqlite3.Row
    con.executescript(store.SCHEMA)
    verify.ensure(con)
    ts = "2026-11-15T15:00:00+00:00"
    con.execute("insert into frames(cam, ts, path, phash, gate_status, gates) values (?,?,?,?,?,?)",
                ("okemo-summit", ts, "x", "0", "pass", "{}"))
    con.execute("insert into readings(frame_id, cam, ts, model, prompt_version, fields, notes, raw) values (1,?,?,?,?,?,?,?)",
                ("okemo-summit", ts, "m", "v", "{}", "", "{}"))
    st = fusion.fuse("Okemo", [("summit", {"snow_cover": f("patchy"), "fresh_snow": f("heavy"),
                                           "precip_now": f("snow"), "lift_line": f("long")})], [])
    assert st.verdict == "Thin cover" and "thin cover" in st.line
    con.execute("insert into resort_state(resort, ts, verdict, line, fused) values (?,?,?,?,?)",
                ("okemo", ts, st.verdict, st.line, json.dumps(st.fused)))
    # Published with only snow_cover and precip_now passed: lift_line and fresh snow stay out of the line.
    con.execute("insert into publish_state(resort, publish, fields, summary, changed) values (?,?,?,?,?)",
                ("okemo", 1, json.dumps(["snow_cover", "precip_now"]), "{}", ts))
    out = feed.build(reg, con, now=datetime(2026, 11, 15, 15, 10, tzinfo=timezone.utc))["resorts"]["okemo"]
    assert out["fields"] == {"snow_cover": {"value": "thin", "conf": 0.9},
                             "falling_snow": {"value": "yes", "conf": 0.9}}
    assert out["status"] == "Thin cover" and "lift line" not in out["line"] and "fresh" not in out["line"]


def test_rescore_updates_stored_checks(tmp_path):
    import sqlite3, json
    from snowcam import store, verify
    con = sqlite3.connect(tmp_path / "t.db"); con.row_factory = sqlite3.Row
    con.executescript(store.SCHEMA)
    verify.ensure(con)
    f = lambda v: {"value": v, "conf": 0.9}
    con.execute("insert into readings(frame_id, cam, ts, model, prompt_version, fields, notes, raw) values (1,'c','t','m','v',?,'','{}')",
                (json.dumps({"snow_cover": f("full")}),))
    con.execute("insert into verifications(reading_id, cam, resort, ts, phase, model, fields, tiebreak, created) "
                "values (1,'c','okemo','2026-11-15T15:00:00+00:00','initial','m',?,?,'t')",
                (json.dumps({"snow_cover": f("mostly")}), json.dumps({"compare": {"snow_cover": False}, "false_fresh": False})))
    assert verify.rescore(con) == 1 and verify.rescore(con) == 0
    assert verify.evaluate(con, "okemo")["detail"]["snow_cover"]["agreement"] == 1.0


def test_reader_normalizes_fields_missing_conf():
    from snowcam.reader import normalize
    out = normalize({"fresh_snow": {"value": "light"}, "visibility": {"value": "clear", "conf": "0.9"},
                     "snowmaking": "on", "snow_cover_note": "x", "conf": 0.7})
    assert out["fresh_snow"]["conf"] == 0.0 and out["visibility"]["conf"] == 0.9
    assert out["snowmaking"] == {"value": "on", "conf": 0.0}
    assert "snow_cover_note" not in out and "conf" not in out


def test_early_release_rules(tmp_path):
    import sqlite3, json
    from snowcam import store, verify
    con = sqlite3.connect(tmp_path / "t.db"); con.row_factory = sqlite3.Row
    con.executescript(store.SCHEMA); verify.ensure(con)
    def add(resort, day, cmp):
        con.execute("insert into verifications(reading_id, cam, resort, ts, phase, model, fields, tiebreak, created) "
                    "values (null,'c',?,?,'initial','m','{}',?,'t')", (resort, f"2026-11-{day}T15:00:00+00:00",
                                                                       json.dumps({"compare": cmp, "false_fresh": False})))
    for i in range(20):  # 20 checks over 2 days, cover agrees 19/20 (95%), precip all agree
        add("a", 15 + i % 2, {"snow_cover": i != 0, "precip_now": True})
        add("b", 15 + i % 2, {"precip_now": True})  # weather only: never publishes
        add("c", 15, {"snow_cover": True})           # one day only
    assert verify.evaluate(con, "a")["ready"] and verify.evaluate(con, "a")["fields_ok"] == ["precip_now", "snow_cover"]
    assert not verify.evaluate(con, "b")["ready"]
    assert not verify.evaluate(con, "c")["ready"]
    for i in range(3):  # cover slips to 17/23 = 74%: new fields need 90%, published ones 85%
        add("a", 16, {"snow_cover": False})
    assert "snow_cover" not in verify.evaluate(con, "a", keep={"snow_cover"})["fields_ok"]


def _rowdict(**kw):
    return kw


def test_skip_reason_same_dark_blurry():
    import json
    from snowcam import cli
    from snowcam.config import load_registry
    cam = load_registry().cam("okemo-summit")
    g = lambda **m: json.dumps({"metrics": m})
    prev = {"phash": "ffff0000ffff0000", "read_ts": "2026-11-24T15:00:00+00:00"}
    row = lambda gates, ph="ffff0000ffff0001", ts="2026-11-24T16:00:00+00:00": {"gates": gates, "phash": ph, "ts": ts}
    ref = {"sharpness": 500.0}
    assert cli.skip_reason(cam, row(g(brightness=120, sharpness=400)), prev, ref)[0] == "same"
    # older than 3 hours: read again even if identical
    assert cli.skip_reason(cam, row(g(brightness=120, sharpness=400), ts="2026-11-24T18:30:00+00:00"), prev, ref)[0] is None
    # a changed scene is read
    assert cli.skip_reason(cam, row(g(brightness=120, sharpness=400), ph="0000ffff0000ffff"), prev, ref)[0] is None
    assert cli.skip_reason(cam, row(g(brightness=20)), None, ref)[0] == "dark"
    assert cli.skip_reason(cam, row(g(brightness=120, sharpness=30, edge_ratio=0.3)), None, ref)[0] == "blurry"
    # low sharpness alone (flat light, haze) is still read
    assert cli.skip_reason(cam, row(g(brightness=120, sharpness=30, edge_ratio=0.7)), None, ref)[0] is None


def test_cam_blocks_drop_a_disagreeing_cam_field(tmp_path):
    import sqlite3, json
    from snowcam import store, verify
    con = sqlite3.connect(tmp_path / "t.db"); con.row_factory = sqlite3.Row
    con.executescript(store.SCHEMA); verify.ensure(con)
    def add(cam, day, cmp):
        con.execute("insert into verifications(reading_id, cam, resort, ts, phase, model, fields, tiebreak, created) "
                    "values (null,?,'h',?,'initial','m','{}',?,'t')", (cam, f"2026-11-{day}T15:00:00+00:00",
                                                                       json.dumps({"compare": cmp, "false_fresh": False})))
    for i in range(20):
        add("h-lake", 15 + i % 2, {"snow_cover": i < 4})   # a lake view: readers split on cover
        add("h-slope", 15 + i % 2, {"snow_cover": True})
    assert verify.cam_blocks(con) == {"h-lake": {"snow_cover"}}
    ev = verify.evaluate(con, "h")
    assert ev["detail"]["snow_cover"] == {"n": 20, "agreement": 1.0} and ev["ready"]


def test_archive_part_and_merge(tmp_path):
    import sqlite3, json, zipfile
    from PIL import Image
    from snowcam import store, verify, archive
    from snowcam.config import load_registry
    reg = load_registry()
    con = sqlite3.connect(tmp_path / "t.db"); con.row_factory = sqlite3.Row
    con.executescript(store.SCHEMA); verify.ensure(con)
    img = tmp_path / "f.jpg"; Image.new("RGB", (1920, 1080), (200, 210, 220)).save(img)
    for i, (ts, sun) in enumerate([("2026-11-24T15:00:00+00:00", 20), ("2026-11-25T03:00:00+00:00", -40)], 1):
        con.execute("insert into frames(id, cam, ts, path, phash, gate_status, gates) values (?,?,?,?,?,?,?)",
                    (i, "okemo-summit", ts, str(img), "0", "pass", json.dumps({"status": "pass", "metrics": {"sun_elevation": sun}})))
        con.execute("insert into readings(id, frame_id, cam, ts, model, prompt_version, fields, notes, raw) values (?,?,?,?,?,?,?,?,?)",
                    (i, i, "okemo-summit", ts, "claude-haiku-5-5", "v0.3",
                     json.dumps({"snow_cover": {"value": "patchy", "conf": 0.8}}), "", "{}"))
    con.execute("insert into verifications(reading_id, cam, resort, ts, phase, model, fields, tiebreak, created) values "
                "(1,'okemo-summit','okemo','2026-11-24T15:00:00+00:00','initial','claude-sonnet-5-5',?,?,'t')",
                (json.dumps({"snow_cover": {"value": "bare", "conf": 0.7}}), json.dumps({"compare": {"snow_cover": True}, "false_fresh": False})))
    parts = archive.build_part(reg, con, "2026-11-24T00:00:00+00:00", tmp_path / "out", "a")
    assert [p.name for p in parts] == ["snowcam-2026-11-24-part-a.zip"]  # the night frame is left out
    with zipfile.ZipFile(parts[0]) as z:
        names = sorted(z.namelist())
        assert names == ["okemo/okemo-summit/2026-11-24T150000Z.jpg", "okemo/okemo-summit/2026-11-24T150000Z.json"]
        lab = json.loads(z.read(names[1]))
        assert lab["teacher"]["fields"]["snow_cover"] == {"value": "patchy", "conf": 0.8}
        assert lab["check"]["fields"]["snow_cover"]["value"] == "bare" and lab["check"]["agrees"] == {"snow_cover": True}
        from io import BytesIO
        assert Image.open(BytesIO(z.read(names[0]))).size == (640, 360)
    out = tmp_path / "snowcam-2026-11-24.zip"
    assert archive.merge(parts + parts, out) == 1
    with zipfile.ZipFile(out) as z:
        assert "manifest.csv" in z.namelist() and "patchy" in z.read("manifest.csv").decode()


def test_requests_use_768_px_and_prompt_cache():
    import base64, io
    from PIL import Image
    from snowcam import reader, verify
    from snowcam.config import load_registry
    reg = load_registry()
    im = Image.new("RGB", (1920, 1080), (180, 190, 200))
    def widths(req):
        out = []
        for b in req["messages"][0]["content"]:
            if b["type"] == "image":
                out.append(Image.open(io.BytesIO(base64.b64decode(b["source"]["data"]))).width)
        return out
    summit = reg.cam("okemo-summit")
    req = reader.build_request(summit, [im], "10:00", im, (im, "09:00"), reader.DEFAULT_MODEL)
    assert widths(req) == [768, 768, 768]
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert req["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    stake = reg.cam("okemo-stake")
    sreq = reader.build_request(stake, [im], "10:00", im, None, reader.ESCALATE_MODEL)
    w = widths(sreq)
    assert w[0] == 1092 and w[-1] == 1092  # stakes keep full detail; the close-up stays native
    vreq = verify.build_request(summit, im, im, verify.INITIAL_MODEL)
    assert widths(vreq) == [1092, 1092] and vreq["system"][0]["cache_control"]


def test_stake_due_twice_a_day_plus_snowfall(tmp_path):
    import sqlite3, json
    from datetime import datetime, timezone
    from snowcam import store, cli
    from snowcam.config import load_registry
    reg = load_registry()
    cli._REG = reg
    con = sqlite3.connect(tmp_path / "t.db"); con.row_factory = sqlite3.Row
    con.executescript(store.SCHEMA)
    cam = reg.cam("okemo-stake")  # America/New_York
    from datetime import timedelta
    at = lambda h, m=17: datetime(2026, 11, 27, 5, m, tzinfo=timezone.utc) + timedelta(hours=h)  # h is Eastern
    prev = lambda h: {"read_ts": at(h).isoformat()}
    assert cli.stake_due(con, cam, None, at(7)) == (True, "morning")
    assert cli.stake_due(con, cam, prev(7), at(9))[0] is False          # morning done
    assert cli.stake_due(con, cam, prev(7), at(13)) == (True, "afternoon")
    assert cli.stake_due(con, cam, prev(13), at(15))[0] is False
    assert cli.stake_due(con, cam, prev(13), at(19))[0] is False        # evening: only if snowing
    # another Okemo cam saw snow falling within 2 hours -> every 2 hours, any time of day
    con.execute("insert into readings(frame_id, cam, ts, model, prompt_version, fields, notes, raw) values (1,'okemo-summit',?,?,?,?,?,?)",
                (at(18, 30).isoformat(), "m", "v", json.dumps({"precip_now": {"value": "snow", "conf": 0.9}}), "", "{}"))
    assert cli.stake_due(con, cam, prev(13), at(19)) == (True, "snow falling")
    assert cli.stake_due(con, cam, prev(18), at(19))[0] is False        # read under 2 h ago


def test_flat_stake_answer_and_legacy_text():
    from snowcam.reader import normalize
    out = normalize({"stake_inches": 3, "stake_inches_conf": 0.8, "precip_now": {"value": "none", "conf": 0.9}})
    assert out["stake_inches"] == {"value": 3.0, "conf": 0.8}
    out = normalize({"stake_inches": '<parameter name="value">2', "stake_inches_conf": 0.7})
    assert out["stake_inches"] == {"value": 2.0, "conf": 0.7}
    out = normalize({"stake_inches": None, "stake_inches_conf": 0.9})
    assert out["stake_inches"] == {"value": None, "conf": 0.9}
    out = normalize({"stake_inches": "lots", "stake_inches_conf": 0.9})
    assert out["stake_inches"]["conf"] == 0.0   # unreadable text never counts
    assert normalize({"stake_inches": "0"})["stake_inches"]["conf"] == 0.0  # old shape: no confidence


def test_stake_gate_shadow(tmp_path):
    import sqlite3
    import numpy as np
    from PIL import Image, ImageDraw
    from snowcam import stakegate
    con = sqlite3.connect(tmp_path / "t.db")
    stakegate.ensure(con)
    def board(snow: int, seed: int):
        rng = np.random.default_rng(seed)
        im = Image.new("L", (640, 360), 90)
        d = ImageDraw.Draw(im)
        d.rectangle([340, 40, 380, 320], fill=235)          # ruler
        for y in range(60, 320, 20):
            d.rectangle([350, y, 370, y + 6], fill=20)       # marks
        if snow:
            d.rectangle([300, 320 - snow, 430, 340], fill=250)  # snow over the lower ruler
        a = np.clip(np.asarray(im, dtype=np.float32) + rng.normal(0, 4, (360, 640)), 0, 255).astype("uint8")
        p = tmp_path / f"b{snow}_{seed}.jpg"
        Image.fromarray(a).convert("RGB").save(p)
        return str(p)
    cam = "okemo-stake"
    for i in range(3):  # Sonnet reads 0 in. confidently -> references
        stakegate.record(con, cam, f"t{i}", i, i, board(0, i), False, {"stake_inches": {"value": 0, "conf": 0.9}})
    s_clean, tau = stakegate.score(con, cam, board(0, 9), False)
    s_snow, _ = stakegate.score(con, cam, board(120, 9), False)
    assert s_clean >= tau > s_snow
    stakegate.record(con, cam, "t9", 9, 9, board(120, 9), False, {"stake_inches": {"value": 5, "conf": 0.8}})
    assert con.execute("select clean from stake_gate_log where ts='t9'").fetchone()[0] == 0
    assert con.execute("select count(*) from stake_refs").fetchone()[0] == 3  # snow frames never become references
    assert stakegate.SKIP is False
