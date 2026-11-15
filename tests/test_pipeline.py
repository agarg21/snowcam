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
    assert "fresh snow" in st.line and "base light" in st.line


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
    assert st.verdict == "Thin cover" and "bare cover" in st.line


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
    assert shown["resorts"]["okemo"]["fields"] == {"fresh_snow": "heavy"}
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
