from datetime import datetime, timezone

from PIL import Image

from snowcam import fusion, gates, reader
from snowcam.cli import clock_lag_minutes
from snowcam.config import load_registry


def test_registry_loads_14_cams():
    reg = load_registry()
    assert len(reg.cams) == 14
    assert reg.cam("hunter-base").source_url.endswith("/snapshot/hunter1")


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
