"""Combine the latest cam readings per resort into a verdict and one line.

Rules, not the model, decide the verdict. Cams answer by role: summit/trail for snow
and visibility, base for busyness and lifts, stake for measured snow.
"""
from __future__ import annotations

from dataclasses import dataclass

MIN_CONF = 0.6  # below this a field is treated as unknown for publishing

SNOW_RANK = {"none": 0, "dusting": 1, "light": 2, "heavy": 3}
BUSY_RANK = {"empty": 0, "light": 1, "normal": 2, "busy": 3, "packed": 4}


@dataclass
class ResortState:
    verdict: str          # Go | OK | Busy | Thin cover | Lifts idle | No read
    line: str
    fused: dict


def _val(reading: dict, field: str):
    f = reading.get(field)
    if not isinstance(f, dict) or f.get("value") in (None, "unknown"):
        return None
    if f.get("conf", 0) < MIN_CONF:
        return None
    return f["value"]


def fuse(resort_name: str, readings: list[tuple[str, dict]], no_read_reasons: list[str]) -> ResortState:
    """readings: list of (role, fields) for usable recent frames."""
    if not readings:
        why = no_read_reasons[0] if no_read_reasons else "no usable cam frames"
        return ResortState("No read", f"{resort_name}: no clear view right now ({why}).", {"reasons": no_read_reasons})

    by_role: dict[str, list[dict]] = {}
    for role, fields in readings:
        by_role.setdefault(role, []).append(fields)

    def best(field: str, roles: list[str], rank: dict | None = None):
        vals = [v for r in roles for f in by_role.get(r, []) if (v := _val(f, field)) is not None]
        if not vals:
            return None
        return max(vals, key=lambda v: rank.get(v, -1)) if rank else max(set(vals), key=vals.count)

    fused = {
        "fresh_snow": best("fresh_snow", ["summit", "trail", "base"], SNOW_RANK),
        "stake_new_snow": best("stake_new_snow", ["stake"], SNOW_RANK),
        "snow_cover": best("snow_cover", ["trail", "summit", "base"]),
        "visibility": best("visibility", ["summit", "trail"]),
        "snowmaking": best("snowmaking", ["summit", "trail", "base"]),
        "lifts_moving": best("lifts_moving", ["base", "trail", "summit"]),
        "base_busyness": best("base_busyness", ["base"], BUSY_RANK),
        "lift_line": best("lift_line", ["base"]),
        "precip_now": best("precip_now", ["summit", "trail", "base", "stake"]),
    }
    stake = [f.get("stake_inches", {}).get("value") for f in by_role.get("stake", [])
             if isinstance(f.get("stake_inches"), dict) and f["stake_inches"].get("conf", 0) >= MIN_CONF]
    fused["stake_inches"] = next((s for s in stake if s is not None), None)

    snow = max(SNOW_RANK.get(fused["fresh_snow"] or "", -1), SNOW_RANK.get(fused["stake_new_snow"] or "", -1))
    busy = BUSY_RANK.get(fused["base_busyness"] or "", -1)

    parts = []
    if snow >= 2:
        parts.append("fresh snow")
    elif snow == 1:
        parts.append("a dusting of new snow")
    if fused["stake_inches"]:
        parts.append(f"stake reads about {fused['stake_inches']:g} in.")
    if fused["snowmaking"] == "on":
        parts.append("snowmaking on")
    if fused["snow_cover"] in ("bare", "patchy"):
        parts.append(f"{fused['snow_cover']} cover")
    if fused["visibility"] in ("fog", "whiteout"):
        parts.append(f"{fused['visibility']} up top")
    if fused["base_busyness"]:
        parts.append(f"base {fused['base_busyness']}")
    if fused["lift_line"] in ("moderate", "long"):
        parts.append(f"{fused['lift_line']} lift line")
    if fused["precip_now"] in ("snow", "rain_or_mix"):
        parts.append("snowing now" if fused["precip_now"] == "snow" else "rain or mix now")

    # Pages may only say what the cams show (Family Tripwise D47), never an official open/closed
    # status, so thin cover and idle lifts are reported as exactly that. A snow stake piles up
    # snow the slopes may not hold, so "Go" needs new snow seen on a slope cam and real cover.
    slope_snow = SNOW_RANK.get(fused["fresh_snow"] or "", -1)
    thin = fused["snow_cover"] in ("bare", "patchy")
    if thin and fused["lifts_moving"] != "yes":
        verdict = "Thin cover"
    elif fused["lifts_moving"] == "no" and snow < 1:
        verdict = "Lifts idle"
    elif busy >= 3 or fused["lift_line"] == "long":
        verdict = "Busy"
    elif slope_snow >= 2 and not thin and fused["visibility"] not in ("fog", "whiteout") and busy <= 2:
        verdict = "Go"
    elif any(v is not None for v in fused.values()):
        verdict = "OK"
    else:
        verdict = "No read"

    # A snow stake alone says nothing about the slopes, lifts or crowds: report its snow, no verdict.
    if not any(by_role.get(r) for r in ("summit", "trail", "base")):
        verdict = "No read"
        parts.append("snow stake only, slopes not in view")

    line = f"{resort_name}: " + (", ".join(parts) if parts else "nothing notable visible") + "."
    return ResortState(verdict, line[0].upper() + line[1:], fused)
