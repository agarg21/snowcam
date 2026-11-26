"""Vision reader: one structured call per usable frame.

The model fills a fixed set of categorical fields chosen by the cam's role, each with a
confidence, and may always answer "unknown". The prompt sends images first (current
frame, the cam's clear-day reference, the last frame it read), then the instructions.

Without an API key, `dry_run=True` writes the exact request bundle to disk so prompts
can be reviewed or labeled by hand.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .config import Cam

PROMPT_VERSION = "v0.6"  # v0.4 (2026-11-26): 768 px regular reads, prompt cache; v0.5: flat stake_inches + stake_inches_conf; v0.6: shared field definitions with the check
DEFAULT_MODEL = os.environ.get("SNOWCAM_MODEL", "claude-haiku-5-5")
ESCALATE_MODEL = os.environ.get("SNOWCAM_ESCALATE_MODEL", "claude-sonnet-5-5")
# Image width sent to the model. Regular reads use 768 px (2026-11-26: same answers as 1092 px in
# the lab cost test, about 17% cheaper); snow stakes, second-opinion checks and one-time profiles
# keep 1092 px (39 x 22 patches at 16:9, about 860 tokens an image) because they need the detail
# or set the accuracy standard.
SEND_W = 768
FULL_W = 1092

FIELDS: dict[str, dict] = {
    # v0.6 (2026-11-26): one written definition per field, shared word for word with the second-
    # opinion check, so both readers judge the same thing. Before, the check judged fresh snow
    # against a clear-day reference from mid-November and the first read against the previous
    # frame, and no read said where "mostly" ends and "patchy" begins.
    "fresh_snow": {"enum": ["none", "dusting", "light", "heavy", "unknown"],
                   "help": ("Snow that fell in about the last 24 hours, judged from the current image itself: clean "
                            "white snow still sitting on tree branches, roof edges, railings, lift chairs or posts, "
                            "or a smooth untracked layer. Snow that has been there for days (gone from branches, grey, "
                            "tracked, crusted) is not fresh. Do not call snow fresh only because the clear-day "
                            "reference shows less. dusting = a thin coat under 1 in.; light = about 1-3 in.; "
                            "heavy = more than 3 in.")},
    "snow_cover": {"enum": ["full", "mostly", "patchy", "bare", "unknown"],
                   "help": ("Share of the visible ski slope or trail surface covered by snow (natural or man-made); "
                            "ignore trees, sky, buildings and parking. full = 90% or more; mostly = 70-90%; "
                            "patchy = 20-70% with grass, dirt or rock showing; bare = under 20%. "
                            "unknown if no ski slope is in view.")},
    "surface_look": {"enum": ["corduroy", "packed", "icy_sheen", "slushy", "unknown"],
                     "help": "Groomed lines, packed snow, glare suggesting ice, or wet slush. Use unknown unless clear."},
    "snowmaking": {"enum": ["on", "off", "unknown"],
                   "help": ("on = white plumes of spray or mist blowing from snow guns somewhere in view; off = no "
                            "plumes anywhere in view; unknown = you cannot see the slopes well enough to tell.")},
    "precip_now": {"enum": ["none", "snow", "rain_or_mix", "unknown"],
                   "help": "Snow or rain falling right now: flakes, streaks or drops in the air or on the lens; none if the air is clear."},
    "visibility": {"enum": ["clear", "hazy", "fog", "whiteout", "unknown"],
                   "help": ("How far you can see. clear = the far features of the view are sharp; hazy = far features "
                            "soft but visible; fog = mid-distance features hidden; whiteout = almost nothing beyond "
                            "the near foreground. The clear-day reference shows what the full view contains.")},
    "lifts_moving": {"enum": ["yes", "no", "unknown"],
                     "help": "Chairs visibly spaced along the line or moving between burst frames. With a single frame, use unknown unless obvious."},
    "base_busyness": {"enum": ["empty", "light", "normal", "busy", "packed", "unknown"],
                      "help": "Number of people in the base area relative to this cam's references. Never count individuals."},
    "lift_line": {"enum": ["none", "short", "moderate", "long", "unknown"],
                  "help": "Only if a lift loading maze is clearly in view; otherwise unknown."},
    "stake_new_snow": {"enum": ["none", "dusting", "light", "heavy", "unknown"],
                       "help": "Snow piled at the base of the stake or on the board."},
    "tubing": {"enum": ["closed", "open_quiet", "open_busy", "unknown"], "help": "Tubing lanes in use."},
    "road_surface": {"enum": ["dry", "wet", "snow_covered", "unknown"], "help": "Road pavement state."},
}

ROLE_FIELDS: dict[str, list[str]] = {
    "summit": ["fresh_snow", "snow_cover", "snowmaking", "visibility", "precip_now", "lifts_moving"],
    "trail": ["fresh_snow", "snow_cover", "surface_look", "snowmaking", "visibility", "precip_now", "lifts_moving"],
    "base": ["fresh_snow", "snow_cover", "snowmaking", "lifts_moving", "base_busyness", "lift_line", "precip_now"],
    "stake": ["stake_new_snow", "precip_now"],
    "tubing": ["tubing", "snow_cover", "precip_now"],
    "road": ["road_surface", "precip_now"],
}

SYSTEM = (
    "You read fixed ski-resort webcams for a conditions service. You describe only what is "
    "visible. You never identify, describe or count individual people; crowd levels are "
    "relative judgments only. When the view does not let you answer a field, answer "
    "'unknown' with low confidence. A wrong confident answer is far worse than 'unknown'."
)


STAKE_PROPS = {
    "stake_inches": {"type": ["number", "null"],
                     "description": "Snow depth read off the stake ruler in inches; 0 if the board is bare, null if unreadable."},
    "stake_inches_conf": {"type": "number", "minimum": 0, "maximum": 1,
                          "description": "Confidence in stake_inches, 0 to 1."},
}


def _num(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        t = v.strip().split(">")[-1].strip()  # tolerate '<parameter name="value">3'
        try:
            return float(t)
        except ValueError:
            return None
    return None


def _schema(fields: list[str], stake: bool) -> dict:
    props: dict = {}
    for f in fields:
        props[f] = {
            "type": "object",
            "properties": {"value": {"type": "string", "enum": FIELDS[f]["enum"]},
                           "conf": {"type": "number", "minimum": 0, "maximum": 1}},
            "required": ["value", "conf"],
        }
    if stake:
        # Flat on purpose (v0.5, 2026-11-26): as a nested {value, conf} object Sonnet returned the
        # value as text in 30% of stake reads ("0", '<parameter name="value">0'), which were lost.
        props.update(STAKE_PROPS)
        props["clock_time"] = {
            "type": ["string", "null"],
            "description": "Time shown on any clock in the frame as HH:MM, or null if none is readable.",
        }
    props["evidence"] = {"type": "string", "description": "One short sentence: what in the image supports the answers."}
    return {"type": "object", "properties": props, "required": list(props.keys())}


def _instructions(cam: Cam, fields: list[str], labels: list[str], burst: int) -> str:
    lines = [
        f"Cam: {cam.resort_name} - {cam.name} (role: {cam.role}). {cam.describe}",
        "Images above: " + "; ".join(labels) + ".",
    ]
    if burst > 1:
        lines.append(f"The current frame is a burst of {burst} images about 5 seconds apart; use them to judge motion.")
    lines.append("Answer these fields:")
    for f in fields:
        lines.append(f"- {f}: {FIELDS[f]['help']} Options: {', '.join(FIELDS[f]['enum'])}.")
    if cam.role == "stake":
        lines.append("- stake_inches: the snow depth read off the stake ruler in inches (0 if the board is bare), "
                     "or null if you cannot read it. The ruler's 0 is at the board surface, so the depth is the "
                     "number where the snow surface meets the ruler; numbers buried in snow are below it. "
                     "Give stake_inches as a plain number and its confidence separately in stake_inches_conf.")
        lines.append("- clock_time: the time on any clock in the frame as HH:MM, or null. Some displays alternate "
                     "between the time, a temperature (e.g. 37F) and the year (e.g. 2026): report a time only if "
                     "the display shows a colon between hours and minutes; otherwise use null.")
    lines.append("Give conf between 0 and 1 for each field. Record your answers with the record_reading tool.")
    return "\n".join(lines)


def _b64(im: Image.Image, width: int = FULL_W) -> str:
    if im.width > width:
        im = im.resize((width, int(im.height * width / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def cached_prefix(system: str, tools: list[dict]) -> dict:
    """Tools and system prompt with a cache breakpoint: they are identical for every call with the
    same role and model, so later calls in a pass read them from the prompt cache (2026-11-26)."""
    tools = [dict(t) for t in tools]
    tools[-1]["cache_control"] = {"type": "ephemeral"}
    return {"tools": tools, "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]}


def system_text(request: dict) -> str:
    s = request["system"]
    return s if isinstance(s, str) else "\n".join(b.get("text", "") for b in s)


@dataclass
class Reading:
    fields: dict
    evidence: str
    model: str
    raw: dict


def build_request(cam: Cam, current: list[Image.Image], ts_label: str,
                  reference: Image.Image | None, previous: tuple[Image.Image, str] | None,
                  model: str) -> dict:
    fields = ROLE_FIELDS[cam.role]
    w = FULL_W if cam.role == "stake" else SEND_W
    _b64w = lambda im: _b64(im, w)
    content: list[dict] = []
    labels: list[str] = []
    n = 1
    for i, im in enumerate(current):
        content += [{"type": "text", "text": f"Image {n}:"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _b64w(im)}}]
        labels.append(f"Image {n} = current frame{'' if len(current) == 1 else f' {i + 1}/{len(current)}'} at {ts_label}")
        n += 1
    if cam.crop:
        im = current[0]
        x0, y0, x1, y1 = cam.crop
        close = im.crop((int(x0 * im.width), int(y0 * im.height), int(x1 * im.width), int(y1 * im.height)))
        content += [{"type": "text", "text": f"Image {n}:"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _b64(close)}}]
        labels.append(f"Image {n} = full-resolution close-up of the current frame (read small markings here)")
        n += 1
    if reference is not None:
        content += [{"type": "text", "text": f"Image {n}:"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _b64w(reference)}}]
        labels.append(f"Image {n} = clear-day reference for this cam")
        n += 1
    if previous is not None:
        content += [{"type": "text", "text": f"Image {n}:"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _b64w(previous[0])}}]
        labels.append(f"Image {n} = previous read frame at {previous[1]}")
    content.append({"type": "text", "text": _instructions(cam, fields, labels, len(current))})
    return {
        "model": model,
        "max_tokens": 600,
        **cached_prefix(SYSTEM, [{"name": "record_reading", "description": "Record the cam reading.",
                                  "input_schema": _schema(fields, cam.role == "stake")}]),
        "tool_choice": {"type": "tool", "name": "record_reading"},
        "messages": [{"role": "user", "content": content}],
    }


# Usage of every model call in this process, flushed to the calls table by the CLI (2026-11-26).
CALLS: list[tuple] = []
PRICE = {  # $ per million tokens: input, output, cache read, cache write (5 min)
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
    "claude-sonnet-5-5": (2.0, 10.0, 0.20, 2.5),
}


def price(model: str, u: dict) -> float:
    i, o, cr, cw = PRICE.get(model, (0, 0, 0, 0))
    return (u.get("input_tokens", 0) * i + u.get("output_tokens", 0) * o
            + (u.get("cache_read_input_tokens") or 0) * cr + (u.get("cache_creation_input_tokens") or 0) * cw) / 1e6


def call(request: dict, kind: str = "read") -> Reading:
    import anthropic  # imported lazily so collection works without the SDK

    client = anthropic.Anthropic()
    try:
        msg = client.messages.create(**request)
    except anthropic.BadRequestError as e:
        # Some models refuse a forced tool call; let them choose it (the prompt asks for it).
        if "tool_choice" not in str(e):
            raise
        # Such models may think first, so give them room to finish the tool call.
        request = {**request, "tool_choice": {"type": "auto"}, "max_tokens": 6000}
        msg = client.messages.create(**request)
    u = msg.usage.model_dump() if hasattr(msg.usage, "model_dump") else {}
    CALLS.append((datetime.now(timezone.utc).isoformat(timespec="seconds"), kind, request["model"],
                  u.get("input_tokens", 0), u.get("output_tokens", 0), u.get("cache_read_input_tokens") or 0,
                  u.get("cache_creation_input_tokens") or 0, price(request["model"], u)))
    block = next((b for b in msg.content if b.type == "tool_use"), None)
    if block is None or not block.input or msg.stop_reason == "max_tokens":
        raise RuntimeError(f"{request['model']} gave no complete reading (stop_reason={msg.stop_reason})")
    data = dict(block.input)
    evidence = data.pop("evidence", "")
    if "clock_time" in data:
        clock = data.pop("clock_time")
        if isinstance(clock, dict):  # the model sometimes answers {value, conf} here too
            clock = clock.get("value")
        if isinstance(clock, str) and clock.strip().startswith("{"):
            try:
                clock = json.loads(clock).get("value")
            except (ValueError, AttributeError):
                clock = None
        if isinstance(clock, str):
            clock = clock.strip().strip('"').strip() or None
        data["clock_time"] = {"value": clock, "conf": 1.0 if clock else 0.0}
    return Reading(fields=normalize(data), evidence=evidence, model=request["model"],
                   raw={"stop_reason": msg.stop_reason,
                        "usage": msg.usage.model_dump() if hasattr(msg.usage, "model_dump") else str(msg.usage)})


def normalize(data: dict) -> dict:
    """Every field as {value, conf} with a numeric conf; a missing or bad conf counts as 0, so the
    field is never published (seen 2026-11-16: one field came back without its conf)."""
    out = {}
    data = dict(data)
    if "stake_inches_conf" in data:  # flat stake answer (v0.5) -> {value, conf}
        c = data.pop("stake_inches_conf")
        raw = data.get("stake_inches")
        if isinstance(raw, dict):
            raw = raw.get("value")
        val = _num(raw)
        data["stake_inches"] = {"value": val, "conf": c if (val is not None or raw is None) else 0.0}
    for k, v in data.items():
        if k not in FIELDS and k not in ("stake_inches", "clock_time"):
            continue  # stray keys the model sometimes adds (e.g. "snow_cover_note")
        if isinstance(v, dict):
            c = v.get("conf")
            try:
                c = max(0.0, min(1.0, float(c)))
            except (TypeError, ValueError):
                c = 0.0
            out[k] = dict(v, value=v.get("value"), conf=c)
        else:
            out[k] = {"value": v, "conf": 0.0}  # a bare value has no confidence: never published
    return out


def merge_samples(a: dict, b: dict) -> dict:
    """Two samples on the same frame: agreement keeps confidence, disagreement halves it."""
    out = {}
    for k, va in a.items():
        vb = b.get(k, {})
        if isinstance(va, dict) and va.get("value") == vb.get("value"):
            out[k] = {"value": va["value"], "conf": round(min(va["conf"], vb.get("conf", va["conf"])), 2)}
        elif isinstance(va, dict):
            out[k] = {"value": va["value"], "conf": round(min(va.get("conf", 0), vb.get("conf", 0)) * 0.5, 2),
                      "disagreed_with": vb.get("value")}
    return out


def write_bundle(request: dict, out_dir: Path) -> Path:
    """Dry run: save images and the prompt so a person (or Claude in chat) can review."""
    out_dir.mkdir(parents=True, exist_ok=True)
    texts = []
    img_i = 0
    for block in request["messages"][0]["content"]:
        if block["type"] == "image":
            img_i += 1
            (out_dir / f"image{img_i}.jpg").write_bytes(base64.b64decode(block["source"]["data"]))
        else:
            texts.append(block["text"])
    (out_dir / "prompt.txt").write_text(system_text(request) + "\n\n" + "\n".join(texts))
    (out_dir / "schema.json").write_text(json.dumps(request["tools"][0]["input_schema"], indent=2))
    return out_dir
