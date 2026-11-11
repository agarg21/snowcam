"""Vision reader: one structured call per usable frame.

The model fills a fixed set of categorical fields chosen by the cam's role, each with a
confidence, and may always answer "unknown". The prompt sends images first (current
frame, the cam's clear-day reference, the last frame it read), then the instructions.

Without an API key, `dry_run=True` writes the exact request bundle to disk so prompts
can be reviewed or labeled by hand.
"""
from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .config import Cam

PROMPT_VERSION = "v0.1"
DEFAULT_MODEL = os.environ.get("SNOWCAM_MODEL", "claude-haiku-5-5")
ESCALATE_MODEL = os.environ.get("SNOWCAM_ESCALATE_MODEL", "claude-sonnet-5-5")
SEND_W = 1092  # 39 x 22 patches at 16:9, about 860 tokens per image

FIELDS: dict[str, dict] = {
    "fresh_snow": {"enum": ["none", "dusting", "light", "heavy", "unknown"],
                   "help": "New snow since the reference/previous frame on trees, roofs, railings, decks. light = about 1-3 in., heavy = 3+ in."},
    "snow_cover": {"enum": ["full", "mostly", "patchy", "bare", "unknown"],
                   "help": "How much of the visible slope surface is covered by snow (natural or man-made)."},
    "surface_look": {"enum": ["corduroy", "packed", "icy_sheen", "slushy", "unknown"],
                     "help": "Groomed lines, packed snow, glare suggesting ice, or wet slush. Use unknown unless clear."},
    "snowmaking": {"enum": ["on", "off", "unknown"], "help": "Visible snow-gun plumes."},
    "precip_now": {"enum": ["none", "snow", "rain_or_mix", "unknown"], "help": "Falling precipitation visible in the frame."},
    "visibility": {"enum": ["clear", "hazy", "fog", "whiteout", "unknown"], "help": "Compared with the clear-day reference."},
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
    "summit": ["fresh_snow", "snow_cover", "visibility", "precip_now", "lifts_moving"],
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
        props["stake_inches"] = {
            "type": "object",
            "properties": {"value": {"type": ["number", "null"]}, "conf": {"type": "number", "minimum": 0, "maximum": 1}},
            "required": ["value", "conf"],
        }
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
                     "or null if you cannot read it.")
        lines.append("- clock_time: the time on any clock in the frame as HH:MM, or null.")
    lines.append("Give conf between 0 and 1 for each field. Record your answers with the record_reading tool.")
    return "\n".join(lines)


def _b64(im: Image.Image) -> str:
    if im.width > SEND_W:
        im = im.resize((SEND_W, int(im.height * SEND_W / im.width)))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


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
    content: list[dict] = []
    labels: list[str] = []
    n = 1
    for i, im in enumerate(current):
        content += [{"type": "text", "text": f"Image {n}:"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _b64(im)}}]
        labels.append(f"Image {n} = current frame{'' if len(current) == 1 else f' {i + 1}/{len(current)}'} at {ts_label}")
        n += 1
    if reference is not None:
        content += [{"type": "text", "text": f"Image {n}:"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _b64(reference)}}]
        labels.append(f"Image {n} = clear-day reference for this cam")
        n += 1
    if previous is not None:
        content += [{"type": "text", "text": f"Image {n}:"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _b64(previous[0])}}]
        labels.append(f"Image {n} = previous read frame at {previous[1]}")
    content.append({"type": "text", "text": _instructions(cam, fields, labels, len(current))})
    return {
        "model": model,
        "max_tokens": 600,
        "system": SYSTEM,
        "tools": [{"name": "record_reading", "description": "Record the cam reading.",
                   "input_schema": _schema(fields, cam.role == "stake")}],
        "tool_choice": {"type": "tool", "name": "record_reading"},
        "messages": [{"role": "user", "content": content}],
    }


def call(request: dict) -> Reading:
    import anthropic  # imported lazily so collection works without the SDK

    client = anthropic.Anthropic()
    msg = client.messages.create(**request)
    block = next(b for b in msg.content if b.type == "tool_use")
    data = dict(block.input)
    evidence = data.pop("evidence", "")
    if "clock_time" in data:
        clock = data.pop("clock_time")
        data["clock_time"] = {"value": clock, "conf": 1.0 if clock else 0.0}
    return Reading(fields=data, evidence=evidence, model=request["model"],
                   raw={"usage": msg.usage.model_dump() if hasattr(msg.usage, "model_dump") else str(msg.usage)})


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
    (out_dir / "prompt.txt").write_text(request["system"] + "\n\n" + "\n".join(texts))
    (out_dir / "schema.json").write_text(json.dumps(request["tools"][0]["input_schema"], indent=2))
    return out_dir
