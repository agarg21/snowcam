"""One-time cam profiles for cams added without a hand-written description.

The first time a new cam shows a clear daylight frame, a model names it, picks its role and
writes a one-line description; cams that are not useful outdoor ski views (indoor, promo
stills, pointed at the sky) are marked unusable and never read. Profiles live in the database
(data branch), so registry.yaml stays the list of feeds and the model fills in the rest.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from PIL import Image

from . import reader

PROFILE_MODEL = reader.ESCALATE_MODEL
TOOL = {
    "name": "record_profile",
    "description": "Record what this ski-resort webcam shows.",
    "input_schema": {
        "type": "object",
        "properties": {
            "usable": {"type": "boolean", "description": "true only if this is a live outdoor view of ski terrain, a lift, a base area or a snow stake. False for indoor views, promotional or title-card images, views of only sky or a parking lot, or a blank or broken image."},
            "role": {"type": "string", "enum": ["summit", "trail", "base", "stake", "tubing", "other"],
                     "description": "summit = high on the mountain looking out or along a ridge; trail = a slope or lift line on the mountain; base = base area, village, lodge or lift loading area where people gather; stake = a snow-depth stake or board with a ruler; tubing = tubing lanes."},
            "name": {"type": "string", "description": "Short human name for the view, 2-5 words, e.g. 'Summit lift top', 'Base village', 'Snow stake'. Use any place name printed on the image."},
            "describe": {"type": "string", "description": "One sentence: what is in view and where things are (lifts, trails, buildings, stake), for a reader who will judge snow and crowds from it later."},
            "lit_at_night": {"type": "boolean", "description": "For a stake only: does the image suggest it is lit at night? false if unknown."},
        },
        "required": ["usable", "role", "name", "describe", "lit_at_night"],
    },
}


def build_request(resort_name: str, im: Image.Image) -> dict:
    return {
        "model": PROFILE_MODEL,
        "max_tokens": 6000,
        "system": "You catalogue public ski-resort webcams. Describe only what is visible. Never describe people.",
        "tools": [TOOL],
        "tool_choice": {"type": "tool", "name": "record_profile"},
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Image 1:"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": reader._b64(im)}},
            {"type": "text", "text": f"This is a webcam frame from {resort_name}. Record its profile with the record_profile tool."},
        ]}],
    }


def call(request: dict) -> dict:
    import anthropic
    client = anthropic.Anthropic()
    try:
        msg = client.messages.create(**request)
    except anthropic.BadRequestError as e:
        if "tool_choice" not in str(e):
            raise
        msg = client.messages.create(**{**request, "tool_choice": {"type": "auto"}})
    u = msg.usage.model_dump() if hasattr(msg.usage, "model_dump") else {}
    from datetime import datetime, timezone
    reader.CALLS.append((datetime.now(timezone.utc).isoformat(timespec="seconds"), "profile", request["model"],
                         u.get("input_tokens", 0), u.get("output_tokens", 0), u.get("cache_read_input_tokens") or 0,
                         u.get("cache_creation_input_tokens") or 0, reader.price(request["model"], u)))
    block = next((b for b in msg.content if b.type == "tool_use"), None)
    if block is None or not block.input:
        raise RuntimeError(f"no profile (stop_reason={msg.stop_reason})")
    return dict(block.input)


def save(con, cam_id: str, profile: dict, model: str) -> None:
    con.execute("insert or replace into cam_profiles(cam, profile, model, created) values (?,?,?,?)",
                (cam_id, json.dumps(profile), model, datetime.now(timezone.utc).isoformat(timespec="seconds")))
    con.commit()


def apply(reg, con) -> None:
    """Fill in name, role and description for cams that have a stored profile."""
    rows = {r["cam"]: json.loads(r["profile"]) for r in con.execute("select cam, profile from cam_profiles")}
    for cam in reg.cams:
        p = rows.get(cam.id)
        if not p or cam.describe:
            continue
        cam.name = p.get("name") or cam.name
        cam.describe = p.get("describe", "")
        cam.role = p["role"] if p.get("role") in ("summit", "trail", "base", "stake") else cam.role
        cam.usable = cam.usable and bool(p.get("usable", True)) and p.get("role") != "other"
        if cam.role == "stake":
            cam.night_ok = bool(p.get("lit_at_night", False))
