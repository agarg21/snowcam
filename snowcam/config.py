"""Load the cam registry into simple dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Cam:
    id: str
    name: str
    feed: str
    role: str
    resort_id: str
    resort_name: str
    lat: float
    lon: float
    tz: str
    describe: str = ""
    stream: str | None = None
    url: str | None = None
    night_ok: bool = False
    cam_page: str | None = None
    crop: tuple[float, float, float, float] | None = None
    usable: bool = True  # false once a profile says the view is not a ski view  # close-up box as fractions (x0, y0, x1, y1)

    @property
    def source_url(self) -> str:
        if self.feed == "brownrice":
            return f"https://player.brownrice.com/snapshot/{self.stream}"
        if not self.url:
            raise ValueError(f"cam {self.id}: feed {self.feed} needs a url")
        return self.url


@dataclass
class Registry:
    cams: list[Cam]
    resorts: dict[str, dict]
    defaults: dict = field(default_factory=dict)

    def cam(self, cam_id: str) -> Cam:
        for c in self.cams:
            if c.id == cam_id:
                return c
        raise KeyError(cam_id)

    def by_resort(self, resort_id: str) -> list[Cam]:
        return [c for c in self.cams if c.resort_id == resort_id]


def load_registry(path: Path | str | None = None) -> Registry:
    path = Path(path) if path else ROOT / "registry.yaml"
    raw = yaml.safe_load(path.read_text())
    cams: list[Cam] = []
    for rid, r in raw["resorts"].items():
        for c in r["cams"]:
            cams.append(
                Cam(
                    id=c["id"],
                    name=c["name"],
                    feed=c["feed"],
                    role=c["role"],
                    resort_id=rid,
                    resort_name=r["name"],
                    lat=r["lat"],
                    lon=r["lon"],
                    tz=r["tz"],
                    describe=c.get("describe", ""),
                    stream=c.get("stream"),
                    url=c.get("url"),
                    night_ok=bool(c.get("night_ok", False)),
                    cam_page=r.get("cam_page"),
                    crop=tuple(c["crop"]) if c.get("crop") else None,
                    usable=bool(c.get("usable", True)),
                )
            )
    return Registry(cams=cams, resorts=raw["resorts"], defaults=raw.get("defaults", {}))
