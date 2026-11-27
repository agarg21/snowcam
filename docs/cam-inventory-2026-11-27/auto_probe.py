"""Turn render.js hits into still-image candidates and keep the live ones.

Candidates per resort: direct .jpg/.jpeg stills, YouTube live stills (maxresdefault_live.jpg of each
video id seen), HDRelay snapshots and Roundshot "medium" stills. Each candidate is fetched twice,
5 minutes apart, with our honest user agent and only where robots.txt allows it; a still whose
bytes changed is live. Writes probe.json plus one contact sheet per resort of its live stills
(640 px wide tiles, numbered as in probe.json) for a quick look at what each cam shows.

Usage: python3 auto_probe.py render.json OUT_DIR
"""
import hashlib, io, json, re, sys, time, urllib.parse, urllib.robotparser
from pathlib import Path

import requests
from PIL import Image, ImageDraw

UA = "snowcam-research/0.1 (+https://familytripwise.com)"
S = requests.Session(); S.headers["User-Agent"] = UA
render = json.load(open(sys.argv[1])); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
SKIP = re.compile(r"logo|icon|sprite|favicon|avatar|banner|hero|thumb_|placeholder|/wp-content/themes/", re.I)
MAX_PER_RESORT = 30
rp_cache = {}


def allowed(url):
    p = urllib.parse.urlparse(url); base = f"{p.scheme}://{p.netloc}"
    if base not in rp_cache:
        try:
            r = S.get(base + "/robots.txt", timeout=15)
            if r.status_code in (401, 403):
                rp_cache[base] = False
            elif r.status_code >= 400:
                rp_cache[base] = None
            else:
                rp = urllib.robotparser.RobotFileParser(); rp.parse(r.text.splitlines()); rp_cache[base] = rp
        except Exception:
            rp_cache[base] = None
    rp = rp_cache[base]
    return rp is not False and (rp is None or rp.can_fetch("snowcam-research", url))


def candidates(hits):
    c = {}
    for h in hits:
        for vid in re.findall(r"(?:youtube(?:-nocookie)?\.com/(?:embed|live)/|ytimg\.com/vi(?:_webp)?/|[?&]v=)([A-Za-z0-9_-]{11})", h):
            c[f"https://i.ytimg.com/vi/{vid}/maxresdefault_live.jpg"] = "youtube"
        m = re.search(r"(https://[a-z0-9]+\.hdrelay\.com)/camera/([0-9a-f]{24})", h)
        if m:
            c[f"{m.group(1)}/camera/{m.group(2)}/snapshot"] = "hdrelay"
        m = re.search(r"backend\.roundshot\.com/cams/([0-9a-f]{32})", h)
        if m:
            c[f"https://backend.roundshot.com/cams/{m.group(1)}/medium"] = "roundshot"
        if re.search(r"\.jpe?g(\?|$)", h, re.I) and not SKIP.search(h) and "ytimg.com" not in h:
            c[h] = "still"
    return list(c.items())[:MAX_PER_RESORT]


def fetch(url):
    try:
        r = S.get(url, timeout=20)
        if not r.ok or len(r.content) < 3000 or "image" not in (r.headers.get("content-type") or "image"):
            return {"status": r.status_code, "bytes": len(r.content)}, None
        return {"status": r.status_code, "bytes": len(r.content), "sha": hashlib.sha1(r.content).hexdigest()[:12]}, r.content
    except Exception as e:
        return {"error": str(e)[:160]}, None


report = {}
for resort, rec in render.items():
    cands = []
    for url, kind in candidates(rec.get("hits") or []):
        cands.append({"url": url, "kind": kind, "robots": "allow" if allowed(url) else "disallow"})
    report[resort] = {"cam_page": rec.get("camUrl"), "render_error": rec.get("err"), "candidates": cands}
for tag in ("a", "b"):
    for resort, r in report.items():
        for c in r["candidates"]:
            if c["robots"] != "allow":
                continue
            c[tag], data = fetch(c["url"])
            if tag == "b" and data:
                c["_img"] = data
    if tag == "a":
        time.sleep(300)

sheets = out / "sheets"; sheets.mkdir(exist_ok=True)
for resort, r in report.items():
    tiles = []
    for i, c in enumerate(r["candidates"]):
        img = c.pop("_img", None)
        a, b = c.get("a", {}), c.get("b", {})
        c["live"] = bool(a.get("sha") and b.get("sha") and a["sha"] != b["sha"])
        if c["live"] and img:
            try:
                im = Image.open(io.BytesIO(img)).convert("RGB")
                c["size"] = list(im.size)
                im.thumbnail((640, 400))
                ImageDraw.Draw(im).text((8, 8), str(i), fill=(255, 0, 0))
                tiles.append(im)
            except Exception as e:
                c["image_error"] = str(e)[:80]
    r["live"] = sum(c["live"] for c in r["candidates"])
    if tiles:
        cols = 3; rows = (len(tiles) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * 640, rows * 400), (40, 40, 40))
        for k, t in enumerate(tiles):
            sheet.paste(t, ((k % cols) * 640, (k // cols) * 400))
        slug = re.sub(r"[^a-z0-9]+", "-", resort.lower()).strip("-")
        sheet.save(sheets / f"{slug}.jpg", quality=80)
json.dump(report, open(out / "probe.json", "w"), indent=1)
for resort, r in report.items():
    print(f"{resort}: page {r['cam_page']} | {len(r['candidates'])} candidates, {r['live']} live")
