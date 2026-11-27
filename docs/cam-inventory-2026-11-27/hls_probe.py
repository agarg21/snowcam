"""Find public live video streams on resort cam pages and grab one frame from each (owner, 11-27:
"yes, lets do it" to reading frames from public streams).

Rule: only a stream the resort's own player loads with no token, signature or session in its URL
(nothing locked or signed), fetched with the collector's own user agent and no Referer, where
robots.txt allows it. Signed or tokenised hosts are skipped (WetMet, Angelcam, Twitch, Verkada).
A stream counts as live when its playlist has no #EXT-X-ENDLIST (not a looping clip) and its media
sequence moved between two looks GAP_SECONDS apart. Writes hls.json plus one contact sheet per
resort (frame numbers match hls.json).

Usage: python3 hls_probe.py OUT_DIR render.json [render.json ...]
"""
import io, json, os, re, subprocess, sys, time, urllib.parse, urllib.robotparser
from pathlib import Path

import requests
from PIL import Image, ImageDraw

UA = "snowcam/0.1 (research prototype; contact via repo)"  # the collector's agent (registry.yaml)
S = requests.Session(); S.headers["User-Agent"] = UA
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
SIGNED = re.compile(r"token=|sign|auth|acmb=|sig=|exp=|expires=|policy=|session|jwt|key=", re.I)
SKIP_HOSTS = re.compile(r"wetmet|angelcam|ttvnw|twitch|verkada|prismcam|vimeo", re.I)
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
    return rp is not False and (rp is None or rp.can_fetch("snowcam", url))


def normalise(u):
    """Playlist URL a collector can open cold, or None if it is signed/locked."""
    if SKIP_HOSTS.search(u):
        return None
    p = urllib.parse.urlparse(u)
    q = [(k, v) for k, v in urllib.parse.parse_qsl(p.query) if not k.startswith("_HLS")]
    if any(SIGNED.search(k + "=" + v) for k, v in q):
        return None
    path = re.sub(r"/chunklist[^/]*\.m3u8$", "/playlist.m3u8", p.path)  # Wowza: master, not a session chunklist
    return urllib.parse.urlunparse(p._replace(path=path, query=urllib.parse.urlencode(q)))


def look(url):
    try:
        r = S.get(url, timeout=20)
    except Exception as e:
        return {"error": str(e)[:120]}
    txt = r.text if r.ok else ""
    if "#EXT-X-STREAM-INF" in txt:  # master playlist: follow the first variant
        var = [l for l in txt.splitlines() if l and not l.startswith("#")]
        if var:
            sub = urllib.parse.urljoin(url, var[0])
            try:
                txt = S.get(sub, timeout=20).text
            except Exception:
                pass
    seq = re.search(r"#EXT-X-MEDIA-SEQUENCE:(\d+)", txt)
    return {"status": r.status_code, "endlist": "#EXT-X-ENDLIST" in txt, "seq": int(seq.group(1)) if seq else None}


def frame(url):
    try:
        p = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-user_agent", UA, "-rw_timeout", "15000000",
                            "-i", url, "-frames:v", "1", "-f", "image2", "-c:v", "mjpeg", "pipe:1"],
                           capture_output=True, timeout=45)
        return (p.stdout if p.returncode == 0 and len(p.stdout) > 3000 else None), p.stderr.decode(errors="ignore")[:160]
    except subprocess.TimeoutExpired:
        return None, "ffmpeg timeout"


streams = {}
for f in sys.argv[2:]:
    for resort, rec in json.load(open(f)).items():
        for h in rec.get("hits") or []:
            if ".m3u8" not in h:
                continue
            n = normalise(h)
            if n is None:
                streams.setdefault(resort, {}).setdefault("_skipped", set()).add(urllib.parse.urlparse(h).netloc)
                continue
            streams.setdefault(resort, {})[n] = {"cam_page": rec.get("camUrl")}
report = {}
for resort, ss in streams.items():
    skipped = sorted(ss.pop("_skipped", set()))
    report[resort] = {"skipped_signed_hosts": skipped, "streams": []}
    for u, meta in ss.items():
        report[resort]["streams"].append({"url": u, "cam_page": meta["cam_page"], "robots": "allow" if allowed(u) else "disallow"})
for s in (s for r in report.values() for s in r["streams"] if s["robots"] == "allow"):
    s["a"] = look(s["url"])
time.sleep(int(os.environ.get("GAP_SECONDS", "900")))
sheets = out / "sheets"; sheets.mkdir(exist_ok=True)
for resort, r in report.items():
    tiles = []
    for i, s in enumerate(r["streams"]):
        if s["robots"] != "allow":
            continue
        s["b"] = look(s["url"])
        a, b = s["a"], s["b"]
        moving = a.get("seq") is not None and b.get("seq") is not None and b["seq"] > a["seq"]
        img, err = frame(s["url"])
        s["frame"] = bool(img); s["ffmpeg"] = err if not img else ""
        s["live"] = bool(img) and moving and not (a.get("endlist") or b.get("endlist"))
        if img:
            im = Image.open(io.BytesIO(img)).convert("RGB"); s["size"] = list(im.size)
            im.thumbnail((640, 400)); ImageDraw.Draw(im).text((8, 8), f"{i}{'' if s['live'] else ' NOT LIVE'}", fill=(255, 0, 0))
            tiles.append(im)
    r["live"] = sum(s.get("live", False) for s in r["streams"])
    if tiles:
        cols = 3; rows = (len(tiles) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * 640, rows * 400), (40, 40, 40))
        for k, t in enumerate(tiles):
            sheet.paste(t, ((k % cols) * 640, (k // cols) * 400))
        sheet.save(sheets / (re.sub(r"[^a-z0-9]+", "-", resort.lower()).strip("-") + ".jpg"), quality=80)
json.dump(report, open(out / "hls.json", "w"), indent=1)
for resort, r in sorted(report.items()):
    print(f"{resort}: {len(r['streams'])} public streams, {r['live']} live; signed hosts skipped: {r['skipped_signed_hosts']}")
