"""Fetch candidate still-image URLs twice (5 min apart) to tell live cams from fixed covers, and
grep a few cam pages for image/player sources. Honest UA, robots.txt respected, one pass."""
import hashlib, json, re, sys, time, urllib.parse, urllib.robotparser
from pathlib import Path
import requests
UA = "snowcam-research/0.1 (+https://familytripwise.com)"
S = requests.Session(); S.headers["User-Agent"] = UA
cfg = json.load(open(sys.argv[1])); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
rp_cache = {}
def allowed(url):
    p = urllib.parse.urlparse(url); base = f"{p.scheme}://{p.netloc}"
    if base not in rp_cache:
        rp = urllib.robotparser.RobotFileParser(base + "/robots.txt")
        try: rp.read()
        except Exception: rp = None
        rp_cache[base] = rp
    rp = rp_cache[base]
    return True if rp is None else rp.can_fetch(UA, url)
report = {"pages": {}, "images": {}}
for name, url in cfg["pages"].items():
    if not allowed(url): report["pages"][name] = {"url": url, "robots": "disallow"}; continue
    try:
        r = S.get(url, timeout=20); html = r.text
        hits = sorted(set(re.findall(r'''https?://[^"'\s<>\\]+''', html)))
        keep = [h for h in hits if re.search(r"roundshot|hdrelay|brownrice|ipcamlive|ytimg|youtube\.com/embed|youtu\.be|mux\.com|camstreamer|verkada|webcam|\.jpe?g|prism|panomax|wetmet|ozolio|snapshot", h, re.I)]
        report["pages"][name] = {"url": url, "status": r.status_code, "hits": keep[:80]}
    except Exception as e:
        report["pages"][name] = {"url": url, "error": str(e)[:200]}
def grab(tag):
    for name, url in cfg["images"].items():
        rec = report["images"].setdefault(name, {"url": url})
        if not allowed(url): rec["robots"] = "disallow"; continue
        try:
            r = S.get(url, timeout=20)
            rec[tag] = {"status": r.status_code, "type": r.headers.get("content-type"), "bytes": len(r.content),
                        "sha": hashlib.sha1(r.content).hexdigest()[:12], "last_modified": r.headers.get("last-modified")}
            if r.ok and len(r.content) > 2000:
                (out / f"{name}_{tag}.jpg").write_bytes(r.content)
        except Exception as e:
            rec[tag] = {"error": str(e)[:200]}
grab("a"); time.sleep(300); grab("b")
for name, rec in report["images"].items():
    a, b = rec.get("a", {}), rec.get("b", {})
    rec["changed"] = bool(a.get("sha") and b.get("sha") and a["sha"] != b["sha"])
json.dump(report, open(out / "probe.json", "w"), indent=1)
print(json.dumps(report, indent=1)[:20000])
