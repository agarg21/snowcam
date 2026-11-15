import re, sys, json, urllib.parse, urllib.robotparser, concurrent.futures as cf
import requests
UA = "snowcam-research/0.1 (+https://familytripwise.com; one-time cam inventory)"
S = requests.Session(); S.headers["User-Agent"] = UA
SIGS = {
 "brownrice": r"brownrice\.com", "roundshot": r"roundshot\.com", "hdrelay": r"hdrelay\.com",
 "ozolio": r"ozolio\.com", "wetmet": r"wetmet\.net", "ipcamlive": r"ipcamlive\.com", "earthcam": r"earthcam\.(com|net)",
 "youtube": r"youtube(-nocookie)?\.com/(embed|live)|youtu\.be/", "panomax": r"panomax\.com", "feratel": r"feratel\.", "skaping": r"skaping\.com",
 "prismcam": r"prismcam\.com|prism\.cam", "camstreamer": r"camstreamer", "iceberg": r"icebergcams", "angelcam": r"angelcam\.com",
 "rtsp.me": r"rtsp\.me", "streamlock/wowza": r"streamlock\.net|wowza", "vimeo": r"player\.vimeo\.com", "verkada": r"verkada",
 "resortcams": r"resortcams\.com", "skyline": r"skylinewebcams", "webcamstream": r"\.m3u8",
}
def robots_ok(url):
    p = urllib.parse.urlparse(url); rp = urllib.robotparser.RobotFileParser()
    try:
        r = S.get(f"{p.scheme}://{p.netloc}/robots.txt", timeout=10)
        rp.parse(r.text.splitlines() if r.ok else [])
    except Exception: return True
    return rp.can_fetch(UA, url)
def get(url):
    if not robots_ok(url): return None, "robots"
    try:
        r = S.get(url, timeout=20, allow_redirects=True)
        return r, r.status_code
    except Exception as e: return None, type(e).__name__
def scan(name, home):
    out = {"resort": name, "home": home, "pages": [], "providers": {}, "jpgs": [], "notes": []}
    r, st = get(home)
    if r is None or r.status_code >= 400:
        out["notes"].append(f"home {st}"); return out
    links = set()
    for m in re.finditer(r'href="([^"#]+)"[^>]*>([^<]{0,80})', r.text, re.I):
        href, text = m.group(1), m.group(2)
        if re.search(r"cam", href, re.I) or re.search(r"\bcams?\b|webcam", text, re.I):
            u = urllib.parse.urljoin(r.url, href)
            if urllib.parse.urlparse(u).netloc.split(".")[-2:] == urllib.parse.urlparse(r.url).netloc.split(".")[-2:]:
                links.add(u)
    pages = [r.url] + sorted(links, key=len)[:3]
    for u in pages:
        pr, st = (r, 200) if u == r.url else get(u)
        if pr is None or pr.status_code >= 400:
            out["notes"].append(f"{u} {st}"); continue
        out["pages"].append(pr.url)
        for k, sig in SIGS.items():
            hits = set(m.group(0) for m in re.finditer(r"[\w./:-]*(" + sig + r")[\w./?=&%-]*", pr.text, re.I))
            if hits: out["providers"].setdefault(k, set()).update(list(hits)[:4])
        for m in re.finditer(r'https?://[^"\'\s>]+?(?:cam|snapshot|webcam)[^"\'\s>]*?\.(?:jpe?g)(?:\?[^"\'\s>]*)?', pr.text, re.I):
            out["jpgs"].append(m.group(0))
    out["providers"] = {k: sorted(v)[:4] for k, v in out["providers"].items()}
    out["jpgs"] = sorted(set(out["jpgs"]))[:6]
    return out
if __name__ == '__main__':
  resorts = json.load(open(sys.argv[1]))
  with cf.ThreadPoolExecutor(6) as ex:
    res = list(ex.map(lambda kv: scan(*kv), resorts.items()))
  json.dump(res, open(sys.argv[2], "w"), indent=1)
  for o in res:
      print(f'{o["resort"]:20} {",".join(o["providers"]) or "-":40} jpgs={len(o["jpgs"])} pages={len(o["pages"])} {"; ".join(o["notes"])[:80]}')
