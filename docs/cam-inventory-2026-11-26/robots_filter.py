"""Keep only targets whose robots.txt allows our honest user agent (snowcam-research/0.1)."""
import json, sys, urllib.robotparser, urllib.parse
UA = "snowcam-research/0.1"
src = json.load(open(sys.argv[1])); out = {}
for name, url in src.items():
    p = urllib.parse.urlparse(url)
    rp = urllib.robotparser.RobotFileParser(f"{p.scheme}://{p.netloc}/robots.txt")
    try:
        rp.read(); ok = rp.can_fetch(UA, url)
    except Exception as e:
        ok = True; print(name, "robots.txt unreadable:", e)
    print(f"{name}: robots {'allows' if ok else 'DISALLOWS'} {url}")
    if ok: out[name] = url
json.dump(out, open(sys.argv[2], "w"), indent=1)
