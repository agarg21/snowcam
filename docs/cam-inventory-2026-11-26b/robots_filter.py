"""Keep only targets whose robots.txt allows our honest user agent. robots.txt is fetched with that
same user agent (Python's default agent gets 403 on some sites, which robotparser reads as
"disallow all"). A 401/403 for our own agent is treated as a refusal and respected."""
import json, sys, urllib.robotparser, urllib.parse
import requests
UA = "snowcam-research/0.1 (+https://familytripwise.com)"
src = json.load(open(sys.argv[1])); out = {}
for name, url in src.items():
    p = urllib.parse.urlparse(url)
    try:
        r = requests.get(f"{p.scheme}://{p.netloc}/robots.txt", headers={"User-Agent": UA}, timeout=15)
        if r.status_code in (401, 403):
            ok, why = False, f"robots.txt refused ({r.status_code})"
        elif r.status_code >= 400:
            ok, why = True, f"no robots.txt ({r.status_code})"
        else:
            rp = urllib.robotparser.RobotFileParser(); rp.parse(r.text.splitlines())
            ok, why = rp.can_fetch("snowcam-research", url), "robots.txt"
    except Exception as e:
        ok, why = True, f"robots.txt unreadable ({type(e).__name__})"
    print(f"{name}: {'allows' if ok else 'DISALLOWS'} ({why}) {url}")
    if ok: out[name] = url
json.dump(out, open(sys.argv[2], "w"), indent=1)
