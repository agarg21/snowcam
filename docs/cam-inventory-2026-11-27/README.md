# Cam inventory: 150 more US resorts (2026-11-27)

Owner D59 (Family Tripwise OWNER.md): add about 55 more resorts as cams allow.

Method: `lab-scan-wide.yml`, six shards. robots.txt first with our own agent (`robots_filter.py`), a
headless render of each resort's cam page (`render.js`, honest user agent), then `auto_probe.py`
fetches every still candidate twice and keeps the ones whose bytes changed (live).
- Pass 1 (run 46020808992): direct stills, YouTube live stills, HDRelay snapshots, Roundshot; 5 min gap.
- Pass 2 (run 46028910948): reused pass 1's renders; added Brownrice snapshots, resortcams
  thumbnails, ipcamlive snapshots and stills named inside HDRelay/Angelcam player pages; 15 min gap.
Results (render.json, probe.json, contact sheets) are in the private agarg21/snowcam-lab release
`scan-cam-inventory-2026-11-27`. No player, block or challenge was worked around.

| | Resorts |
| --- | --- |
| Targets | 150 (`targets.json`) |
| robots.txt disallows us | 4: Beaver Mountain, Mt Bohemia, Mountain High, Gunstock |
| Live still cams found | 53 resorts, 137 cams after dedupe (`added.json`), all added to `registry.yaml` |
| Video players only (no still) | most of the rest: WetMet, Camstreamer, Verkada, Mux, Prism, Vimeo, resortcams HLS |

New cams enter without a description; the one-time Sonnet profile sets name, role and usability
(non-ski views are dropped), and each resort publishes after the automated read check (D49).

Loveland (5 cams) was taken back out on 11-27: its cam server returned images to the scan's agent
but a web page to the collector's agent (`snowcam/0.1`), on every cam. Treated as a refusal; not
worked around (no agent change). 52 resorts, 132 cams remain.

## Video streams (owner 11-27: "yes, lets do it")

`hls_probe.py` via `lab-scan-video.yml` (run 46037262742): every .m3u8 the cam pages loaded,
kept only when the URL carries no token, signature or session (WetMet, Angelcam, Twitch, Verkada
skipped), robots.txt allows it, the collector's own agent with no Referer can open it, and the
playlist is live (no ENDLIST, media sequence moving over 15 min). 32 live streams; most duplicate
cams we already read as stills (all Brownrice streams; Beech, Wisp, Wintergreen). Added 12 cams at
6 new resorts as `feed: hls`: Appalachian Ski Mtn (3), Massanutten (3), Ski Cooper (3, incl. a
snow stake), Spirit Mountain, Bryce Resort, Cataloochee. Out: Mux at Big Sky, Brighton and The
Highlands are recorded clips (ENDLIST); three Ski Cooper streams answer 401; Loon's Camstreamer
embeds expose no public stream; Steamboat and Sunday River load no stream without a click.
