# Cam access inventory: the 14 shortlist resorts not yet live (2026-11-26)

Method as on 11-15: robots.txt first, honest user agent (`snowcam-research/0.1`), headless render
of each cam page logging image/player requests (render.js), then a probe that fetches candidate
stills twice 5 minutes apart to tell live cams from fixed covers (probe.py). Runs: lab-scan.yml
45871632320, 45872126584, 45873313644. No player, block or challenge was worked around.

| Resort | Result | Why |
| --- | --- | --- |
| Boyne Mountain | **added, 7 cams** | YouTube live stills (`maxresdefault_live.jpg`) of slopes, lifts and base; waterpark cam and 2 unchanged covers left out |
| Crystal Mountain | **added, 1 cam** | Roundshot 360 summit (`backend.roundshot.com/cams/<id>/medium`); the other cams are Verkada video players |
| Mt Hood Meadows | **added, 1 cam** | HDRelay snapshot from the resort's own cam API: snow stake with a trail behind; the other cams are WetMet players |
| Schweitzer | already in the registry | taken off the pages, now in the re-check (8 checks on 11-26) |
| Loon | no | YouTube stills are fixed "live stream" promo covers; cams are Camstreamer video |
| Steamboat | no | Camstreamer video only |
| Sunday River | no | Camstreamer video; the one YouTube still shows the golf course |
| Big Sky | no | Mux video; its still is a marketing clip, not a cam |
| Snowshoe | no | no still-image cam on the resort or partner pages |
| Mountain Creek | not yet | robots.txt allows us (the first check fetched robots.txt with Python's default agent, which got a 403 and read as "disallow"; corrected in batch b); cam page not found by the render, needs a direct link |
| Mt Baker | not yet | same as Mountain Creek |
| Whiteface, Gore | no | refused automated requests on 11-15 (403); not retried |
| Mammoth | no | cams behind a bot challenge; not retried |

Camstreamer (Loon, Steamboat, Sunday River) publishes no still-image URL that we could find; a
resort can add a still feed or a YouTube relay, which is the realistic way to add these three.
