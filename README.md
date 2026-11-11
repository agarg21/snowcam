# snowcam

Reads public ski-resort webcams every 15 minutes and turns them into a short conditions
line per resort ("fresh snow, base quiet") or an honest "No read" when the view is dark,
foggy or frozen. Design doc: Snow Cam Reader — Pipeline Design v0.

## Pilot cams (14)

| Resort | Cams | Feed |
| --- | --- | --- |
| Hunter Mountain, NY | Base, Midstation, Hunter East, Summit | Brownrice snapshots |
| Mount Snow, VT | Snow stake, Summit, Somerset, Carinthia, Main base | Brownrice snapshots |
| Okemo, VT | Snow stake, Summit, Sugar House, Roundhouse | Brownrice snapshots |
| Windham Mountain Club, NY | Base | Brownrice snapshot |

Left out for now: Camelback and Blue Mountain (their WetMet streams only play on the
resorts' own sites), Belleayre (site refuses automated access), Jack Frost (cams offline).
We do not work around any of these; ask the resort if you want them.

## How it works

1. `registry.yaml` lists every cam: feed, role (summit, trail, base, stake), description.
2. **Grab** one snapshot per cam (one polite GET; HLS feeds get a 3-frame burst).
3. **Gates** (CPU, milliseconds): offline placeholder, darkness (sun position + pixels),
   stale image, visibility vs the cam's clear-day reference, view moved. Any fail →
   "No read" with the reason; the model is never called.
4. **Reader**: one Claude call per usable frame, images first, forced tool call that
   fills role-specific categorical fields with confidences; "unknown" always allowed.
   Snow-stake cams also return the clock in the frame, which catches stale feeds.
5. **Fusion**: rules (not the model) combine cams by role into Go / OK / Busy /
   Closed / No read and a one-line summary.

## Run

```bash
pip install -r requirements.txt
python -m snowcam.cli collect                 # one pass
python -m snowcam.cli read --dry-run          # write request bundles, no API call
ANTHROPIC_API_KEY=... python -m snowcam.cli read
python -m snowcam.cli fuse
python -m snowcam.cli auto-reference          # after a day of daylight frames
python -m snowcam.cli report                  # data/report.html contact sheet
python -m pytest -q tests
```

## Cloud schedule

`.github/workflows/collect.yml` runs every 15 minutes in daylight and hourly at night.
The SQLite database and reference frames live on the `data` branch; raw frames are
uploaded as private run artifacts kept 30 days and are never published.

Set the repository secret `ANTHROPIC_API_KEY` to turn on the reader; without it the
pipeline still collects and gates frames.

## Privacy and terms

Frames are private working data. Pages show text plus a link to the resort's own cam
page, never rehosted images. No face or plate recognition, no crops of people;
busyness is a relative level only.
