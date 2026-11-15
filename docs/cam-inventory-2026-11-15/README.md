# Cam access inventory, 45 shortlisted resorts (2026-11-15)

Method: one polite pass per resort with an honest user agent (`snowcam-research/0.1`), robots.txt
checked first; static HTML scan (scan.py) then headless render of the cam page (render.js) to log
the image and player requests it makes. Raw results in the JSON files here. No player was bypassed.

| Status | Resorts | Still-image source |
| --- | --- | --- |
| Ready (33) | Stowe, Okemo, Mount Snow, Hunter, Seven Springs, Windham, Breckenridge, Park City, Vail, Stevens Pass, Heavenly, Keystone, Kirkwood, Northstar, Crested Butte, Wolf Creek | Brownrice `player.brownrice.com/snapshot/<stream>` (+ Vail Resorts stake JPEGs on cache.snow.com) |
| | Killington, Jay Peak, Holiday Valley, Copper, Snowbird | HDRelay `bNN.hdrelay.com/camera/<id>/snapshot` or `img.hdrelay.com/frames/<id>/default/last.jpg`; Roundshot `backend.roundshot.com/cams/<id>/medium` |
| | Jackson Hole, Telluride, Alta, Timberline, Schweitzer, Bogus Basin, Mt Bachelor | resort-hosted JPEGs (cams.jacksonhole.com, alta-webcams CDN, timberlinelodge.com/snowcameras, api.mtbachelor.com), ipcamlive snapshot.php, Roundshot, Prism JPEG |
| | Sugarbush, Stratton, Loon, Deer Valley, Winter Park | webcam.io latest.jpg or YouTube live thumbnails (`i.ytimg.com/vi/<id>/maxresdefault_live.jpg`); Winter Park thin (one still, rest video) |
| Video only (2) | Steamboat, Big Sky | Camstreamer / Mux HLS players, no still image found |
| Needs a closer look (6) | Crystal Mountain (site lists 8 cams incl. Roundshot 360 and a snow stake; images did not load in the render), Snowshoe, Sunday River, Mountain Creek, Mt Baker, Boyne Mountain | cam page not found or loaded no cams |
| Excluded (4) | Whiteface, Gore | refuse automated requests (403), same operator as Belleayre |
| | Mammoth | video behind a bot challenge |
| | Mt Hood Meadows | WetMet players (one HDRelay cam only) |
