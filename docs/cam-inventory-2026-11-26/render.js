const { chromium } = require('playwright');
const fs = require('fs');
const targets = JSON.parse(fs.readFileSync(process.argv[2]));
const UA = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome snowcam-research/0.1 (+https://familytripwise.com)';
const PROV = /brownrice|hdrelay|roundshot|ozolio|wetmet|ipcamlive|earthcam|youtube|ytimg|panomax|prismcam|prism-cam|camstreamer|webcam\.io|angelcam|rtsp\.me|streamlock|wowza|vimeo|verkada|\.m3u8|snow\.com\/mtncams|skaping|livecam|iplivecams|cam/i;
(async () => {
  const browser = await chromium.launch();
  const out = {};
  for (const [name, home] of Object.entries(targets)) { if (name==='__direct') continue;
    const ctx = await browser.newContext({ userAgent: UA });
    const page = await ctx.newPage();
    const hits = new Set();
    page.on('request', r => { const u = r.url(); if (PROV.test(u) && !/\.(css|woff2?|svg|png)(\?|$)/i.test(u) && !/google|facebook|doubleclick|cookie|analytics|camp/i.test(u)) hits.add(u.slice(0, 200)); });
    let camUrl = null, err = null;
    try {
      await page.goto(home, { waitUntil: 'domcontentloaded', timeout: 30000 });
      await page.waitForTimeout(2500);
      camUrl = (targets.__direct||{})[name] || await page.evaluate(() => {
        const as = [...document.querySelectorAll('a[href]')].filter(a => /web ?cams?|mountain cams?|\bcams?\b/i.test(a.textContent + ' ' + a.getAttribute('aria-label')) || /webcam|mountain-cam|\/cams?\b/i.test(a.href));
        const same = as.filter(a => a.host === location.host && !/camp/i.test(a.href));
        same.sort((a, b) => a.href.length - b.href.length);
        return same.length ? same[0].href : null;
      });
      if (camUrl) { hits.clear(); await page.goto(camUrl, { waitUntil: 'domcontentloaded', timeout: 30000 }); for (let y=0;y<6;y++){ await page.mouse.wheel(0,1200); await page.waitForTimeout(1500);} await page.waitForTimeout(4000);
        const frames = page.frames().map(f => f.url()).filter(u => PROV.test(u)); frames.forEach(f => hits.add(f.slice(0,200))); }
    } catch (e) { err = e.message.slice(0, 120); }
    out[name] = { camUrl, err, hits: [...hits].slice(0, 25) };
    console.log('##', name, camUrl, err || '', '\n   ' + [...hits].slice(0, 12).join('\n   '));
    await ctx.close();
  }
  fs.writeFileSync(process.argv[3], JSON.stringify(out, null, 1));
  await browser.close();
})();
