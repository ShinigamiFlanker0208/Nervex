// A side-view brain as an ink line drawing (outline, gyri, cerebellum, stem), sampled into points.
// Procedural on purpose: no traced image, no asset, deterministic, any size.
export function sampleBrain(w: number, h: number, count: number): Float32Array {
  const S = Math.min(w * .5, h * .9);                      // brain box width
  const bw = S, bh = S * .72, ox = (w - bw) / 2, oy = (h - bh) / 2 - h * .02;
  const cv = document.createElement("canvas"); cv.width = Math.max(2, w | 0); cv.height = Math.max(2, h | 0);
  const c = cv.getContext("2d")!;
  const P = (x: number, y: number): [number, number] => [ox + x * bw, oy + y * bh];
  const outline = new Path2D();
  const m = (x: number, y: number) => outline.moveTo(...P(x, y));
  const b = (a: number, bb: number, cc: number, d: number, e: number, f: number) => outline.bezierCurveTo(...P(a, bb), ...P(cc, d), ...P(e, f));
  m(.16, .70);                                                // under the frontal lobe
  b(.02, .62, .00, .30, .16, .14);                            // forehead
  b(.30, -.02, .62, -.04, .80, .08);                          // crown
  b(.96, .18, 1.02, .42, .92, .60);                           // occipital
  b(.86, .68, .76, .70, .66, .66);                            // above cerebellum
  b(.58, .74, .50, .80, .42, .76);                            // temporal lobe underside
  b(.34, .74, .26, .74, .16, .70);
  outline.closePath();
  const cereb = new Path2D(); cereb.ellipse(...P(.77, .74), bw * .14, bh * .11, -.12, 0, Math.PI * 2);
  c.lineCap = "round"; c.lineJoin = "round"; c.strokeStyle = "#000";
  // gyri: meandering strokes clipped to the outline
  c.save(); c.clip(outline);
  let s = 7; const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647);
  c.lineWidth = Math.max(1.4, S / 260);
  // gyri: smooth, curling strokes (turning rate itself drifts, so folds curl back on themselves like real sulci)
  for (let k = 0; k < 26; k++) {
    let x = .1 + rnd() * .8, y = .08 + rnd() * .58, a = rnd() * 6.28, da = 0;
    const len = 14 + (rnd() * 22) | 0;
    c.beginPath(); c.moveTo(...P(x, y));
    for (let i = 0; i < len; i++) {
      da += (rnd() - .5) * .55; da *= .86; a += da + (i % 9 === 8 ? (rnd() < .5 ? 2.2 : -2.2) : 0) * .35;
      const nx = x + Math.cos(a) * .018, ny = y + Math.sin(a) * .024;
      c.quadraticCurveTo(...P(x + Math.cos(a - da) * .009, y + Math.sin(a - da) * .012), ...P(nx, ny)); x = nx; y = ny;
    }
    c.stroke();
  }
  // the two big landmarks: lateral sulcus and central sulcus
  c.lineWidth = Math.max(2, S / 170);
  c.beginPath(); c.moveTo(...P(.28, .58)); c.bezierCurveTo(...P(.42, .46), ...P(.56, .44), ...P(.70, .40)); c.stroke();
  c.beginPath(); c.moveTo(...P(.50, .02)); c.bezierCurveTo(...P(.46, .16), ...P(.54, .26), ...P(.46, .42)); c.stroke();
  c.restore();
  c.lineWidth = Math.max(2.2, S / 150); c.stroke(outline);
  // cerebellum with folia, and the stem
  c.save(); c.clip(cereb); c.lineWidth = Math.max(1, S / 320);
  for (let i = 0; i < 5; i++) { const yy = .66 + i * .04; c.beginPath(); c.moveTo(...P(.62 + i * .01, yy)); c.quadraticCurveTo(...P(.77, yy + .05), ...P(.92 - i * .01, yy - .005)); c.stroke(); }
  c.restore(); c.lineWidth = Math.max(1.8, S / 190); c.stroke(cereb);
  c.beginPath(); c.moveTo(...P(.60, .74)); c.bezierCurveTo(...P(.62, .86), ...P(.60, .94), ...P(.57, 1.02)); c.moveTo(...P(.68, .80)); c.bezierCurveTo(...P(.67, .90), ...P(.66, .96), ...P(.65, 1.02)); c.stroke();

  const { data } = c.getImageData(0, 0, cv.width, cv.height);
  // sample ALL ink pixels, then take a random subset: a grid sample draws ruler-straight dotted rows
  const pts: number[] = [];
  for (let y = 0; y < cv.height; y++) for (let x = 0; x < cv.width; x++) if (data[(y * cv.width + x) * 4 + 3] > 100) pts.push(x, y);
  const n = Math.max(1, pts.length / 2), out = new Float32Array(count * 2);
  const idx = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) { const j = (rnd() * (i + 1)) | 0; [idx[i], idx[j]] = [idx[j], idx[i]]; }
  for (let i = 0; i < count; i++) { const k = idx[i % n], j = i >= n ? 1.6 : 0; out[i * 2] = pts[k * 2] + (rnd() - .5) * j; out[i * 2 + 1] = pts[k * 2 + 1] + (rnd() - .5) * j; }
  return out;
}
