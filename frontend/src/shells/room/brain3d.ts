// A 3D brain for the intro: a stippled surface, drawn like an engraving on paper.
// Surface = union of anatomical ellipsoids (two hemispheres split by the longitudinal fissure, temporal lobes,
// cerebellum, brain stem). Points inside another part are discarded, so only the outer skin remains.
// Folds: a warped sine field pushes the skin inward in the sulci; it also darkens the ink there.
// Axes: x front(+)/back(-), y up, z right/left. Procedural and deterministic: no mesh, no asset.
export interface Brain3D { n: number; p: Float32Array; nrm: Float32Array; fold: Float32Array }

// Implicit surface (signed distance, < 0 inside). Parts blend with a smooth-min, so lobes join like tissue,
// not like balls. Sulci are narrow grooves carved by a warped sine field; the hemispheres are split by a
// fissure groove on top; the cerebellum gets tight parallel folia. Points are snapped onto F = 0 by Newton steps.
type P3 = [number, number, number];
const ell = (x: number, y: number, z: number, c: P3, r: P3) => {
  const qx = (x - c[0]) / r[0], qy = (y - c[1]) / r[1], qz = (z - c[2]) / r[2];
  return (Math.sqrt(qx * qx + qy * qy + qz * qz) - 1) * Math.min(r[0], r[1], r[2]);
};
const smin = (a: number, b: number, k: number) => { const h = Math.max(k - Math.abs(a - b), 0) / k; return Math.min(a, b) - h * h * k * .25; };
function sulcus(x: number, y: number, z: number) {
  const wx = x + .3 * Math.sin(2.6 * y + 1.7 * z), wy = y + .3 * Math.sin(2.3 * z + 1.5 * x), wz = z + .26 * Math.sin(2.1 * x + 2.4 * y);
  const g = Math.sin(7.6 * wx + 1.3) * Math.sin(6.9 * wy) + .8 * Math.sin(6.4 * wz + 7.4 * wx);
  return Math.exp(-((g / .3) ** 2));                            // 1 at the bottom of a groove, 0 on a crest
}
function field(x: number, y: number, z: number, out?: Float32Array) {
  const hemi = smin(ell(x, y, z, [0.02, 0.12, 0.35], [1.08, 0.78, 0.47]), ell(x, y, z, [0.02, 0.12, -0.35], [1.08, 0.78, 0.47]), .05);
  const temp = Math.min(ell(x, y, z, [0.30, -0.36, 0.47], [0.56, 0.30, 0.28]), ell(x, y, z, [0.30, -0.36, -0.47], [0.56, 0.30, 0.28]));
  const cortex = smin(hemi, temp, .2);
  const cereb = ell(x, y, z, [-0.80, -0.54, 0], [0.38, 0.24, 0.60]);
  const stem = ell(x, y, z, [-0.38, -0.90, 0], [0.14, 0.40, 0.14]);
  let d = smin(smin(cortex, cereb, .07), stem, .12);
  const wc = Math.min(1, Math.max(0, (cortex - cereb) / .08 + .5)), ws = Math.min(1, Math.max(0, (Math.min(cortex, cereb) - stem) / .08 + .5));
  const s1 = sulcus(x, y, z) * (1 - wc) * (1 - ws);
  const fol = Math.exp(-((Math.sin(y * 42 + x * 5) / .4) ** 2)) * wc * (1 - ws);
  const fis = Math.exp(-((z / .045) ** 2)) * Math.min(1, Math.max(0, (y + .05) / .25)) * (1 - wc);
  d += .042 * s1 + .022 * fol + .1 * fis;
  if (out) out[0] = Math.max(s1, fol * .8, fis);
  return d;
}

export function makeBrain(n: number, seed = 23): Brain3D {
  let s = seed; const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647);
  const p = new Float32Array(n * 3), nrm = new Float32Array(n * 3), fold = new Float32Array(n), o = new Float32Array(1);
  const e = 1e-3; let i = 0, guard = 0;
  while (i < n && guard++ < n * 8) {
    let x = (rnd() * 2 - 1) * 1.25, y = rnd() * 2.2 - 1.35, z = (rnd() * 2 - 1) * .95;
    let ok = false, gx = 0, gy = 0, gz = 0;
    for (let it = 0; it < 9; it++) {
      const d = field(x, y, z);
      gx = (field(x + e, y, z) - d) / e; gy = (field(x, y + e, z) - d) / e; gz = (field(x, y, z + e) - d) / e;
      const g2 = gx * gx + gy * gy + gz * gz || 1;
      x -= d * gx / g2; y -= d * gy / g2; z -= d * gz / g2;
      if (Math.abs(d) < 2e-3) { ok = true; break; }
    }
    if (!ok || y < -1.3) continue;
    const gl = Math.hypot(gx, gy, gz) || 1;
    field(x, y, z, o);
    p.set([x, y, z], i * 3); nrm.set([gx / gl, gy / gl, gz / gl], i * 3); fold[i] = o[0]; i++;
  }
  return { n: i, p, nrm, fold };
}

export interface View { cY: number; sY: number; cP: number; sP: number; cx: number; cy: number; scale: number }
export const view = (yaw: number, pitch: number, cx: number, cy: number, scale: number): View =>
  ({ cY: Math.cos(yaw), sY: Math.sin(yaw), cP: Math.cos(pitch), sP: Math.sin(pitch), cx, cy, scale });

/** out = [screenX, screenY, perspective, facing(-1..1), lambert(0..1)] */
export function project(b: Brain3D, i: number, V: View, out: Float32Array) {
  const rot = (x: number, y: number, z: number, o: number) => {
    const x1 = x * V.cY + z * V.sY, z1 = -x * V.sY + z * V.cY;
    const y2 = y * V.cP - z1 * V.sP, z2 = y * V.sP + z1 * V.cP; tmp[o] = x1; tmp[o + 1] = y2; tmp[o + 2] = z2;
  };
  rot(b.p[i * 3], b.p[i * 3 + 1], b.p[i * 3 + 2], 0); rot(b.nrm[i * 3], b.nrm[i * 3 + 1], b.nrm[i * 3 + 2], 3);
  const persp = 3.4 / (3.4 - tmp[2]);
  out[0] = V.cx + tmp[0] * V.scale * persp; out[1] = V.cy - tmp[1] * V.scale * persp; out[2] = persp;
  out[3] = tmp[5];                                               // normal towards the viewer
  out[4] = Math.max(0, tmp[3] * -.42 + tmp[4] * .62 + tmp[5] * .66); // light from upper-left-front
}
const tmp = new Float32Array(6);
