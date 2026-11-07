// Sample the pixels of rendered text into points. The hero's dots ARE the word, not an image of it.
export async function sampleText(text: string, w: number, h: number, count: number, fontPx: number): Promise<Float32Array> {
  try { await document.fonts.load(`italic 400 ${fontPx}px "Instrument Serif"`); } catch { /* fall back to serif */ }
  const cv = document.createElement("canvas"); cv.width = Math.max(2, Math.floor(w)); cv.height = Math.max(2, Math.floor(h));
  const ctx = cv.getContext("2d")!;
  ctx.fillStyle = "#000"; ctx.font = `italic 400 ${fontPx}px "Instrument Serif", Georgia, serif`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillText(text, cv.width / 2, cv.height / 2);
  const { data } = ctx.getImageData(0, 0, cv.width, cv.height);
  const pts: number[] = [];
  const step = Math.max(2, Math.round(fontPx / 52));
  for (let y = 0; y < cv.height; y += step) for (let x = 0; x < cv.width; x += step) if (data[(y * cv.width + x) * 4 + 3] > 120) pts.push(x, y);
  // deterministic shuffle so the same particle always lands in the same letter
  let s = 12345; const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647);
  const n = pts.length / 2, idx = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) { const j = (rnd() * (i + 1)) | 0; [idx[i], idx[j]] = [idx[j], idx[i]]; }
  const out = new Float32Array(count * 2);
  for (let i = 0; i < count; i++) {
    const k = idx[i % n]; const jitter = i >= n ? 2.2 : 0;                  // spare particles re-use points with a little scatter
    out[i * 2] = pts[k * 2] + (rnd() - .5) * jitter; out[i * 2 + 1] = pts[k * 2 + 1] + (rnd() - .5) * jitter;
  }
  return out;
}
