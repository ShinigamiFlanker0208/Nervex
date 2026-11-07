// Line diff (LCS). Drafts are a few hundred lines, so O(n*m) is fine and needs no dependency.
export type DiffLine = { kind: "same" | "add" | "del"; text: string; a?: number; b?: number };

export function diffLines(oldText: string, newText: string): DiffLine[] {
  const A = oldText.split("\n"), B = newText.split("\n"), n = A.length, m = B.length;
  if (n * m > 400_000) return [{ kind: "same", text: "(drafts too large to diff in the browser)" }];
  const L = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--) L[i][j] = A[i] === B[j] ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1]);
  const out: DiffLine[] = []; let i = 0, j = 0;
  while (i < n && j < m) {
    if (A[i] === B[j]) { out.push({ kind: "same", text: A[i], a: i + 1, b: j + 1 }); i++; j++; }
    else if (L[i + 1][j] >= L[i][j + 1]) { out.push({ kind: "del", text: A[i], a: i + 1 }); i++; }
    else { out.push({ kind: "add", text: B[j], b: j + 1 }); j++; }
  }
  while (i < n) out.push({ kind: "del", text: A[i], a: ++i });
  while (j < m) out.push({ kind: "add", text: B[j], b: ++j });
  return out;
}

/** Keep only changed lines plus `ctx` lines of context around them; long unchanged runs become a fold marker. */
export function hunks(d: DiffLine[], ctx = 2): (DiffLine | { kind: "fold"; n: number })[] {
  const keep = d.map(x => x.kind !== "same");
  const show = d.map((_, i) => { for (let k = Math.max(0, i - ctx); k <= Math.min(d.length - 1, i + ctx); k++) if (keep[k]) return true; return false; });
  const out: (DiffLine | { kind: "fold"; n: number })[] = []; let fold = 0;
  d.forEach((x, i) => { if (show[i]) { if (fold) { out.push({ kind: "fold", n: fold }); fold = 0; } out.push(x); } else fold++; });
  if (fold) out.push({ kind: "fold", n: fold });
  return out;
}
