// `# Implements: <Node_ID: multi_head_attention>` comments are the bridge between code and graph.
export interface Tag { line: number; nodeId: string }

const RE = /#\s*Implements:.*?Node_ID:\s*([A-Za-z0-9_.\-]+)/i;

/** 1-based line numbers. The tag also covers the next non-comment line (what it annotates). */
export function parseTags(code: string): Tag[] {
  const lines = code.split("\n");
  const out: Tag[] = [];
  lines.forEach((l, i) => { const m = RE.exec(l); if (m) out.push({ line: i + 1, nodeId: m[1] }); });
  return out;
}

/** nodeId whose tag governs `line` (nearest tag above, within the same block), or null. */
export function tagForLine(tags: Tag[], lines: string[], line: number): string | null {
  let best: Tag | null = null;
  for (const t of tags) if (t.line <= line && (!best || t.line > best.line)) best = t;
  if (!best) return null;
  for (let i = best.line; i < line - 1; i++) if (lines[i] !== undefined && lines[i].trim() === "") return null;
  return best.nodeId;
}

/** `[B, S, D] -> [B, S, D]` shape comments for the margin. */
export const SHAPE_RE = /#\s*(\[[^\]]+\]\s*->\s*\[[^\]]+\])/;
