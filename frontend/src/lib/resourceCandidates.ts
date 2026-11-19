import type { Candidate } from "../types";

export type CandidateScope = "ready" | "all";

export function candidateView(
  candidates: Candidate[],
  scope: CandidateScope,
  requestedLine?: string,
) {
  const available = candidates.filter((c) => scope === "all" || c.hasArchive);
  const lines = [...new Set(available.map((c) => c.line))];
  const line = lines.includes(requestedLine ?? "")
    ? requestedLine!
    : lines.includes("cap_2.50MW")
      ? "cap_2.50MW"
      : (lines[0] ?? "");
  return { lines, line, rows: available.filter((c) => c.line === line) };
}

export function demoComparison(candidates: Candidate[]) {
  const view = candidateView(candidates, "ready", "cap_2.50MW");
  const preferred = [
    "cap_2.50MW|kr_notice_example|2:8",
    "cap_2.50MW|new_2025_bnef|2:8",
  ];
  const rows = [...view.rows].sort((a, b) => {
    const rank = (id: string) =>
      preferred.includes(id) ? preferred.indexOf(id) : preferred.length;
    return rank(a.id) - rank(b.id);
  });
  return {
    line: view.line,
    ids: rows.slice(0, 2).map((c) => c.id),
    chosen: null,
  };
}
