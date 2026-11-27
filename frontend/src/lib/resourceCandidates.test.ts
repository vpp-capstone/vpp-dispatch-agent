import { describe, expect, it } from "vitest";
import type { Candidate } from "../types";
import { candidateView, demoComparison } from "./resourceCandidates";

const candidate = (
  id: string,
  line: string,
  hasArchive: boolean,
): Candidate => ({
  id,
  line,
  hasArchive,
  label: id,
  power: 2,
  capacity: 8,
  income: 1,
  npv: null,
  irr: null,
  discountedPaybackYears: null,
  neededAnnualSupport: null,
});
const domestic = "cap_2.50MW|kr_notice_example|2:8";
const bnef = "cap_2.50MW|new_2025_bnef|2:8";
const candidates = [
  candidate("report-only", "cap_1.50MW", false),
  candidate("other-line", "cap_2.00MW", true),
  candidate("report-only-same-line", "cap_2.50MW", false),
  candidate(bnef, "cap_2.50MW", true),
  candidate(domestic, "cap_2.50MW", true),
];

describe("demo candidates without removing the catalog", () => {
  it("defaults to two ready candidates on the same line without making the customer's decision", () => {
    expect(demoComparison(candidates)).toEqual({
      line: "cap_2.50MW",
      ids: [domestic, bnef],
      chosen: null,
    });
    const view = candidateView(candidates, "ready");
    expect(view.lines).toEqual(["cap_2.00MW", "cap_2.50MW"]);
    expect(view.rows.every((c) => c.hasArchive)).toBe(true);
  });
  it("keeps unprepared candidates accessible and falls back when switching back to the demo view", () => {
    expect(
      candidateView(candidates, "all", "cap_1.50MW").rows.map((c) => c.id),
    ).toEqual(["report-only"]);
    expect(candidateView(candidates, "all", "cap_2.50MW").rows).toHaveLength(3);
    expect(candidateView(candidates, "ready", "cap_1.50MW").line).toBe(
      "cap_2.50MW",
    );
    expect(candidates).toHaveLength(5);
  });
  it("handles a missing preferred pair or an empty archive set without pretending data exists", () => {
    expect(demoComparison([candidates[1]])).toEqual({
      line: "cap_2.00MW",
      ids: ["other-line"],
      chosen: null,
    });
    expect(demoComparison([candidates[0]])).toEqual({
      line: "",
      ids: [],
      chosen: null,
    });
    expect(candidateView([candidates[0]], "all").rows).toHaveLength(1);
  });
});
