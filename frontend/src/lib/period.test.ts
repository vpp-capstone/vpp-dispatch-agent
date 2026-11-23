import { describe, it, expect } from "vitest";
import { cumulative, presetEnd, runningIncome } from "./period";
import type { PeriodDay } from "../types";

const day = (
  date: string,
  income: number | null,
  planned: number | null = null,
): PeriodDay => ({
  date,
  plannedSupply: 10,
  actualSupply: 9,
  plannedCharge: 2,
  plannedDischarge: 1,
  charge: 1.5,
  discharge: 1,
  plannedSoc: 50,
  soc: 40,
  shortfall: 0.5,
  plannedIncome: planned,
  income,
});

describe("period replay totals", () => {
  const days = [
    day("2025-01-01", 100),
    day("2025-01-02", 200),
    day("2025-01-03", 300),
  ];
  it("accumulates only up to the replay cursor", () => {
    expect(cumulative(days, 1)).toMatchObject({
      days: 2,
      income: 300,
      supply: 18,
      charge: 3,
      discharge: 2,
      shortfall: 1,
    });
    expect(cumulative(days, -1).days).toBe(0);
    expect(cumulative(days, 99).income).toBe(600);
  });
  it("uses the plan when no execution exists and keeps missing values missing", () => {
    expect(cumulative([day("2025-01-01", null, 50)], 0).income).toBe(50);
    expect(
      cumulative([day("2025-01-01", 10), day("2025-01-02", null)], 1).income,
    ).toBeNull();
    expect(runningIncome([day("a", 1), day("b", null), day("c", 3)])).toEqual([
      1,
      null,
      null,
    ]);
  });
  it("clamps preset ranges to the saved year", () => {
    expect(presetEnd("2025-01-01", "week", "2025-12-31")).toBe("2025-01-07");
    expect(presetEnd("2025-01-31", "month", "2025-12-31")).toBe("2025-03-02");
    expect(presetEnd("2025-04-01", "quarter", "2025-12-31")).toBe("2025-06-30");
    expect(presetEnd("2025-06-01", "year", "2025-12-31")).toBe("2025-12-31");
  });
});
