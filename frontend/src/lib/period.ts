import type { Nullable, PeriodDay } from "../types";

// Executed value when the saved result has it, otherwise the plan; never mixed per field.
export const dayIncome = (d: PeriodDay): Nullable =>
  d.income ?? d.plannedIncome;
export const daySupply = (d: PeriodDay) => d.actualSupply ?? d.plannedSupply;
export const dayCharge = (d: PeriodDay) => d.charge ?? d.plannedCharge;
export const dayDischarge = (d: PeriodDay) => d.discharge ?? d.plannedDischarge;

export interface Totals {
  days: number;
  income: Nullable;
  plannedIncome: Nullable;
  fixedIncome: Nullable;
  executedIncome: Nullable;
  supply: number;
  charge: number;
  discharge: number;
  shortfall: Nullable;
}

// Totals up to and including `cursor` only: the replay must not reveal later days.
export function cumulative(days: PeriodDay[], cursor: number): Totals {
  const seen = days.slice(0, Math.max(0, Math.min(days.length, cursor + 1)));
  const incomes = seen.map(dayIncome);
  const shortfalls = seen.map((d) => d.shortfall);
  const sum = (values: Nullable[]) =>
    values.some((v) => v === null)
      ? null
      : values.reduce<number>((a, v) => a + (v as number), 0);
  return {
    days: seen.length,
    income: sum(incomes),
    plannedIncome: sum(seen.map((d) => d.plannedIncome)),
    fixedIncome: sum(seen.map((d) => d.fixedIncome)),
    executedIncome: sum(seen.map((d) => d.income)),
    supply: seen.reduce((a, d) => a + daySupply(d), 0),
    charge: seen.reduce((a, d) => a + dayCharge(d), 0),
    discharge: seen.reduce((a, d) => a + dayDischarge(d), 0),
    shortfall: shortfalls.some((v) => v === null)
      ? null
      : shortfalls.reduce<number>((a, v) => a + (v as number), 0),
  };
}

export function runningIncome(days: PeriodDay[]): Nullable[] {
  let sum: Nullable = 0;
  return days.map((d) => {
    const v = dayIncome(d);
    sum = sum === null || v === null ? null : sum + v;
    return sum;
  });
}

export function runningIncomeField(
  days: PeriodDay[],
  field: "plannedIncome" | "fixedIncome" | "income",
): Nullable[] {
  let sum: Nullable = 0;
  return days.map((d) => {
    const value = d[field];
    sum = sum === null || value === null ? null : sum + value;
    return sum;
  });
}

const iso = (d: Date) => d.toISOString().slice(0, 10);

// End date for a preset length starting at `start`, clamped to the last saved day.
export function presetEnd(
  start: string,
  preset: "week" | "month" | "quarter" | "year",
  last: string,
) {
  const d = new Date(`${start}T00:00:00Z`);
  if (preset === "week") d.setUTCDate(d.getUTCDate() + 6);
  else {
    d.setUTCMonth(d.getUTCMonth() + { month: 1, quarter: 3, year: 12 }[preset]);
    d.setUTCDate(d.getUTCDate() - 1);
  }
  const end = iso(d);
  return end > last ? last : end;
}
