export type Source = "portfolio" | "distribution";
export type Page =
  | "welcome"
  | "onboarding"
  | "overview"
  | "plan"
  | "settlement"
  | "resources";
export type Nullable = number | null;
export interface Candidate {
  id: string;
  label: string;
  line: string;
  capacity: number;
  power: number;
  npv: Nullable;
  irr: Nullable;
  discountedPaybackYears: Nullable;
  neededAnnualSupport: Nullable;
  income: number;
  hasArchive: boolean;
  archiveIssue?: string | null;
}
export interface Catalog {
  portfolio: { id: string; label: string }[];
  candidates: Candidate[];
  pipelineState: string;
  warnings: string[];
}
export interface Hour {
  hour: number;
  forecastGeneration: number;
  actualGeneration: Nullable;
  plannedSupply: number;
  actualSupply: Nullable;
  plannedCharge: number;
  plannedDischarge: number;
  charge: Nullable;
  discharge: Nullable;
  plannedSoc: number;
  soc: Nullable;
  shortfall: Nullable;
  fixedSupply: Nullable;
  fixedCharge: Nullable;
  fixedDischarge: Nullable;
  fixedSoc: Nullable;
  fixedShortfall: Nullable;
  plannedIncome: Nullable;
  fixedIncome: Nullable;
  income: Nullable;
}
export interface MoneyRow {
  label: string;
  planned: Nullable;
  fixed: Nullable;
  actual: Nullable;
}
export interface Snapshot {
  id: string;
  source: Source;
  selection: string;
  title: string;
  date: string;
  dates: string[];
  resource: { label: string; capacity: number; power: number; basis: string };
  basis: string;
  incomeLabel: string;
  incomeNote: string;
  executionAvailable: boolean;
  hours: Hour[];
  ledger: MoneyRow[];
  plannedIncome: Nullable;
  fixedPlanIncome: Nullable;
  actualIncome: Nullable;
  cumulativePlannedIncome: Nullable;
  cumulativeFixedIncome: Nullable;
  cumulativeIncome: Nullable;
  trend: { month: string; planned: Nullable; fixed: Nullable; actual: Nullable }[];
  warnings: string[];
  provenance: {
    snapshot: string;
    operation: string;
    basis: string;
    outputHash: string;
    report: string;
  };
}
export interface PeriodDay {
  date: string;
  plannedSupply: number;
  actualSupply: Nullable;
  plannedCharge: number;
  plannedDischarge: number;
  charge: Nullable;
  discharge: Nullable;
  plannedSoc: Nullable;
  soc: Nullable;
  shortfall: Nullable;
  fixedSupply: Nullable;
  fixedCharge: Nullable;
  fixedDischarge: Nullable;
  fixedSoc: Nullable;
  fixedShortfall: Nullable;
  plannedIncome: Nullable;
  fixedIncome: Nullable;
  income: Nullable;
}
export interface Period {
  source: Source;
  selection: string;
  start: string;
  end: string;
  dates: string[];
  incomeLabel: string;
  executionAvailable: boolean;
  days: PeriodDay[];
  snapshot: string;
}
