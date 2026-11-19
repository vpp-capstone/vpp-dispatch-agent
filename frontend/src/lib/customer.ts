import type { Catalog, Source } from "../types";

export interface CustomerSelection {
  source: Source;
  selection: string;
  day: string;
}

export const customerKey = "jeju-flow:operating-selection:v1";

export function parseCustomerSelection(
  raw: string | null,
  catalog: Catalog,
): CustomerSelection | null {
  try {
    const value = JSON.parse(raw ?? "null");
    if (!value || value.version !== 1) return null;
    const available =
      value.source === "distribution"
        ? catalog.candidates.some(
            (c) => c.id === value.selection && c.hasArchive,
          )
        : value.source === "portfolio" &&
          catalog.portfolio.some((p) => p.id === value.selection);
    if (!available || !/^\d{4}-\d{2}-\d{2}$/.test(value.day)) return null;
    return { source: value.source, selection: value.selection, day: value.day };
  } catch {
    return null;
  }
}

export function loadCustomerSelection(catalog: Catalog) {
  try {
    return parseCustomerSelection(localStorage.getItem(customerKey), catalog);
  } catch {
    return null;
  }
}

export function saveCustomerSelection(selection: CustomerSelection): boolean {
  try {
    localStorage.setItem(
      customerKey,
      JSON.stringify({ version: 1, ...selection }),
    );
    return true;
  } catch {
    return false;
  }
}
