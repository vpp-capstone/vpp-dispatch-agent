import { afterEach, describe, expect, it, vi } from "vitest";
import type { Catalog } from "../types";
import {
  customerKey,
  loadCustomerSelection,
  parseCustomerSelection,
  saveCustomerSelection,
} from "./customer";
import { initialWorkspace } from "./workspace";
import { pageFromHash } from "../hooks/useRoute";

const catalog: Catalog = {
  candidates: [
    {
      id: "first",
      label: "A",
      line: "line",
      capacity: 8,
      power: 2,
      npv: 1,
      income: 1,
      hasArchive: true,
    },
    {
      id: "chosen",
      label: "B",
      line: "line",
      capacity: 12,
      power: 3,
      npv: 1,
      income: 1,
      hasArchive: true,
    },
    {
      id: "report-only",
      label: "C",
      line: "line",
      capacity: 16,
      power: 4,
      npv: 1,
      income: 1,
      hasArchive: false,
    },
  ],
  portfolio: [{ id: "release", label: "Portfolio" }],
  pipelineState: "ready",
  warnings: [],
};
const chosen = {
  source: "distribution" as const,
  selection: "chosen",
  day: "2025-05-01",
};
const encode = (selection = chosen.selection) =>
  JSON.stringify({ version: 1, ...chosen, selection });

afterEach(() => vi.unstubAllGlobals());

describe("first visit and returning customer", () => {
  it("starts a first visit at welcome and restores a confirmed resource for a return visit", () => {
    const values = new Map<string, string>();
    vi.stubGlobal("localStorage", {
      getItem: (key: string) => values.get(key) ?? null,
      setItem: (key: string, value: string) => values.set(key, value),
    });
    expect(loadCustomerSelection(catalog)).toBeNull();
    expect(pageFromHash("", "welcome")).toBe("welcome");
    expect(saveCustomerSelection(chosen)).toBe(true);
    expect(values.has(customerKey)).toBe(true);
    const restored = loadCustomerSelection(catalog);
    expect(restored).toEqual(chosen);
    expect(initialWorkspace(catalog, restored)).toMatchObject(chosen);
    expect(pageFromHash("", restored ? "plan" : "welcome")).toBe("plan");
    expect(pageFromHash("#/welcome", "plan")).toBe("welcome");
    expect(pageFromHash("#/onboarding", "plan")).toBe("onboarding");
    expect(pageFromHash("#/settlement", "plan")).toBe("settlement");
  });
  it("does not restore removed, unavailable, or malformed selections", () => {
    for (const raw of [
      null,
      "broken",
      "null",
      "[]",
      encode("missing"),
      encode("report-only"),
      JSON.stringify({ version: 2, ...chosen }),
      JSON.stringify({ version: 1, ...chosen, day: null }),
      JSON.stringify({ version: 1, ...chosen, source: "portfolio" }),
    ]) {
      expect(parseCustomerSelection(raw, catalog)).toBeNull();
    }
    expect(pageFromHash("#/unknown", "welcome")).toBe("welcome");
  });
  it("keeps the customer journey usable when browser storage is blocked", () => {
    vi.stubGlobal("localStorage", {
      getItem() {
        throw new Error("blocked");
      },
      setItem() {
        throw new Error("blocked");
      },
    });
    expect(loadCustomerSelection(catalog)).toBeNull();
    expect(saveCustomerSelection(chosen)).toBe(false);
    expect(initialWorkspace(catalog, chosen)).toMatchObject(chosen);
  });
});
