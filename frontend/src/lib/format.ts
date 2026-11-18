import type { Nullable } from "../types";

export const number = (n: Nullable, digits = 1) =>
  n === null
    ? "—"
    : n.toLocaleString("ko-KR", {
        maximumFractionDigits: digits,
        minimumFractionDigits: digits,
      });
export const money = (n: Nullable) =>
  n === null
    ? "미계산"
    : Math.abs(n) >= 1e8
      ? `${number(n / 1e8, 2)}억 원`
      : `${number(n / 10000)}만 원`;
export const time = (h: number) => `${String(h).padStart(2, "0")}:00`;
export const difference = (a: Nullable, b: Nullable): Nullable =>
  a === null || b === null ? null : a - b;
export const lineLabel = (s: string) =>
  s
    .replace("cap_", "고정 상한 ")
    .replace("dyn_", "동적 상한 ")
    .replace("MW", " MW");
