import { useEffect, useState } from "react";
interface Review {
  memo: string;
  complete: boolean;
}
const empty = { memo: "", complete: false };
function read(key: string): Review {
  try {
    const saved = JSON.parse(localStorage.getItem(key) ?? "null");
    return saved &&
      typeof saved.memo === "string" &&
      typeof saved.complete === "boolean"
      ? { memo: saved.memo.slice(0, 500), complete: saved.complete }
      : empty;
  } catch {
    return empty;
  }
}
export function useReview(snapshot: string) {
  const key = `jeju-flow:review:${snapshot}`;
  const [entry, setEntry] = useState({ key, value: read(key) });
  const [error, setError] = useState("");
  useEffect(() => {
    setEntry({ key, value: read(key) });
    setError("");
  }, [key]);
  const value = entry.key === key ? entry.value : read(key);
  const save = (next: Review) => {
    try {
      localStorage.setItem(key, JSON.stringify(next));
      setEntry({ key, value: next });
      setError("");
    } catch {
      setError("이 브라우저에서 검토 기록을 저장할 수 없습니다.");
    }
  };
  return { value, save, error };
}
