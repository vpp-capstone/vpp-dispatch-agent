import { useEffect, useState } from "react";

// Keyed requests prevent a slow response for yesterday or another ESS from replacing
// the current selection. No stale snapshot is rendered under a new resource label.
export function useApi<T>(url: string | null, revision = 0) {
  const key = `${url}:${revision}`;
  const [result, setResult] = useState<{
    key: string;
    data?: T;
    error?: string;
  }>({ key: "" });
  useEffect(() => {
    if (!url) return;
    const controller = new AbortController();
    fetch(url, { signal: controller.signal })
      .then(async (response) => {
        const body = await response.json();
        if (!response.ok)
          throw new Error(
            typeof body.detail === "string"
              ? body.detail
              : "요청을 처리할 수 없습니다. 입력값을 확인해 주세요.",
          );
        if (!controller.signal.aborted) setResult({ key, data: body as T });
      })
      .catch((e: Error) => {
        if (!controller.signal.aborted) setResult({ key, error: e.message });
      });
    return () => controller.abort();
  }, [url, key]);
  return result.key === key
    ? { ...result, loading: false }
    : { data: undefined, error: undefined, loading: !!url };
}
