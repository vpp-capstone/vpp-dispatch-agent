import { useEffect, useState } from "react";
import type { Page } from "../types";
const pages: Page[] = [
  "welcome",
  "onboarding",
  "overview",
  "plan",
  "settlement",
  "resources",
];
export const pageFromHash = (hash: string, fallback: Page): Page =>
  pages.find((p) => hash === `#/${p}`) ?? fallback;
export function useRoute(fallback: Page) {
  const [page, setPage] = useState<Page>(() =>
    pageFromHash(location.hash, fallback),
  );
  useEffect(() => {
    const update = () => setPage(pageFromHash(location.hash, fallback));
    addEventListener("hashchange", update);
    return () => removeEventListener("hashchange", update);
  }, [fallback]);
  return {
    page,
    navigate: (next: Page) => {
      location.hash = `/${next}`;
      setPage(next);
    },
  };
}
