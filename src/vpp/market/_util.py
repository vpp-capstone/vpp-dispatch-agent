"""market 모듈들이 같이 쓰는 작은 도구."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_csv_any(path: Path, **kwargs) -> pd.DataFrame:
    """공공데이터포털 CSV 는 cp949 가 대부분이고 일부는 utf-8 이라 둘 다 시도한다."""
    for encoding in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"인코딩을 알 수 없음: {path.name}")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"저장: {path}")


def print_coverage(df: pd.DataFrame, freq: str) -> None:
    """기간, 행 수, 컬럼별로 값이 있는 구간과 그 사이 빈 값 개수를 출력한다."""
    full = pd.date_range(df["ts"].min(), df["ts"].max(), freq=freq)
    missing_rows = len(full.difference(df["ts"]))
    print(f"기간 {df['ts'].min()} ~ {df['ts'].max()}, {len(df)}행, 빠진 구간 {missing_rows}개")
    for col in df.columns.drop("ts"):
        have = df.loc[df[col].notna(), "ts"]
        if have.empty:
            print(f"  {col:<17} 값 없음")
            continue
        inside = df[(df["ts"] >= have.min()) & (df["ts"] <= have.max())][col].isna().sum()
        print(f"  {col:<17} {have.min()} ~ {have.max()} (그 사이 빈 값 {int(inside)})")
