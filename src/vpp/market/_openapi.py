"""공공데이터포털 KPX OpenAPI 공통 처리: 인증키, 호출, 응답 해석, 페이지 수집.

두 API(day_ahead, realtime_api) 모두 date 를 비우고 부르면 전체 이력이 최신 날짜부터
나온다. 그래서 날짜별로 부르지 않고 페이지를 넘기며 받고, 응답 원본을 그대로 저장한다.
개발계정은 하루 100회까지만 부를 수 있어서, 한도에 걸리면 멈췄다가 다음 실행 때
받아둔 페이지 다음부터 이어 받는다.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

ENV_KEY = "DATA_GO_KR_API_KEY"
PAGE_SIZES = (10000, 1000, 100)  # 큰 것부터 시도해서 서버가 받아주는 크기를 쓴다

Fetch = Callable[[int, int], str]  # (pageNo, numOfRows) -> 응답 본문


class QuotaExceededError(Exception):
    """일일 호출 한도 초과."""


def load_key(env_file: Path = Path(".env")) -> str:
    """인증키를 환경변수에서, 없으면 .env 파일에서 읽는다."""
    key = os.environ.get(ENV_KEY, "").strip()
    if not key and env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == ENV_KEY:
                key = value.strip().strip("\"'")
    if not key:
        raise RuntimeError(f"{ENV_KEY} 가 없습니다. .env 에 인증키를 넣어주세요.")
    return key


def make_fetch(endpoint: str, key: str) -> Fetch:
    """endpoint 를 부르는 함수를 만든다."""
    # 포털의 키는 이미 URL 인코딩된 형태(%2F, %3D)라 그대로 붙인다.
    if "%" not in key:
        key = urllib.parse.quote(key, safe="")

    def fetch(page_no: int, num_rows: int) -> str:
        query = urllib.parse.urlencode(
            {"pageNo": page_no, "numOfRows": num_rows, "dataType": "json"}
        )
        url = f"{endpoint}?serviceKey={key}&{query}"
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError as err:
            if err.code == 429:
                raise QuotaExceededError from err
            return err.read().decode("utf-8", errors="replace")

    return fetch


def parse(text: str) -> tuple[list[dict], int]:
    """응답에서 (행 목록, 전체 행 수)를 꺼낸다. 한도 초과면 QuotaExceededError."""
    if "LIMITED_NUMBER_OF_SERVICE_REQUESTS" in text:
        raise QuotaExceededError
    try:
        response = json.loads(text)["response"]
    except (ValueError, KeyError) as err:
        raise RuntimeError(f"예상과 다른 응답: {text[:300]}") from err
    header = response.get("header", {})
    if header.get("resultCode") == "22":
        raise QuotaExceededError
    if header.get("resultCode") != "00":
        raise RuntimeError(f"API 오류: {header}")
    body = response["body"]
    items = body.get("items") or {}
    rows = items.get("item", []) if isinstance(items, dict) else []
    if isinstance(rows, dict):
        rows = [rows]
    return rows, int(body["totalCount"])


def collect(fetch: Fetch, raw_dir: Path, pause: float = 0.3) -> bool:
    """받을 수 있는 만큼 받아 raw_dir/page_0001.json ... 으로 저장한다. 다 받았으면 True."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta_path = raw_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    try:
        if "page_size" not in meta:
            for size in PAGE_SIZES:
                text = fetch(1, size)
                try:
                    rows, total = parse(text)
                except RuntimeError as err:
                    print(f"numOfRows={size} 실패, 더 작게 시도: {err}")
                    continue
                if not rows:
                    continue
                # 서버가 요청보다 적게 주면 그 개수가 실제 페이지 크기다.
                meta = {"page_size": min(size, len(rows)), "total": total}
                (raw_dir / "page_0001.json").write_text(text, encoding="utf-8")
                meta_path.write_text(json.dumps(meta))
                print(f"페이지 크기 {meta['page_size']}행, 전체 {total}행")
                break
            else:
                raise RuntimeError("어떤 페이지 크기로도 응답을 받지 못했습니다.")

        size = meta["page_size"]
        page = len(list(raw_dir.glob("page_*.json"))) + 1
        while (page - 1) * size < meta["total"]:
            text = fetch(page, size)
            rows, total = parse(text)
            if not rows:
                break
            (raw_dir / f"page_{page:04d}.json").write_text(text, encoding="utf-8")
            print(f"page {page}: {len(rows)}행 (전체 {total}행)")
            meta["total"] = total
            page += 1
            time.sleep(pause)
    except QuotaExceededError:
        print("일일 호출 한도에 걸렸습니다. 내일 다시 실행하면 이어서 받습니다.")
        return False
    return True


def read_pages(raw_dir: Path) -> list[dict]:
    """저장해 둔 페이지들의 행을 모두 읽는다."""
    rows: list[dict] = []
    for path in sorted(raw_dir.glob("page_*.json")):
        rows += parse(path.read_text(encoding="utf-8"))[0]
    if not rows:
        raise FileNotFoundError(f"{raw_dir} 에 받아둔 응답이 없습니다.")
    return rows
