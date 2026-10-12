"""Download public renewable CSV snapshots and retain their provenance.

Only public file downloads are used. A portal CAPTCHA stops collection.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import http.cookiejar
import json
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = "https://www.data.go.kr"
PORTALS = {
    "kpx": "15065269", "jec_solar": "15038930", "jec_wind": "15028181",
}


class Collector:
    """Collect versioned public files, with a manifest suitable for reruns."""

    def __init__(self, output: Path) -> None:
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.client = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
        )

    def request(self, url: str, params: dict | None = None) -> bytes:
        """Request one public resource with a bounded network timeout."""
        data = urllib.parse.urlencode(params).encode() if params is not None else None
        req = urllib.request.Request(url, data=data, headers={"User-Agent": "Mozilla/5.0"})
        with self.client.open(req, timeout=25) as response:
            return response.read()

    def inventory(self, source: str) -> list[dict]:
        """Read snapshot IDs from the portal's current and historical listings."""
        portal = PORTALS[source]
        url = f"{BASE}/data/{portal}/fileData.do"
        page = self.request(url).decode("utf-8")
        detail = re.search(r'id="publicDataDetailPk"[^>]*value="([^"]+)"', page)
        title = re.search(r'<meta[^>]*property="og:title"[^>]*content="([^"]+)"', page)
        if not detail:
            raise ValueError(f"{source}: no current public snapshot ID")
        current_title = html.unescape(title.group(1)) if title else source
        entries = [{"detail": detail.group(1), "title": current_title}]
        history = self.request(BASE + "/tcs/dss/selectHistAndCsvData.do", {
            "publicDataPk": portal, "publicDataDetailPk": detail.group(1),
        }).decode("utf-8")
        hits = re.findall(
            r'data-public-pk="([^"]+)"\s+data-public-detail-sn="([^"]+)"[^>]*>\s*([^<]+)',
            history,
        )
        entries.extend({"detail": key, "title": html.unescape(name.strip())}
                       for key, _, name in hits)
        unique = {entry["detail"]: entry for entry in entries}
        inventory = list(unique.values())
        (self.output / f"{source}_inventory.json").write_text(
            json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        return inventory

    def download(self, source: str, entry: dict) -> dict:
        """Download a CSV or ZIP, reusing only checksum-verified cached files."""
        portal = PORTALS[source]
        detail = entry["detail"]
        safe_detail = re.sub(r"[^a-zA-Z0-9_-]", "_", detail)
        date_match = re.search(r"(20\d{6})", entry["title"])
        publication_date = date_match.group(1) if date_match else "undated"
        stem = f"{source}_{publication_date}_{safe_detail}"
        meta = self.output / f"{stem}.json"
        if meta.exists():
            cached = json.loads(meta.read_text(encoding="utf-8"))
            path = self.output / cached["file"]
            if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == cached["sha256"]:
                return cached
        info = json.loads(self.request(BASE + "/tcs/dss/selectFileDataDownload.do", {
            "publicDataPk": portal, "publicDataDetailPk": detail,
            "atchFileId": "", "fileDetailSn": "1", "publicDataTyCode": "PR0051",
        }))
        if not info.get("status"):
            raise ValueError(f"{source}/{detail}: download lookup failed")
        params = {"atchFileId": info["atchFileId"], "fileDetailSn": info["fileDetailSn"]}
        limit = json.loads(self.request(BASE + "/cmm/cmm/check-limit.json", params))
        if limit.get("needCaptcha"):
            raise RuntimeError("Portal requires a CAPTCHA; collection stopped")
        url = BASE + "/cmm/cmm/fileDownload.do?" + urllib.parse.urlencode({
            **params, "insertDataPrcus": "N",
        })
        data = self.request(url)
        if not data or data.lstrip().lower().startswith((b"<!doctype", b"<html")):
            raise ValueError(f"{source}/{detail}: response is not a data file")
        extension = ".zip" if data.startswith(b"PK\x03\x04") else ".csv"
        path = self.output / f"{stem}{extension}"
        path.write_bytes(data)
        record = {
            **entry, "source": source, "portal_id": portal,
            "source_url": f"{BASE}/data/{portal}/fileData.do",
            "download_url": url, "file": path.name, "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        }
        meta.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return record


def main() -> None:
    """List or download the snapshots within a source publication-date range."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/renewable_sources"))
    parser.add_argument("--source", choices=list(PORTALS), action="append")
    parser.add_argument("--since", default="20210101")
    parser.add_argument("--until", default="20261231")
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    collector = Collector(args.output_dir)
    for source in args.source or list(PORTALS):
        entries = collector.inventory(source)
        for entry in entries:
            match = re.search(r"(20\d{6})", entry["title"])
            if not match or not args.since <= match.group(1) <= args.until:
                continue
            print(json.dumps({"source": source, **entry}, ensure_ascii=False), flush=True)
            if not args.inventory_only:
                record = collector.download(source, entry)
                print(f"Saved {record['file']} ({record['bytes']} bytes)", flush=True)


if __name__ == "__main__":
    main()
