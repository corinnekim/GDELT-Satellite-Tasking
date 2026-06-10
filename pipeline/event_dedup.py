"""
도시별 검증 이벤트 아카이브 + 중복 감지.

목적: 같은 도시가 며칠 뒤 다시 Kalman 이상징후로 잡혔을 때,
      동일한 소스 URL(같은 사건 기사)이면 위성 스케줄링을 건너뛰고
      사람이 확인할 수 있도록 로그만 남긴다.

Archive 구조: logs/verified_events.json
{
  "Minab": [
    {"date": "20260228", "status": "SUCCESS",
     "source_urls": ["https://...", ...],
     "llm_report": "{...}"}
  ]
}
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timedelta

from pipeline.config import PROJECT_ROOT

ARCHIVE_PATH = PROJECT_ROOT / "logs" / "verified_events.json"
LOOKBACK_DAYS = 14  # 이 기간 내 이벤트만 중복 판단 후보


def load_archive() -> dict:
    if not ARCHIVE_PATH.exists():
        return {}
    try:
        with open(ARCHIVE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_archive(archive: dict) -> None:
    ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ARCHIVE_PATH, "w", encoding="utf-8") as f:
        json.dump(archive, f, ensure_ascii=False, indent=2)


def _within_lookback(prior_date: str, target_date: str) -> bool:
    """prior_date가 target_date로부터 LOOKBACK_DAYS 이내인지."""
    try:
        prior = datetime.strptime(prior_date, "%Y%m%d").date()
        target = datetime.strptime(target_date, "%Y%m%d").date()
    except ValueError:
        return False
    delta = (target - prior).days
    return 0 < delta <= LOOKBACK_DAYS


def find_duplicates(city: str, current_urls: list[str], target_date: str,
                    archive: dict) -> list[dict]:
    """현재 URL이 아카이브의 같은 도시 이벤트와 겹치는지 확인.

    Returns: 겹치는 과거 이벤트 리스트 (빈 리스트면 중복 아님)
    """
    if not current_urls:
        return []
    current_set = set(current_urls)
    matches = []
    for entry in archive.get(city, []):
        if not _within_lookback(entry.get("date", ""), target_date):
            continue
        prior_urls = set(entry.get("source_urls", []) or [])
        overlap = current_set & prior_urls
        if overlap:
            matches.append({
                "prior_date": entry["date"],
                "prior_status": entry.get("status"),
                "overlap_urls": sorted(overlap),
                "prior_report": entry.get("llm_report", ""),
            })
    return matches


def archive_verified(city: str, target_date: str, status: str,
                     source_urls: list[str], llm_report: str,
                     archive: dict) -> None:
    """SUCCESS/AMBIGUOUS 이벤트를 아카이브에 추가 (같은 (city, date)면 덮어쓰기)."""
    entries = archive.setdefault(city, [])
    for e in entries:
        if e.get("date") == target_date:
            e["status"] = status
            e["source_urls"] = source_urls
            e["llm_report"] = llm_report
            return
    entries.append({
        "date": target_date,
        "status": status,
        "source_urls": list(source_urls or []),
        "llm_report": llm_report,
    })
