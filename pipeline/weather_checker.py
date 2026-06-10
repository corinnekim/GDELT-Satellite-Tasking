"""
Level 2a, 기상 판별기
Open-Meteo API로 위성 통과 시점의 구름량과 주/야간 상태를 판별한다.
- 미래/현재 pass: forecast endpoint (최대 16일)
- 과거 pass: archive endpoint (ERA5 reanalysis, 1940~현재-5일)
- API 키 불필요
"""
from __future__ import annotations

import json
import sqlite3
import threading
import requests
from datetime import datetime, timezone, timedelta
from functools import lru_cache
from pathlib import Path

from astral import LocationInfo
from astral.sun import sun

from pipeline.config import CLOUD_THRESHOLD, PROJECT_ROOT


# Open-Meteo 엔드포인트
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# 디스크 영구 캐시 (cross-process, cross-day)
# 동일 (lat_r, lon_r, date)에 대한 archive 응답을 SQLite에 저장.
_WEATHER_CACHE_PATH = PROJECT_ROOT / "cache" / "weather_cache.sqlite"
_weather_lock = threading.local()


def _weather_conn() -> sqlite3.Connection:
    if getattr(_weather_lock, "conn", None) is None:
        _WEATHER_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(str(_WEATHER_CACHE_PATH), timeout=30.0, check_same_thread=False)
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        c.execute("""
            CREATE TABLE IF NOT EXISTS archive (
                key TEXT PRIMARY KEY,
                cloud_map TEXT NOT NULL,
                fetched_at REAL NOT NULL
            )
        """)
        c.commit()
        _weather_lock.conn = c
    return _weather_lock.conn


def _archive_cache_get(lat_r: float, lon_r: float, date_str: str) -> dict | None:
    key = f"{lat_r:.2f}|{lon_r:.2f}|{date_str}"
    row = _weather_conn().execute(
        "SELECT cloud_map FROM archive WHERE key = ?", (key,)
    ).fetchone()
    if row is None:
        return None
    return {"cloud_map": json.loads(row[0])}


def _archive_cache_set(lat_r: float, lon_r: float, date_str: str, cloud_map: dict) -> None:
    key = f"{lat_r:.2f}|{lon_r:.2f}|{date_str}"
    import time
    _weather_conn().execute(
        "INSERT OR REPLACE INTO archive (key, cloud_map, fetched_at) VALUES (?, ?, ?)",
        (key, json.dumps(cloud_map), time.time()),
    )
    _weather_conn().commit()


def _normalize_sensor_type(sensor: str | None) -> str:
    """센서 타입 표기를 소문자 기준 공통 포맷으로 맞춘다."""
    return str(sensor or "optical").strip().lower()


def _build_cloud_map(data: dict) -> dict:
    """응답의 hourly 배열을 {time: cloud_cover} HashMap으로 변환."""
    if "hourly" in data:
        times = data["hourly"]["time"]
        clouds = data["hourly"]["cloud_cover"]
        data["cloud_map"] = dict(zip(times, clouds))
    return data


@lru_cache(maxsize=64)
def _fetch_cloud_forecast(lat: float, lon: float) -> dict | None:
    """Open-Meteo forecast(16일)에서 시간별 구름량을 가져온다. 좌표별 캐싱."""
    lat_r = round(lat, 2)
    lon_r = round(lon, 2)
    try:
        resp = requests.get(
            OPEN_METEO_URL,
            params={
                "latitude": lat_r,
                "longitude": lon_r,
                "hourly": "cloud_cover",
                "timezone": "UTC",
            },
            timeout=15,
        )
        resp.raise_for_status()
        return _build_cloud_map(resp.json())
    except requests.RequestException as e:
        print(f"  [WEATHER] ⚠️ Open-Meteo forecast 실패 ({lat_r}, {lon_r}): {e}")
        return None


@lru_cache(maxsize=512)
def _fetch_cloud_archive(lat: float, lon: float, date_str: str) -> dict | None:
    """Open-Meteo archive(ERA5)에서 특정 날짜의 시간별 구름량을 가져온다.
    LRU(인메모리) + SQLite(디스크) 2단 캐시, 한 번 호출한 (lat, lon, date)는 영구 보존."""
    lat_r = round(lat, 2)
    lon_r = round(lon, 2)
    cached = _archive_cache_get(lat_r, lon_r, date_str)
    if cached is not None:
        return cached
    try:
        resp = requests.get(
            OPEN_METEO_ARCHIVE_URL,
            params={
                "latitude": lat_r,
                "longitude": lon_r,
                "start_date": date_str,
                "end_date": date_str,
                "hourly": "cloud_cover",
                "timezone": "UTC",
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = _build_cloud_map(resp.json())
        if data and "cloud_map" in data:
            _archive_cache_set(lat_r, lon_r, date_str, data["cloud_map"])
        return data
    except requests.RequestException as e:
        print(f"  [WEATHER] ⚠️ Open-Meteo archive 실패 ({lat_r}, {lon_r}, {date_str}): {e}")
        return None


def _archive_lag_days() -> int:
    """Open-Meteo archive(ERA5)의 데이터 지연 버퍼(일)."""
    return 5


def get_cloud_cover(lat: float, lon: float, pass_time_utc: str) -> dict:
    """특정 시각의 구름량 정보를 반환한다.

    - pass 날짜가 (오늘 - 5일) 이전이면 archive(ERA5) 사용
    - 그 외는 forecast 사용
    """
    pass_dt = datetime.fromisoformat(pass_time_utc.replace("Z", "+00:00"))

    # 30분 이상이면 다음 시간 정각으로 올림 (정각 매칭)
    if pass_dt.minute >= 30:
        pass_dt += timedelta(hours=1)
    pass_dt = pass_dt.replace(minute=0, second=0, microsecond=0)

    pass_date = pass_dt.date()
    today_utc = datetime.now(timezone.utc).date()
    archive_cutoff = today_utc - timedelta(days=_archive_lag_days())

    if pass_date <= archive_cutoff:
        data = _fetch_cloud_archive(lat, lon, pass_date.strftime("%Y-%m-%d"))
    else:
        data = _fetch_cloud_forecast(lat, lon)

    if data is None or "cloud_map" not in data:
        return {
            "cloud_cover_pct": -1,
            "cloud_status": "unknown",
            "shootable_eo": False,
        }

    target_hour = pass_dt.strftime("%Y-%m-%dT%H:00")
    cloud_pct = data["cloud_map"].get(target_hour, -1)

    # 구름 상태 판정
    if cloud_pct < 0:
        status = "unknown"
        shootable = False
    elif cloud_pct <= 20:
        status = "clear"
        shootable = True
    elif cloud_pct <= CLOUD_THRESHOLD:
        status = "partial"
        shootable = True
    else:
        status = "overcast"
        shootable = False

    return {
        "cloud_cover_pct": cloud_pct,
        "cloud_status": status,
        "shootable_eo": shootable,
    }


def is_daylight(lat: float, lon: float, pass_time_utc: str) -> bool:
    """통과 시각이 주간(일출~일몰)인지 판별한다."""
    pass_dt = datetime.fromisoformat(pass_time_utc.replace("Z", "+00:00"))

    loc = LocationInfo(latitude=lat, longitude=lon)
    try:
        s = sun(loc.observer, date=pass_dt.date(), tzinfo=timezone.utc)
        return s["sunrise"] <= pass_dt <= s["sunset"]
    except Exception:
        # 극지방 등 일출/일몰 계산 불가 시 주간으로 간주
        return True


def check_weather(pass_event: dict) -> dict:
    """통과 이벤트에 구름량 + 주야간 정보를 추가한다.

    Args:
        pass_event: pass_predictor에서 반환된 통과 이벤트 dict

    Returns:
        원본 dict에 weather 필드가 추가된 새 dict
    """
    lat = pass_event["lat"]
    lon = pass_event["lon"]
    pass_time = pass_event["pass_time_utc"]
    sensor = _normalize_sensor_type(pass_event["sensor_type"])

    cloud = get_cloud_cover(lat, lon, pass_time)
    daylight = is_daylight(lat, lon, pass_time)

    # SAR은 구름·야간 무관하게 항상 촬영 가능
    if sensor == "sar":
        shootable = True
    else:
        shootable = cloud["shootable_eo"] and daylight

    result = pass_event.copy()
    result.update({
        "sensor_type": sensor,
        "cloud_cover_pct": cloud["cloud_cover_pct"],
        "cloud_status": cloud["cloud_status"],
        "daylight": daylight,
        "shootable": shootable,
    })

    return result


# CLI 실행
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="기상 판별기")
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    args = parser.parse_args()

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    cloud = get_cloud_cover(args.lat, args.lon, now_utc)
    daylight = is_daylight(args.lat, args.lon, now_utc)

    print(f"\n  ── 기상 판별 결과 ──")
    print(f"  좌표: ({args.lat}, {args.lon})")
    print(f"  시각: {now_utc}")
    print(f"  구름량: {cloud['cloud_cover_pct']}%  ({cloud['cloud_status']})")
    print(f"  주간 여부: {'☀️ 주간' if daylight else '🌙 야간'}")
    print(f"  EO 촬영 가능: {'✅' if cloud['shootable_eo'] and daylight else '❌'}")
