"""
Level 2a, TLE 수집기
- operational 모드: CelesTrak GP API에서 최신 TLE 수집
- backtest 모드: Space-Track gp_history에서 요청 날짜 기준 과거 TLE 수집
양쪽 모두 결과는 data/tle/tle_YYYYMMDD.json 캐시에 저장된다.
"""
from __future__ import annotations

import requests
import json
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from pipeline.config import TLE_CACHE_DIR, SPACETRACK_USER, SPACETRACK_PASSWORD
from pipeline.satellite_catalog import load_satellite_catalog, DEFAULT_SATELLITES as SATELLITES


# API 엔드포인트
CELESTRAK_GP_URL = "https://celestrak.org/NORAD/elements/gp.php"
SPACETRACK_LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
SPACETRACK_QUERY_URL = "https://www.space-track.org/basicspacedata/query/class/gp_history"


def fetch_tle(norad_id: int, session: requests.Session | None = None) -> tuple[int, tuple[str, str, str] | None]:
    """CelesTrak에서 단일 위성의 TLE를 가져온다.

    Returns:
        (norad_id, (name, line1, line2) 또는 실패 시 None)
    """
    try:
        req_func = session.get if session else requests.get
        resp = req_func(
            CELESTRAK_GP_URL,
            params={"CATNR": norad_id, "FORMAT": "TLE"},
            timeout=15,
        )
        resp.raise_for_status()

        lines = [l.strip() for l in resp.text.strip().split("\n") if l.strip()]
        if len(lines) < 3:
            print(f"  [TLE] NORAD {norad_id}: 데이터 불충분 ({len(lines)}줄)")
            return norad_id, None

        name, line1, line2 = lines[0], lines[1], lines[2]
        return norad_id, (name, line1, line2)

    except requests.RequestException as e:
        print(f"  [TLE] NORAD {norad_id}: 요청 실패 - {e}")
        return norad_id, None


def _spacetrack_login(session: requests.Session) -> None:
    """Space-Track 세션 로그인. 쿠키 기반."""
    if not SPACETRACK_USER or not SPACETRACK_PASSWORD:
        raise RuntimeError(
            "SPACETRACK_USER / SPACETRACK_PASSWORD 환경변수가 필요합니다. .env를 확인하세요."
        )
    resp = session.post(
        SPACETRACK_LOGIN_URL,
        data={"identity": SPACETRACK_USER, "password": SPACETRACK_PASSWORD},
        timeout=20,
    )
    resp.raise_for_status()


def fetch_historical_tle(
    norad_id: int,
    target_date: str,
    session: requests.Session,
    lookback_days: int = 3,
) -> tuple[str, str, str] | None:
    """Space-Track에서 target_date(YYYYMMDD) 기준 가장 가까운 과거 TLE를 반환한다.

    전략: [target - lookback_days, target + 1일] 범위에서 EPOCH desc로 받아,
    target 날짜 종료(23:59:59Z) 이전 스냅샷 중 최신을 선택. 없으면 가장 가까운 미래 스냅샷.
    """
    target_dt = datetime.strptime(target_date, "%Y%m%d")
    start = (target_dt - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end = (target_dt + timedelta(days=1)).strftime("%Y-%m-%d")

    url = (
        f"{SPACETRACK_QUERY_URL}/NORAD_CAT_ID/{norad_id}"
        f"/EPOCH/{start}--{end}/orderby/EPOCH%20desc/format/json"
    )
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        rows = resp.json()
    except (requests.RequestException, ValueError) as e:
        print(f"  [TLE] Space-Track NORAD {norad_id} 요청 실패: {e}")
        return None

    if not rows:
        print(f"  [TLE] Space-Track NORAD {norad_id}: {start}~{end} 데이터 없음")
        return None

    # target 자정(익일 00:00) 이전 epoch 우선
    cutoff = target_dt + timedelta(days=1)
    before = [r for r in rows if datetime.fromisoformat(r["EPOCH"].replace("Z", "")) < cutoff]
    chosen = before[0] if before else rows[-1]  # desc 정렬이므로 before[0]이 가장 최근 과거

    return chosen["OBJECT_NAME"], chosen["TLE_LINE1"], chosen["TLE_LINE2"]


def _cache_path(date_str: str, catalog_key: str = "default") -> Path:
    """날짜별 TLE 캐시 파일 경로."""
    TLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if catalog_key == "default" else f"_{catalog_key}"
    return TLE_CACHE_DIR / f"tle_{date_str}{suffix}.json"


def _normalize_reference_date(reference_date: str | None = None) -> str:
    """YYYYMMDD 문자열을 정규화한다. None이면 오늘 UTC 날짜를 사용한다."""
    if reference_date:
        return str(reference_date)
    return datetime.utcnow().strftime("%Y%m%d")


def list_cached_tle_dates(catalog_key: str = "default") -> list[str]:
    """로컬에 저장된 TLE 캐시 날짜 목록을 반환한다."""
    TLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dates = []
    suffix = "" if catalog_key == "default" else f"_{catalog_key}"
    pattern = f"tle_*{suffix}.json"
    for path in TLE_CACHE_DIR.glob(pattern):
        stem = path.stem.replace("tle_", "", 1)
        if suffix and stem.endswith(suffix):
            stem = stem[: -len(suffix)]
        date_str = stem
        if len(date_str) == 8 and date_str.isdigit():
            dates.append(date_str)
    return sorted(dates)


def resolve_tle_cache_date(
    reference_date: str | None = None,
    mode: str = "operational",
    catalog_key: str = "default",
) -> str | None:
    """요청 날짜와 모드에 맞는 TLE 캐시 기준 날짜를 결정한다."""
    requested_date = _normalize_reference_date(reference_date)

    if mode != "backtest":
        return requested_date

    cached_dates = list_cached_tle_dates(catalog_key=catalog_key)
    if requested_date in cached_dates:
        return requested_date

    previous_dates = [d for d in cached_dates if d <= requested_date]
    if previous_dates:
        return previous_dates[-1]

    return None


def load_all_tle(
    force_refresh: bool = False,
    reference_date: str | None = None,
    mode: str = "operational",
    return_info: bool = False,
    satellites: list[dict] | None = None,
    catalog_key: str = "default",
) -> dict | tuple[dict, dict]:
    """모든 위성의 TLE를 수집하고 캐시에 저장한다.

    Returns:
        {norad_id: {"name": str, "line1": str, "line2": str, "meta": dict}, ...}
    """
    requested_date = _normalize_reference_date(reference_date)
    satellite_defs = satellites or SATELLITES
    resolved_date = resolve_tle_cache_date(requested_date, mode=mode, catalog_key=catalog_key)
    info = {
        "mode": mode,
        "requested_date": requested_date,
        "resolved_date": resolved_date,
        "source": "unknown",
        "catalog_key": catalog_key,
    }

    if mode == "backtest":
        # 캐시 우선 (force_refresh 아닐 때)
        if resolved_date is not None and not force_refresh:
            cache_file = _cache_path(resolved_date, catalog_key=catalog_key)
            source_label = "exact" if resolved_date == requested_date else "previous"
            print(
                f"  [TLE] 백테스트 캐시 로드: {cache_file.name} "
                f"(요청 {requested_date}, 사용 {resolved_date}, {source_label})"
            )
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            info["source"] = "historical-cache" if source_label == "exact" else "historical-cache-fallback"
            return (data, info) if return_info else data

        # 캐시 없으면 Space-Track에서 과거 TLE 수집
        print(f"  [TLE] Space-Track에서 {requested_date} 기준 과거 TLE 수집 중... ({len(satellite_defs)}개 위성)")
        result: dict = {}
        try:
            with requests.Session() as session:
                _spacetrack_login(session)
                for sat in satellite_defs:
                    tle = fetch_historical_tle(sat["norad_id"], requested_date, session)
                    if tle is None:
                        print(f"  [TLE] ⚠️ {sat['name']} (NORAD {sat['norad_id']}) 과거 TLE 없음, 스킵")
                        continue
                    name, line1, line2 = tle
                    result[str(sat["norad_id"])] = {
                        "name": name,
                        "line1": line1,
                        "line2": line2,
                        "meta": {
                            "display_name": sat["name"],
                            "type": sat["type"],
                            "swath_km": sat["swath_km"],
                            "resolution_m": sat["resolution_m"],
                            "off_nadir_deg": sat["off_nadir_deg"],
                            "altitude_km": sat["altitude_km"],
                            "priority": sat["priority"],
                        },
                    }
                    print(f"  [TLE] ✅ {sat['name']} ({name.strip()}) 과거 TLE 수집 완료")
        except Exception as e:
            print(f"  [TLE] ❌ Space-Track 조회 실패: {e}")
            info["source"] = "spacetrack-failed"
            return ({}, info) if return_info else {}

        if not result:
            info["source"] = "spacetrack-empty"
            return ({}, info) if return_info else {}

        # 캐시 저장 (요청 날짜 기준)
        cache_file = _cache_path(requested_date, catalog_key=catalog_key)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  [TLE] 캐시 저장: {cache_file.name} ({len(result)}개 위성)")
        info["resolved_date"] = requested_date
        info["source"] = "space-track-historical"
        return (result, info) if return_info else result

    cache_file = _cache_path(requested_date, catalog_key=catalog_key)

    # 캐시가 있고 강제 갱신이 아니면 캐시 반환
    if cache_file.exists() and not force_refresh:
        print(f"  [TLE] 캐시 로드: {cache_file.name}")
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        info["source"] = "cache"
        return (data, info) if return_info else data

    print(f"  [TLE] CelesTrak에서 {len(satellite_defs)}개 위성의 TLE 수집 중... (병렬 처리)")
    result = {}

    with requests.Session() as session:
        # 워커 개수는 위성 개수만큼, 단 최대 10개로 제한
        max_workers = min(10, len(satellite_defs))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 퓨처 객체 매핑
            futures = {
                executor.submit(fetch_tle, sat["norad_id"], session): sat
                for sat in satellite_defs
            }
            
            for future in as_completed(futures):
                sat = futures[future]
                norad_id, tle = future.result()

                if tle is None:
                    print(f"  [TLE] ⚠️ {sat['name']} (NORAD {norad_id}) 수집 실패, 스킵")
                    continue

                name, line1, line2 = tle
                result[str(norad_id)] = {
                    "name": name,
                    "line1": line1,
                    "line2": line2,
                    "meta": {
                        "display_name": sat["name"],
                        "type": sat["type"],
                        "swath_km": sat["swath_km"],
                        "resolution_m": sat["resolution_m"],
                        "off_nadir_deg": sat["off_nadir_deg"],
                        "altitude_km": sat["altitude_km"],
                        "priority": sat["priority"],
                    },
                }
                print(f"  [TLE] ✅ {sat['name']} ({name.strip()}) 수집 완료")

    # 캐시 저장
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  [TLE] 캐시 저장: {cache_file.name} ({len(result)}개 위성)")
    info["source"] = "fetch"

    return (result, info) if return_info else result


# CLI 실행
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CelesTrak TLE 수집기")
    parser.add_argument("--satellite", type=str, help="특정 위성만 수집")
    parser.add_argument("--refresh", action="store_true", help="캐시 무시, 재수집")
    parser.add_argument("--date", type=str, help="캐시 기준 날짜 (YYYYMMDD)")
    parser.add_argument("--mode", choices=["operational", "backtest"], default="operational", help="TLE 사용 모드")
    parser.add_argument("--scenario", default="default", help="위성 카탈로그 시나리오")
    args = parser.parse_args()

    catalog = load_satellite_catalog(args.scenario)

    if args.satellite:
        # 특정 위성만 조회
        sat_info = next(
            (s for s in catalog if s["name"].lower() == args.satellite.lower()),
            None,
        )
        if sat_info:
            _, tle = fetch_tle(sat_info["norad_id"])
            if tle:
                print(f"\n{'='*50}")
                print(f"위성: {sat_info['name']} (NORAD {sat_info['norad_id']})")
                print(f"{'='*50}")
                print(tle[0])
                print(tle[1])
                print(tle[2])
        else:
            print(f"위성 '{args.satellite}'을(를) 찾을 수 없습니다.")
    else:
        data = load_all_tle(
            force_refresh=args.refresh,
            reference_date=args.date,
            mode=args.mode,
            satellites=catalog,
            catalog_key=args.scenario,
        )
        print(f"\n총 {len(data)}개 위성 TLE 수집 완료.")
