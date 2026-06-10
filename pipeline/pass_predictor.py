"""
Level 2a, 궤도 통과 예측기
Skyfield SGP4 전파 엔진으로 ROI 상공 위성 통과(Overpass) 시각을 계산한다.
Phase 1: 직하(Nadir) Swath 기준 촬영 가능 판정
Phase 2: Off-Nadir 공개 위성은 확장 범위 반영
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from skyfield.api import load, EarthSatellite, wgs84
from geopy.distance import geodesic

from pipeline.config import (
    MIN_ELEVATION_DEG, PREDICTION_HOURS,
    MAX_OPERATIONAL_OFF_NADIR_DEG,
)
from pipeline.roi_cities import ROI_CITIES
from pipeline.tle_fetcher import load_all_tle

EARTH_RADIUS_KM = 6371.0


def compute_off_nadir_deg(elevation_deg: float, altitude_km: float) -> float:
    """관측자 앙각 → 위성 천저각(off-nadir) 변환.

    sin(off_nadir) = (R / (R + h)) × cos(elevation)
    - elevation 90° → off-nadir 0° (위성 바로 위)
    - elevation 57.4° → off-nadir ~30° (실무 한계, h=500km 기준)
    - elevation 20° → off-nadir ~62°
    """
    elev_clamped = max(min(elevation_deg, 90.0), 0.0)
    sin_theta = (EARTH_RADIUS_KM / (EARTH_RADIUS_KM + altitude_km)) * math.cos(math.radians(elev_clamped))
    sin_theta = max(min(sin_theta, 1.0), 0.0)
    return math.degrees(math.asin(sin_theta))


def _build_satellite(tle_entry: dict, ts) -> EarthSatellite:
    """TLE 데이터로 Skyfield EarthSatellite 객체를 생성한다."""
    return EarthSatellite(
        tle_entry["line1"],
        tle_entry["line2"],
        tle_entry["name"],
        ts,
    )


def _predict_single_pass(
    satellite: EarthSatellite,
    sat_name: str,
    norad_id_str: str,
    meta: dict,
    city_name: str,
    coord: dict,
    t0,
    t1
) -> list[dict]:
    """단일 위성과 단일 도시 간의 통과 이벤트를 타깃으로 예측합니다."""
    passes_local = []
    observer = wgs84.latlon(coord["lat"], coord["lon"])

    try:
        t_events, events = satellite.find_events(
            observer, t0, t1, altitude_degrees=MIN_ELEVATION_DEG
        )
    except Exception:
        return passes_local

    swath_km = meta["swath_km"]
    off_nadir_deg = meta["off_nadir_deg"]
    altitude_km = meta["altitude_km"]
    priority = meta["priority"]

    i = 0
    while i < len(events):
        if events[i] == 1:
            t_culm = t_events[i]
            culm_dt = t_culm.utc_datetime()

            diff = satellite - observer
            topocentric = diff.at(t_culm)
            alt_deg, _, _ = topocentric.altaz()
            max_elev = alt_deg.degrees

            geocentric = satellite.at(t_culm)
            subpoint = wgs84.subpoint(geocentric)
            sub_lat = subpoint.latitude.degrees
            sub_lon = subpoint.longitude.degrees

            distance_km = geodesic(
                (sub_lat, sub_lon),
                (coord["lat"], coord["lon"])
            ).km

            nadir_range = swath_km / 2.0
            within_nadir = distance_km <= nadir_range

            # 모든 위성 ±30° off-nadir 기준 통일 (운영 표준)
            off_nadir_limit = MAX_OPERATIONAL_OFF_NADIR_DEG

            # 이 통과에서 실제로 필요한 off-nadir (앙각으로부터 계산)
            required_off_nadir = compute_off_nadir_deg(max_elev, altitude_km)

            # 운영 가능 = required_off_nadir ≤ 30°
            within_offnadir = required_off_nadir <= off_nadir_limit

            # 참조용, 30° 기준 swath 확장 거리
            extended_range_km = altitude_km * math.tan(math.radians(off_nadir_limit))
            within_swath = within_nadir or within_offnadir

            passes_local.append({
                "satellite": sat_name,
                "norad_id": int(norad_id_str),
                "sensor_type": meta["type"],
                "resolution_m": meta["resolution_m"],
                "priority": priority,
                "city": city_name,
                "lat": coord["lat"],
                "lon": coord["lon"],
                "pass_time_utc": culm_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "max_elevation_deg": round(max_elev, 1),
                "off_nadir_deg": round(required_off_nadir, 1),  # 대시보드용
                "required_off_nadir_abs_deg": round(required_off_nadir, 2),
                "off_nadir_limit_deg": round(off_nadir_limit, 1),
                "subpoint_lat": round(sub_lat, 4),
                "subpoint_lon": round(sub_lon, 4),
                "distance_km": round(distance_km, 1),
                "nadir_range_km": round(nadir_range, 1),
                "extended_range_km": round(extended_range_km, 1),
                "within_swath": within_swath,
                "swath_phase": "nadir" if within_nadir else (
                    "off-nadir" if within_offnadir else "out-of-range"
                ),
            })
        i += 1

    return passes_local


def predict_passes(
    tle_data: dict,
    cities: dict | None = None,
    hours: int | None = None,
    base_time_utc: datetime | None = None,
) -> list[dict]:
    """모든 위성 × 모든 도시의 통과 이벤트를 병렬 예측한다.

    Args:
        tle_data: load_all_tle() 반환값
        cities: {city_name: {"lat": float, "lon": float}} (None이면 config ROI_CITIES)
        hours: 예측 범위(시간) (None이면 config PREDICTION_HOURS)
        base_time_utc: 예측 시작 기준 시각(UTC). None이면 현재 시각

    Returns:
        통과 이벤트 리스트 (dict)
    """
    if cities is None:
        cities = ROI_CITIES
    if hours is None:
        hours = PREDICTION_HOURS
    if base_time_utc is None:
        base_time_utc = datetime.now(timezone.utc)
    elif base_time_utc.tzinfo is None:
        base_time_utc = base_time_utc.replace(tzinfo=timezone.utc)
    else:
        base_time_utc = base_time_utc.astimezone(timezone.utc)

    ts = load.timescale()
    t0 = ts.from_datetime(base_time_utc)
    t1 = ts.from_datetime(base_time_utc + timedelta(hours=hours))

    # 실행할 작업을 담을 리스트 구성
    tasks = []
    for norad_id_str, tle_entry in tle_data.items():
        meta = tle_entry["meta"]
        sat_name = meta["display_name"]

        try:
            satellite = _build_satellite(tle_entry, ts)
        except Exception as e:
            print(f"  [PASS] ⚠️ {sat_name} 위성 객체 생성 실패: {e}")
            continue

        for city_name, coord in cities.items():
            tasks.append((
                satellite, sat_name, norad_id_str, meta, city_name, coord, t0, t1
            ))

    passes = []
    
    # Thread 풀을 통한 병렬 처리
    print(f"  [PASS] 총 {len(tasks)}개의 물리 연산 태스크 병렬 전파 중...")
    max_workers = min(20, max(1, len(tasks)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_predict_single_pass, *task): task
            for task in tasks
        }
        
        for future in as_completed(futures):
            res = future.result()
            if res:
                passes.extend(res)

    # 시간순 정렬
    passes.sort(key=lambda x: x["pass_time_utc"])
    return passes


def filter_shootable(passes: list[dict]) -> list[dict]:
    """운영 가능 통과만 필터링.
    조건: within_swath AND required_off_nadir ≤ MAX_OPERATIONAL_OFF_NADIR_DEG (±30°).
    nadir 통과(swath 직하)는 off-nadir 무관하게 통과."""
    out = []
    for p in passes:
        if not p.get("within_swath"):
            continue
        # nadir 통과는 off-nadir 0 근처라 자동 통과
        if p.get("swath_phase") == "nadir":
            out.append(p)
            continue
        if p.get("required_off_nadir_abs_deg", 0) <= MAX_OPERATIONAL_OFF_NADIR_DEG:
            out.append(p)
    return out


# CLI 실행
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="위성 통과 예측기")
    parser.add_argument("--city", type=str, help="특정 도시만 예측")
    parser.add_argument("--hours", type=int, default=PREDICTION_HOURS, help="예측 범위(시간)")
    parser.add_argument("--start", type=str, help="예측 시작 시각 UTC (예: 2026-04-01T00:00:00Z)")
    args = parser.parse_args()

    tle_data = load_all_tle()
    cities = None
    if args.city:
        if args.city in ROI_CITIES:
            cities = {args.city: ROI_CITIES[args.city]}
        else:
            print(f"도시 '{args.city}'을(를) 찾을 수 없습니다.")
            exit(1)

    print(f"\n  ── 위성 통과 예측 ({args.hours}시간) ──")
    base_time_utc = None
    if args.start:
        base_time_utc = datetime.fromisoformat(args.start.replace("Z", "+00:00"))
    all_passes = predict_passes(
        tle_data,
        cities=cities,
        hours=args.hours,
        base_time_utc=base_time_utc,
    )
    shootable = filter_shootable(all_passes)

    print(f"\n  전체 통과: {len(all_passes)}건 | 촬영 가능: {len(shootable)}건\n")

    for p in shootable[:20]:
        print(
            f"  🛰️ {p['satellite']:12s} | {p['city']:15s} | "
            f"{p['pass_time_utc']} | 앙각 {p['max_elevation_deg']:5.1f}° | "
            f"거리 {p['distance_km']:6.1f}km | {p['swath_phase']}"
        )

    if len(shootable) > 20:
        print(f"  ... 외 {len(shootable)-20}건")
