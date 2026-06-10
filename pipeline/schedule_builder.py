"""
Level 2a, 촬영 스케줄 빌더
Level 1 갈등 탐지 결과 + 위성 통과 예측 + 기상 조건을 통합하여
최종 촬영 스케줄 데이터셋(JSON)을 생성한다.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pipeline.config import SCHEDULE_OUTPUT_DIR, PREDICTION_HOURS, OPERATIONAL_SATELLITE_SCENARIO
from pipeline.roi_cities import ROI_CITIES
from pipeline.tle_fetcher import load_all_tle
from pipeline.satellite_catalog import load_satellite_catalog
from pipeline.pass_predictor import predict_passes, filter_shootable
from pipeline.weather_checker import check_weather

MIN_CAPTURE_GAP_MINUTES = 10
KST = timezone(timedelta(hours=9))
PRIORITY_DISPLAY_ORDER = [
    "즉시 촬영",
    "우선 촬영",
    "정밀 관측",
    "주의 관측",
    "일반 모니터링",
]


def normalize_sensor_type(sensor: str | None) -> str:
    """센서 타입 표기를 소문자 기준 공통 포맷으로 정규화한다."""
    return str(sensor or "optical").strip().lower()


def get_urgency_index(event: dict) -> float:
    """확장성을 염두에 두고 이벤트 객체에서 위험/긴급성에 대한 스코어를 추출한다."""
    return float(event.get("severity_score", event.get("innovation_z", 0.0)))


def classify_priority_band(event: dict) -> int:
    """정책에 따라 우선순위 등급(Band 0 ~ 5)을 결정합니다."""
    # risk_label·risk_level 다형성 지원
    risk = event.get("risk_label", event.get("risk_level", "N/A"))
    z_score = get_urgency_index(event)
    sensor = normalize_sensor_type(event.get("sensor_type", "optical"))
    daylight = event.get("daylight", True)
    cloud = event.get("cloud_cover_pct", 50)
    elev = event.get("max_elevation_deg", 10)

    # 광학 적합성(eo_favorable)
    eo_favorable = daylight and (cloud <= 40) and (elev >= 30)
    
    # 광학 촬영 불가(eo_impossible)
    eo_impossible = not daylight or (cloud > 50)

    band = 0

    if risk == "RED":
        band = 4
        # SAR 강제 승격 정책 (높은 z-score + RED + 기존 EO 불가능 상황 시 SAR 위성 승격)
        if z_score >= 4.0 and eo_impossible and sensor == "sar":
            band = 5
        # RED지만 Z-score가 낮고 EO 촬영 조건이 극도로 불량한 타겟(SAR가 아닐 경우 강등)
        elif z_score < 4.0 and eo_impossible and sensor == "optical":
            band = 2
            
    elif risk == "ORANGE":
        band = 3
        # ORANGE 등급에서 광학 조건이 좋으면 상향
        if eo_favorable and sensor == "optical":
            band = 4
            
    elif risk == "YELLOW":
        band = 2
        # YELLOW 역전 허용 정책 (조건이 좋으면 나쁜 레드 추월 가능)
        if z_score >= 2.0 and eo_favorable and sensor == "optical":
            band = 4
            
    elif risk == "BLUE":
        band = 1

    return band


def compute_quality_score(event: dict) -> float:
    """동일 등급 내에서 촬영 품질(구름, 앙각, off-nadir 여유)에 따른 세부 순서.
    geometry_score = 1 - (required_off_nadir / off_nadir_limit) → off-nadir 작을수록 높음."""
    sensor = normalize_sensor_type(event.get("sensor_type", "optical"))
    elev = max(min(event.get("max_elevation_deg", 10), 90), 1)
    off_nadir_limit = max(float(event.get("off_nadir_limit_deg", 30) or 30), 1)
    off_nadir_abs = max(float(event.get("required_off_nadir_abs_deg", 0) or 0), 0)
    geometry_score = max(0.0, 1.0 - off_nadir_abs / off_nadir_limit)

    if sensor == "sar":
        # SAR은 구름 영향을 거의 받지 않으므로 관측 기하 중심으로 평가한다.
        return round(0.5 * (elev / 90) + 0.5 * geometry_score, 4)

    cloud = max(min(event.get("cloud_cover_pct", 50), 100), 0)
    # EO: 구름 + (앙각 0.4 + off-nadir 여유 0.6) 가중
    return round((1 - cloud / 100) * (0.4 * (elev / 90) + 0.6 * geometry_score), 4)


def compute_policy_preference(event: dict) -> float:
    """운영 정책 기준 가점.

    목표:
    - SpaceEye-T 최우선 광학 자산 (4.0)
    - PlanetScope 군집 가점 (3.2), 감시 공백 메우기
    - 일반 EO (2.0)
    - SAR (광학 불가 시 2.5, 그 외 1.0)
    """
    sensor = normalize_sensor_type(event.get("sensor_type", "optical"))
    satellite = event.get("satellite", "")
    daylight = bool(event.get("daylight", True))
    cloud = max(min(event.get("cloud_cover_pct", 50), 100), 0)
    elev = max(min(event.get("max_elevation_deg", 10), 90), 1)

    eo_favorable = daylight and cloud <= 40 and elev >= 30
    eo_impossible = (not daylight) or cloud > 50

    if satellite == "SpaceEye-T" and sensor == "optical" and eo_favorable:
        return 4.0
    if satellite.startswith("PlanetScope-") and sensor == "optical" and eo_favorable:
        return 3.2
    if sensor == "optical" and eo_favorable:
        return 2.0
    if sensor == "sar" and eo_impossible:
        return 2.5
    if sensor == "sar":
        return 1.0
    return 0.0


def _recommendation_sort_key(event: dict) -> tuple:
    """정책 등급과 품질을 기준으로 추천 후보를 정렬한다."""
    return (
        -event.get("priority_band", 0),
        -event.get("policy_preference", 0.0),
        -get_urgency_index(event),
        -event.get("quality_score", 0.0),
        event.get("pass_time_utc", ""),
        event.get("priority", 99),
    )


def get_action_priority_label(event: dict) -> str:
    """내부 Band를 사용자 친화적 우선순위 라벨로 변환한다."""
    band = event.get("priority_band", 0)
    if band >= 5:
        return "즉시 촬영"
    if band >= 4:
        return "우선 촬영"
    if band >= 3:
        return "정밀 관측"
    if band >= 2:
        return "주의 관측"
    return "일반 모니터링"


def get_urgency_label(event: dict) -> str:
    """이상 징후 강도를 일반 사용자용 라벨로 변환한다."""
    urgency = get_urgency_index(event)
    if urgency >= 50:
        return "매우 높음"
    if urgency >= 10:
        return "높음"
    if urgency >= 4:
        return "상당함"
    if urgency >= 2:
        return "주의"
    return "낮음"


def get_capture_condition_label(event: dict) -> str:
    """센서와 기상 조건을 종합해 촬영 여건 라벨을 만든다."""
    sensor = normalize_sensor_type(event.get("sensor_type", "optical"))
    elev = max(min(event.get("max_elevation_deg", 10), 90), 1)
    cloud = max(min(event.get("cloud_cover_pct", 50), 100), 0)
    daylight = bool(event.get("daylight", True))

    if sensor == "sar":
        if elev >= 75:
            return "SAR 관측 양호"
        if elev >= 55:
            return "SAR 관측 보통"
        return "SAR 관측 제한"

    if daylight and cloud <= 20 and elev >= 50:
        return "EO 촬영 매우 좋음"
    if daylight and cloud <= 40 and elev >= 30:
        return "EO 촬영 좋음"
    if daylight and cloud <= 50:
        return "EO 촬영 보통"
    return "EO 촬영 제한"


def build_recommendation_reason(event: dict) -> str:
    """대시보드에 바로 노출 가능한 짧은 핵심 메시지를 생성한다."""
    sensor = normalize_sensor_type(event.get("sensor_type", "optical"))
    risk = event.get("risk_label", event.get("risk_level", "N/A"))
    urgency = get_urgency_label(event)
    condition = get_capture_condition_label(event)
    cloud = event.get("cloud_cover_pct", -1)
    daylight = bool(event.get("daylight", True))
    band = event.get("priority_band", 0)

    if sensor == "sar" and event.get("priority_band", 0) >= 5:
        if not daylight:
            return f"{risk} 지역이며 야간 통과라 SAR로 우선 확인"
        if cloud > 50:
            return f"{risk} 지역이며 구름이 많아 SAR 촬영 우선"
        return f"{risk} 지역이며 긴급도가 {urgency} 수준이라 SAR 우선"

    if sensor == "sar":
        if band >= 4:
            return f"{risk} 지역의 선제 확인용 SAR 후보"
        if band >= 2:
            return f"{risk} 지역의 보조 감시용 SAR 후보"
        return f"{risk} 지역의 일반 모니터링용 SAR 후보"

    if sensor == "optical":
        if condition in ("EO 촬영 매우 좋음", "EO 촬영 좋음"):
            return f"{risk} 지역이며 광학 촬영 여건이 {condition}"
        if condition == "EO 촬영 보통":
            return f"{risk} 지역이며 광학 촬영 가능"
        return f"{risk} 지역이나 확보된 촬영 기회부터 활용"

    return f"{risk} 지역 후속 판독 필요"


def enrich_display_fields(event: dict) -> dict:
    """대시보드/콘솔용 사용자 친화적 표시 필드를 추가한다."""
    enriched = event.copy()
    enriched["sensor_type"] = normalize_sensor_type(enriched.get("sensor_type"))
    enriched["action_priority_label"] = get_action_priority_label(enriched)
    enriched["urgency_label"] = get_urgency_label(enriched)
    enriched["capture_condition_label"] = get_capture_condition_label(enriched)
    enriched["recommendation_reason"] = build_recommendation_reason(enriched)
    return enriched


def _group_recommendations_by_priority(recs: list[dict]) -> list[tuple[str, list[dict]]]:
    """대응 우선순위 라벨 순서대로 추천 목록을 그룹핑한다."""
    grouped: dict[str, list[dict]] = {}
    for rec in recs:
        grouped.setdefault(rec.get("action_priority_label", "일반 모니터링"), []).append(rec)

    ordered_groups = []
    for label in PRIORITY_DISPLAY_ORDER:
        items = grouped.get(label, [])
        if items:
            ordered_groups.append((label, items))
    return ordered_groups


def build_city_best_recommendations(events: list[dict]) -> list[dict]:
    """도시별 대표 촬영 후보 1건씩 추려 대시보드 친화적 구조로 반환한다."""
    grouped: dict[str, list[dict]] = {}
    for event in events:
        grouped.setdefault(event["city"], []).append(event)

    city_best = []
    for city, candidates in grouped.items():
        ranked = sorted(candidates, key=_recommendation_sort_key)
        top = ranked[0].copy()
        top["candidate_count_for_city"] = len(candidates)
        top["alternative_pass_count"] = max(0, len(candidates) - 1)
        city_best.append(top)

    city_best.sort(key=_recommendation_sort_key)
    return city_best


def build_sensor_condition_summary(events: list[dict]) -> dict:
    """센서별 촬영 가능/제약 요약을 만든다."""
    summary = {
        "optical_total": 0,
        "optical_shootable": 0,
        "optical_blocked_night": 0,
        "optical_blocked_cloud": 0,
        "optical_blocked_unknown": 0,
        "sar_total": 0,
        "sar_shootable": 0,
    }

    for event in events:
        sensor = normalize_sensor_type(event.get("sensor_type"))
        if sensor == "sar":
            summary["sar_total"] += 1
            if event.get("shootable"):
                summary["sar_shootable"] += 1
            continue

        summary["optical_total"] += 1
        if event.get("shootable"):
            summary["optical_shootable"] += 1
            continue

        if not event.get("daylight", True):
            summary["optical_blocked_night"] += 1
        elif event.get("cloud_cover_pct", -1) < 0:
            summary["optical_blocked_unknown"] += 1
        else:
            summary["optical_blocked_cloud"] += 1

    return summary


def _parse_pass_time(pass_time_utc: str) -> datetime:
    """UTC 문자열을 timezone-aware datetime으로 변환한다."""
    return datetime.fromisoformat(pass_time_utc.replace("Z", "+00:00"))


def _format_display_time_kst(time_str: str) -> str:
    """UTC ISO 시각 문자열을 KST 표기 문자열로 변환한다."""
    try:
        dt_utc = _parse_pass_time(time_str)
        return dt_utc.astimezone(KST).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return time_str


def build_satellite_execution_plan(
    events: list[dict],
    min_gap_minutes: int = MIN_CAPTURE_GAP_MINUTES,
) -> dict:
    """단순 greedy 규칙으로 위성별 실제 실행 계획을 생성한다.

    규칙:
    - 전체 후보를 정책 우선순위 순으로 본다.
    - 같은 도시는 1회만 채택한다.
    - 같은 위성에서 기존 채택 이벤트와 min_gap 이내로 겹치면 스킵한다.
    """
    selected_events = []
    skipped_conflicts = []
    city_assignments: dict[str, list[dict]] = {}
    satellite_times: dict[str, list[datetime]] = {}
    gap_seconds = max(min_gap_minutes, 0) * 60

    for event in sorted(events, key=_recommendation_sort_key):
        city = event["city"]
        satellite = event["satellite"]
        sensor = normalize_sensor_type(event.get("sensor_type"))
        pass_dt = _parse_pass_time(event["pass_time_utc"])
        existing_city_events = city_assignments.setdefault(city, [])
        risk = event.get("risk_label", event.get("risk_level", "N/A"))
        city_limit = 2 if risk in {"RED", "ORANGE"} else 1

        if len(existing_city_events) >= city_limit:
            continue

        # RED/ORANGE 도시의 두 번째 슬롯은 가능하면 다른 센서로 채워
        # "primary + backup" 구성을 유도한다.
        if existing_city_events:
            existing_sensors = {
                normalize_sensor_type(item.get("sensor_type"))
                for item in existing_city_events
            }
            if sensor in existing_sensors:
                continue

        existing_times = satellite_times.setdefault(satellite, [])
        has_conflict = any(abs((pass_dt - prev).total_seconds()) < gap_seconds for prev in existing_times)
        if has_conflict:
            skipped_conflicts.append({
                "city": city,
                "satellite": satellite,
                "pass_time_utc": event["pass_time_utc"],
                "priority_band": event.get("priority_band", 0),
                "urgency_index": round(get_urgency_index(event), 2),
                "reason": f"same-satellite gap<{min_gap_minutes}m",
            })
            continue

        selected = event.copy()
        selected_events.append(selected)
        existing_city_events.append(selected)
        existing_times.append(pass_dt)

    grouped: dict[str, list[dict]] = {}
    for event in selected_events:
        grouped.setdefault(event["satellite"], []).append(event)

    execution_plan = []
    for satellite, items in sorted(grouped.items()):
        items.sort(key=lambda x: x["pass_time_utc"])
        timeline = []
        for order, item in enumerate(items, 1):
            scheduled = item.copy()
            scheduled["execution_order"] = order
            scheduled["min_capture_gap_minutes"] = min_gap_minutes
            timeline.append(scheduled)
        execution_plan.append({
            "satellite": satellite,
            "scheduled_count": len(timeline),
            "timeline": timeline,
        })

    execution_plan.sort(
        key=lambda x: (
            -len(x["timeline"]),
            x["timeline"][0]["pass_time_utc"] if x["timeline"] else "",
            x["satellite"],
        )
    )

    return {
        "min_capture_gap_minutes": min_gap_minutes,
        "scheduled_events": selected_events,
        "scheduled_cities": len(city_assignments),
        "satellites_used": len(execution_plan),
        "execution_plan": execution_plan,
        "skipped_conflicts": skipped_conflicts,
    }


def build_schedule(
    cities: dict | None = None,
    risk_cities: dict | None = None,
    hours: int = PREDICTION_HOURS,
    force_tle_refresh: bool = False,
    tle_mode: str = "operational",
    tle_reference_date: str | None = None,
    prediction_start_utc: datetime | None = None,
    satellite_scenario: str = OPERATIONAL_SATELLITE_SCENARIO,
) -> dict:
    """촬영 스케줄을 생성한다.

    Args:
        cities: ROI 도시 좌표. None이면 config 기본값
        risk_cities: Level 1 결과에서 도시별 리스크 등급.
                     예: {"Isfahan": "RED", "Tehran": "YELLOW"}
        hours: 예측 범위(시간)
        force_tle_refresh: TLE 캐시 강제 갱신 여부
        tle_mode: operational | backtest
        tle_reference_date: TLE 캐시 기준 날짜(YYYYMMDD)
        prediction_start_utc: 궤도 예측 시작 시각(UTC)
        satellite_scenario: 사용할 위성 카탈로그 시나리오

    Returns:
        스케줄 JSON 데이터 (dict)
    """
    if cities is None:
        cities = ROI_CITIES
    if risk_cities is None:
        risk_cities = {}

    now_utc = datetime.now(timezone.utc)
    satellites = load_satellite_catalog(satellite_scenario)
    if prediction_start_utc is None:
        prediction_start_utc = now_utc
    elif prediction_start_utc.tzinfo is None:
        prediction_start_utc = prediction_start_utc.replace(tzinfo=timezone.utc)
    else:
        prediction_start_utc = prediction_start_utc.astimezone(timezone.utc)

    if tle_reference_date is None:
        tle_reference_date = prediction_start_utc.strftime("%Y%m%d")
    prediction_end_utc = prediction_start_utc + timedelta(hours=hours)

    # Step 1: TLE 수집
    print("\n  ── Level 2a: 촬영 스케줄 빌더 가동 ──\n")
    tle_data, tle_info = load_all_tle(
        force_refresh=force_tle_refresh,
        reference_date=tle_reference_date,
        mode=tle_mode,
        return_info=True,
        satellites=satellites,
        catalog_key=satellite_scenario,
    )

    if not tle_data:
        print("  [SCHEDULE] ❌ TLE 데이터 없음. 종료.")
        return {
            "error": "TLE 데이터 수집 실패",
            "mode": tle_mode,
            "tle_requested_date": tle_info.get("requested_date"),
            "tle_reference_date": tle_info.get("resolved_date"),
            "tle_source": tle_info.get("source"),
        }

    # Step 2: 궤도 통과 예측
    print(f"\n  [PASS] {len(cities)}개 도시 × {len(tle_data)}개 위성 통과 예측 중...")
    all_passes = predict_passes(
        tle_data,
        cities=cities,
        hours=hours,
        base_time_utc=prediction_start_utc,
    )
    shootable_passes = filter_shootable(all_passes)
    stale_passes_removed = 0
    if tle_mode == "operational":
        operational_cutoff = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        before_cutoff = len(shootable_passes)
        shootable_passes = [
            event for event in shootable_passes
            if event.get("pass_time_utc", "") >= operational_cutoff
        ]
        stale_passes_removed = before_cutoff - len(shootable_passes)
    print(f"  [PASS] 전체 통과: {len(all_passes)}건 | Swath 내: {len(shootable_passes)}건")
    if stale_passes_removed:
        print(f"  [PASS] 운영 기준 시각 이전 통과 {stale_passes_removed}건 제외")

    # Step 3: 기상 판별
    print(f"\n  [WEATHER] {len(shootable_passes)}건에 대해 기상 조건 확인 중...")
    weather_checked = []
    for p in shootable_passes:
        checked = check_weather(p)
        
        # risk_cities가 dict의 dict(상세정보)일 경우 확장성 처리
        city_info = risk_cities.get(p["city"], "N/A")
        if isinstance(city_info, dict):
            checked["risk_label"] = city_info.get("risk_label", "N/A")
            checked["innovation_z"] = city_info.get("innovation_z", 0.0)
            checked["severity_score"] = city_info.get("severity_score", 0.0)
            checked["llm_summary"] = city_info.get("llm_summary", "")
            checked["source_urls"] = city_info.get("source_urls", [])
        else:
            checked["risk_label"] = city_info
            
        checked["priority_band"] = classify_priority_band(checked)
        checked["quality_score"] = compute_quality_score(checked)
        checked["policy_preference"] = compute_policy_preference(checked)
        checked = enrich_display_fields(checked)
        weather_checked.append(checked)

    # Step 4: 촬영 가능 필터링 + 정렬
    final_shootable = [e for e in weather_checked if e["shootable"]]
    
    # [정책 등급(Band) desc -> 품질 점수 desc -> 통과 시간 asc -> 위성 우선순위 asc]
    final_shootable.sort(key=_recommendation_sort_key)
    city_best_recommendations = build_city_best_recommendations(final_shootable)
    satellite_execution = build_satellite_execution_plan(final_shootable)
    sensor_condition_summary = build_sensor_condition_summary(weather_checked)

    # Step 5: 결과 조립
    schedule = {
        "generated_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": tle_mode,
        "satellite_scenario": satellite_scenario,
        "tle_requested_date": tle_info.get("requested_date"),
        "tle_reference_date": tle_info.get("resolved_date"),
        "tle_source": tle_info.get("source"),
        "prediction_start_utc": prediction_start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "prediction_end_utc": prediction_end_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "prediction_hours": hours,
        "satellites_tracked": len(tle_data),
        "cities_monitored": len(cities),
        "total_passes": len(all_passes),
        "swath_passes": len(shootable_passes),
        "stale_passes_removed": stale_passes_removed,
        "shootable_passes": len(final_shootable),
        "cities_with_shootable_passes": len(city_best_recommendations),
        "recommendations": final_shootable,
        "city_best_recommendations": city_best_recommendations,
        "sensor_condition_summary": sensor_condition_summary,
        "satellite_execution_plan": satellite_execution["execution_plan"],
        "execution_min_gap_minutes": satellite_execution["min_capture_gap_minutes"],
        "scheduled_events": len(satellite_execution["scheduled_events"]),
        "scheduled_cities": satellite_execution["scheduled_cities"],
        "satellites_used": satellite_execution["satellites_used"],
        "skipped_execution_conflicts": satellite_execution["skipped_conflicts"],
    }

    return schedule


def save_schedule(schedule: dict, filename: str | None = None) -> Path:
    """스케줄을 JSON 파일로 저장한다."""
    SCHEDULE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if filename is None:
        date_str = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"schedule_{date_str}.json"

    filepath = SCHEDULE_OUTPUT_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(schedule, f, indent=2, ensure_ascii=False)

    print(f"\n  [SCHEDULE] 📄 저장: {filepath}")
    return filepath


def print_schedule(schedule: dict) -> None:
    """스케줄 리포트를 터미널에 출력한다."""
    recs = schedule.get("city_best_recommendations") or schedule.get("recommendations", [])
    execution_plan = schedule.get("satellite_execution_plan", [])
    sensor_summary = schedule.get("sensor_condition_summary", {})

    print("\n" + "═" * 85)
    print("  🛰️  SIA 위성 촬영 스케줄 리포트")
    print(f"  📅 생성 시각: {schedule.get('generated_utc', 'N/A')}")
    print(f"  🧭 실행 모드: {schedule.get('mode', 'operational')}")
    print(
        f"  ⏱️ 예측 구간(KST): {_format_display_time_kst(schedule.get('prediction_start_utc', 'N/A'))} "
        f"→ {_format_display_time_kst(schedule.get('prediction_end_utc', 'N/A'))}"
    )
    print(
        f"  🛰️ TLE 기준일: {schedule.get('tle_reference_date', 'N/A')} "
        f"(source: {schedule.get('tle_source', 'unknown')})"
    )
    print(f"  🔭 추적 위성: {schedule.get('satellites_tracked', 0)}개")
    print(f"  🏙️  모니터링 도시: {schedule.get('cities_monitored', 0)}개")
    print("═" * 85)

    print(f"\n  📊 통과 요약: 전체 {schedule.get('total_passes', 0)}건 "
          f"→ Swath 내 {schedule.get('swath_passes', 0)}건 "
          f"→ 촬영 가능 {schedule.get('shootable_passes', 0)}건 "
          f"→ 도시별 대표 {schedule.get('cities_with_shootable_passes', len(recs))}건 "
          f"→ 실행 계획 {schedule.get('scheduled_cities', 0)}건\n")
    if sensor_summary:
        print(
            "  ℹ️ 센서별 판정: "
            f"EO 후보 {sensor_summary.get('optical_total', 0)}건 중 촬영 가능 {sensor_summary.get('optical_shootable', 0)}건, "
            f"SAR 후보 {sensor_summary.get('sar_total', 0)}건 중 촬영 가능 {sensor_summary.get('sar_shootable', 0)}건"
        )
        if sensor_summary.get("optical_total", 0) and sensor_summary.get("optical_shootable", 0) == 0:
            print(
                "  ℹ️ EO 미선정 사유: "
                f"야간 {sensor_summary.get('optical_blocked_night', 0)}건, "
                f"구름 과다 {sensor_summary.get('optical_blocked_cloud', 0)}건, "
                f"기상 미상 {sensor_summary.get('optical_blocked_unknown', 0)}건"
            )
        print()
    if schedule.get("stale_passes_removed", 0):
        print(f"  ℹ️ 운영 기준 시각 이전 통과 {schedule['stale_passes_removed']}건은 자동 제외했습니다.\n")

    if not execution_plan:
        if recs:
            print("  ⚠️ 추천 후보는 있으나 실행 계획으로 확정된 일정이 없습니다.\n")
        else:
            print("  ⚠️ 촬영 가능한 통과 이벤트가 없습니다.\n")
        return

    gap = schedule.get("execution_min_gap_minutes", MIN_CAPTURE_GAP_MINUTES)
    print(f"  ── 위성별 실행 계획 (최소 간격 {gap}분) ──")
    for satellite_plan in execution_plan:
        print(f"  [{satellite_plan['satellite']}] {satellite_plan['scheduled_count']}건")
        print(f"  {'순번':>2s} | {'도시':15s} | {'대응 우선순위':12s} | "
              f"{'촬영 시각(KST)':16s} | {'핵심 메시지'}")
        print("  " + "─" * 93)
        for item in satellite_plan["timeline"][:12]:
            print(
                f"  {item['execution_order']:>2d} | {item['city']:15s} | "
                f"{item.get('action_priority_label', '확인 필요'):12s} | "
                f"{_format_display_time_kst(item['pass_time_utc']):16s} | "
                f"{item.get('recommendation_reason', '')}"
            )
        if len(satellite_plan["timeline"]) > 12:
            print(f"  ... 외 {len(satellite_plan['timeline']) - 12}건")
        print()

    print()



# CLI 실행

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="촬영 스케줄 빌더")
    parser.add_argument("--hours", type=int, default=PREDICTION_HOURS, help="예측 범위(시간)")
    parser.add_argument("--refresh", action="store_true", help="TLE 캐시 강제 갱신")
    parser.add_argument("--mode", choices=["operational", "backtest"], default="operational", help="실행 모드")
    parser.add_argument("--tle-date", type=str, help="TLE 기준 날짜 (YYYYMMDD)")
    parser.add_argument("--start", type=str, help="예측 시작 시각 UTC (예: 2026-04-01T00:00:00Z)")
    parser.add_argument("--save", action="store_true", help="JSON 파일로 저장")
    args = parser.parse_args()

    prediction_start_utc = None
    if args.start:
        prediction_start_utc = datetime.fromisoformat(args.start.replace("Z", "+00:00"))

    schedule = build_schedule(
        hours=args.hours,
        force_tle_refresh=args.refresh,
        tle_mode=args.mode,
        tle_reference_date=args.tle_date,
        prediction_start_utc=prediction_start_utc,
    )

    if "error" not in schedule:
        print_schedule(schedule)
        if args.save:
            save_schedule(schedule)
