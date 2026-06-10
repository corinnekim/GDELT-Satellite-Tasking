"""
Level 1 + Level 2a 통합 파이프라인
1. GDELT 원본 로드
2. compute_conflict_index (clean_gdelt_data 필터 내부 적용)
3. detect_anomalies (타깃 날짜)
4. verify_anomalies_with_llm (top_k=20)
5. SUCCESS/AMBIGUOUS 필터 → risk_cities dict (RED/ORANGE/YELLOW)
6. build_schedule (backtest 기본)
7. print/save
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone, timedelta

import pandas as pd

from pipeline.config import MAIN_PATH, URL_PATH, PREDICTION_HOURS
from pipeline.roi_cities import ROI_CITIES
from pipeline.kalman_filter import compute_conflict_index, detect_anomalies
from pipeline.llm_verification import verify_anomalies_with_llm
from pipeline.city_utils import normalize_city_name
from pipeline.schedule_builder import build_schedule, save_schedule
from pipeline.event_dedup import (
    load_archive, save_archive, find_duplicates, archive_verified, LOOKBACK_DAYS
)
from pipeline.daily_json_writer import write_daily_json


# risk_level(3/2/1) → 위성 스케줄러의 risk_label
RISK_LEVEL_TO_LABEL = {3: "RED", 2: "ORANGE", 1: "YELLOW"}

# 리스크별 원형 이모지
RISK_EMOJI = {"RED": "🔴", "ORANGE": "🟠", "YELLOW": "🟡", "BLUE": "🔵"}

# SpaceEye-T가 있으면 우선 선택 (SIA 자체 위성)
PREFERRED_SATELLITE = "SpaceEye-T"

# SUCCESS/AMBIGUOUS만 스케줄 후보로 간주 (운영 원칙)
SCHEDULABLE_STATUSES = {"SUCCESS", "AMBIGUOUS"}

KST = timezone(timedelta(hours=9))


def _extract_summary(llm_report_json: str | None) -> str:
    """llm_report JSON 문자열에서 Summary 필드를 꺼낸다."""
    if not llm_report_json:
        return ""
    try:
        return json.loads(llm_report_json).get("Summary", "") or ""
    except (ValueError, TypeError):
        return ""


def _extract_top_urls(source_urls, limit: int = 2) -> list[str]:
    if source_urls is None:
        return []
    try:
        return list(source_urls)[:limit]
    except TypeError:
        return []


def build_risk_cities_from_llm(verified: pd.DataFrame) -> tuple[dict, dict]:
    """LLM 검증 결과에서 스케줄 입력을 추출한다.

    Returns:
        cities: {city_name: {"lat": float, "lon": float}}
        risk_cities: {city_name: {
            "risk_label", "innovation_z", "severity_score",
            "llm_summary", "source_urls"
        }}
    """
    eligible = verified[verified["llm_status"].isin(SCHEDULABLE_STATUSES)].copy()
    if eligible.empty:
        return {}, {}

    cities: dict = {}
    risk_cities: dict = {}

    # 같은 대표 이름이 여러 FeatureID로 들어올 수 있으므로 z-score 최상위 행만 채택
    eligible = eligible.sort_values("std_innov", ascending=False)
    for _, row in eligible.iterrows():
        city_name = normalize_city_name(row["city"])
        if city_name in cities:
            continue

        # ROI에 있으면 ROI 좌표 우선, 없으면 Kalman 집계 좌표 사용
        if city_name in ROI_CITIES:
            coord = ROI_CITIES[city_name]
        else:
            lat = row.get("lat")
            lon = row.get("lng")
            if pd.isna(lat) or pd.isna(lon):
                continue
            coord = {"lat": float(lat), "lon": float(lon)}

        cities[city_name] = coord
        risk_cities[city_name] = {
            "risk_label": RISK_LEVEL_TO_LABEL.get(int(row["risk_level"]), "YELLOW"),
            "innovation_z": float(row["std_innov"]),
            "severity_score": float(row["std_innov"]),
            "llm_summary": _extract_summary(row.get("llm_report")),
            "source_urls": _extract_top_urls(row.get("source_urls"), limit=2),
        }

    return cities, risk_cities


def _format_kst(iso_utc: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_utc.replace("Z", "+00:00")).astimezone(KST)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return iso_utc


def _hyperlink(url: str, label: str) -> str:
    """OSC 8 escape로 클릭 가능한 하이퍼링크 생성 (iTerm2, macOS Terminal, VSCode 지원)."""
    if not url:
        return ""
    return f"\x1b]8;;{url}\x1b\\{label}\x1b]8;;\x1b\\"


def _truncate(text: str, width: int) -> str:
    if not text:
        return ""
    # 한글/영문 혼합 대략 처리: 글자 수 기준 자르기
    if len(text) <= width:
        return text
    return text[: max(0, width - 1)] + "…"


def print_city_schedule(schedule: dict) -> None:
    """도시별 단일 테이블 출력 (z-score desc 정렬).

    컬럼: 리스크이모지 + 대응 우선순위 | 도시 | 촬영 시각(KST) | 메시지 | URL 2개
    """
    execution_plan = schedule.get("satellite_execution_plan", [])
    if not execution_plan:
        print("\n  ⚠️ 촬영 가능한 통과 이벤트가 없습니다.\n")
        return

    # satellite_execution_plan의 timeline에서 모든 이벤트를 뽑아 도시별로 그룹
    # (build_satellite_execution_plan이 이미 RED/ORANGE=2, YELLOW=1 정책 적용)
    by_city: dict[str, list[dict]] = {}
    for sat_plan in execution_plan:
        for ev in sat_plan.get("timeline", []):
            by_city.setdefault(ev["city"], []).append(ev)

    # 도시 정렬: z-score desc
    city_order = sorted(
        by_city.keys(),
        key=lambda c: -max(e.get("innovation_z", 0.0) for e in by_city[c]),
    )

    # 도시별로 pass_time asc 정렬 후 플랫 리스트 구성
    # is_first: 그 도시의 첫 이벤트인지 (도시명·요약·URL은 첫 이벤트에서만 표시)
    city_rows: list[tuple[str, bool, dict]] = []
    for city in city_order:
        events = sorted(by_city[city], key=lambda e: e.get("pass_time_utc", ""))
        for i, ev in enumerate(events):
            city_rows.append((city if i == 0 else "", i == 0, ev))

    print("\n" + "═" * 120)
    print("  🛰️  SIA 위성 촬영 스케줄 리포트 (도시별)")
    print(f"  📅 생성 시각: {schedule.get('generated_utc', 'N/A')}  |  🧭 모드: {schedule.get('mode', 'N/A')}")
    print(
        f"  ⏱️ 예측 구간(KST): {_format_kst(schedule.get('prediction_start_utc', ''))} → "
        f"{_format_kst(schedule.get('prediction_end_utc', ''))}"
    )
    print(f"  🏙️  스케줄된 도시: {len(city_order)}개  |  총 촬영 이벤트: {len(city_rows)}건")
    print("═" * 120)

    header = (
        f"  {'도시':<14} | {'대응 우선순위':<15} | {'촬영 시각(KST) · 위성':<32} | "
        f"{'메시지':<50} | 기사 URL"
    )
    print("\n" + header)
    print("  " + "─" * 118)

    for city_label, is_first, ev in city_rows:
        risk = ev.get("risk_label", "N/A")
        emoji = RISK_EMOJI.get(risk, "⚪")
        label = ev.get("action_priority_label", "확인 필요")
        city = city_label
        kst = _format_kst(ev.get("pass_time_utc", ""))
        sat = ev.get("satellite", "")
        reason = _truncate(ev.get("recommendation_reason", ""), 50)
        summary = _truncate(ev.get("llm_summary", ""), 50) if is_first else ""
        urls = (ev.get("source_urls", []) or []) if is_first else []
        url1_link = _hyperlink(urls[0], "URL1") if len(urls) >= 1 else ""
        url2_link = _hyperlink(urls[1], "URL2") if len(urls) >= 2 else ""

        time_sat = f"{kst}  ({sat})"
        # 메인 라인: '핵심 메시지' (recommendation_reason) + URL1
        print(
            f"  {city:<14} | {emoji} {label:<12} | {time_sat:<32} | "
            f"{reason:<50} | {url1_link}"
        )
        # 연속 라인: LLM Summary + URL2 (도시 첫 이벤트에만 표시)
        if summary or url2_link:
            print(
                f"  {'':<14} | {'':<15} | {'':<32} | "
                f"{summary:<50} | {url2_link}"
            )

    print()


def run(target_date: str, hours: int = PREDICTION_HOURS, mode: str = "backtest",
        top_k: int = 20, save: bool = False) -> dict:
    print("\n" + "=" * 60)
    print(f" SIA 통합 파이프라인: {target_date}  (mode={mode})")
    print("=" * 60)

    print("\n[SYSTEM] 데이터 로드...")
    raw_df = pd.read_parquet(MAIN_PATH)
    url_df = pd.read_parquet(URL_PATH)
    raw_df["date"] = raw_df["SQLDATE"].astype(str).str[:8]

    print("\n[TRACK 1] Kalman Filter...")
    city_daily, filtered = compute_conflict_index(raw_df)
    anomalies_df = detect_anomalies(city_daily, target_date)
    if anomalies_df.empty:
        print(f"\n[INFO] {target_date} 이상징후 없음. 종료.")
        return {}

    print(f"\n[TRACK 1] 이상징후 {len(anomalies_df)}건 탐지.")

    print("\n[TRACK 2] LLM 검증...")
    verified = verify_anomalies_with_llm(anomalies_df, filtered, url_df, target_date, top_k=top_k)

    cities, risk_cities = build_risk_cities_from_llm(verified)
    if not risk_cities:
        print("\n[INFO] SUCCESS/AMBIGUOUS 도시가 없습니다. 스케줄 생략.")
        write_daily_json(target_date, verified, filtered, city_daily=city_daily)  # 위성 정보 없이라도 저장
        return {}

    # 중복 이벤트 감지: 최근 LOOKBACK_DAYS 이내 같은 도시 + URL 겹치면 스케줄 제외
    archive = load_archive()
    duplicates_found: list[dict] = []
    for city_name in list(risk_cities.keys()):
        urls = risk_cities[city_name].get("source_urls", []) or []
        matches = find_duplicates(city_name, urls, target_date, archive)
        if not matches:
            continue
        duplicates_found.append({"city": city_name, "matches": matches})
        # 사람이 읽을 로그만 남기고 스케줄링 제외
        del risk_cities[city_name]
        cities.pop(city_name, None)

    if duplicates_found:
        print(f"\n[DEDUP] 최근 {LOOKBACK_DAYS}일 내 동일 이벤트 후보 {len(duplicates_found)}건 발견, 스케줄 제외, 사람 확인 필요:")
        for d in duplicates_found:
            for m in d["matches"]:
                shared = ", ".join(u[:60] + "…" for u in m["overlap_urls"][:2])
                print(f"  ⚠️  {d['city']:<18} | 이전({m['prior_date']}, {m['prior_status']})과 URL 공유: {shared}")

    if not risk_cities:
        print("\n[INFO] 중복 제거 후 남은 스케줄 대상이 없습니다.")
        write_daily_json(target_date, verified, filtered, city_daily=city_daily)
        return {}

    print(f"\n[ADAPTER] 스케줄 대상 {len(risk_cities)}개 도시:")
    for name, info in risk_cities.items():
        print(f"  - {name:20s} | {info['risk_label']:6s} | z={info['innovation_z']:.2f}")

    # 운영 현실: GDELT v1.0은 전날 데이터를 다음날 11 UTC경 발표 → +35h가 실제 인지 시점.
    # 그 이전 통과는 사건을 모르므로 스케줄링 불가 → prediction_start = target_date + 35h.
    # 표시용 "사건 발생일 12:00 UTC 기준 7일" 윈도우는 daily_json_writer에서 별도 처리.
    prediction_start_utc = (datetime.strptime(target_date, "%Y%m%d").replace(tzinfo=timezone.utc)
                            + timedelta(hours=24 + 11))

    from pipeline.config import BASELINE_NO_PERF as _NO_PERF
    if _NO_PERF:
        print(f"\n[TRACK 3] BASELINE 모드, 위성 스케줄 단계 skip")
        write_daily_json(target_date, verified, filtered, city_daily=city_daily)
        return {'status': 'baseline_no_satellite'}

    print(f"\n[TRACK 3] Level 2a 위성 스케줄 빌드 (mode={mode})...")
    schedule = build_schedule(
        cities=cities,
        risk_cities=risk_cities,
        hours=hours,
        tle_mode=mode,
        tle_reference_date=target_date,
        prediction_start_utc=prediction_start_utc,
    )

    if "error" in schedule:
        print(f"\n[ERROR] 스케줄 빌드 실패: {schedule['error']}")
        write_daily_json(target_date, verified, filtered, city_daily=city_daily)
        return schedule

    print_city_schedule(schedule)
    if save:
        save_schedule(schedule, filename=f"schedule_{target_date}.json")

    # 대시보드용 daily JSON: verified + schedule 통합 (위성 통과 시각 포함)
    dash_path, rec_path = write_daily_json(target_date, verified, filtered, schedule=schedule, city_daily=city_daily)
    print(f"[OUTPUT] dashboard: {dash_path}")
    print(f"[OUTPUT] records:   {rec_path}")

    # 이번에 스케줄된 도시들을 아카이브에 기록 (다음 날짜 실행 시 중복 감지 재료)
    for city_name, info in risk_cities.items():
        archive_verified(
            city=city_name,
            target_date=target_date,
            status=info["risk_label"],  # RED/ORANGE/YELLOW (LLM status 대신 스케줄 라벨)
            source_urls=info.get("source_urls", []) or [],
            llm_report=info.get("llm_summary", ""),
            archive=archive,
        )
    save_archive(archive)

    return schedule


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIA Level 1 + 2a 통합 파이프라인")
    parser.add_argument("--date", type=str, required=True, help="타깃 날짜 YYYYMMDD")
    parser.add_argument("--hours", type=int, default=PREDICTION_HOURS, help="예측 범위(시간)")
    parser.add_argument("--mode", choices=["operational", "backtest"], default="backtest",
                        help="TLE 모드 (기본: backtest)")
    parser.add_argument("--top-k", type=int, default=20, help="LLM 검증 상위 도시 수")
    parser.add_argument("--save", action="store_true", help="스케줄 JSON 저장")
    args = parser.parse_args()

    run(args.date, hours=args.hours, mode=args.mode, top_k=args.top_k, save=args.save)