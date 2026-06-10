"""대시보드용 일별 JSON + 기록용 일별 JSON 출력.

대시보드 (dashboard/daily_<date>.json):
  - SUCCESS / AMBIGUOUS 후보만 (위성 촬영 대상)
  - 각 도시별 가장 빠른 위성 통과 _TOP_PASSES_PER_CITY 개

기록 (records/daily_<date>.json):
  - LLM 검증은 받았으나 reject된 후보 (DROPPED, DATE_MISMATCH, NO_MENTION 등)
  - review_queue (high-z + DROPPED/DATE_MISMATCH)
  - unverified_count (top-K 밖이라 LLM 미검증인 Kalman anomalies 수)
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pipeline.config import (
    PROJECT_ROOT, ARTIFACT_ROOT, COEFS, RISK_THRESHOLDS, RISK_LABELS_LIST,
    REVIEW_Z_THRESHOLD, REVIEW_LLM_STATUSES, LLM_TOP_N, KALMAN_Q, KALMAN_R,
)
from pipeline.llm_verification import PROMPT_VERSION, MODEL_ID

DASHBOARD_DIR = ARTIFACT_ROOT / "dashboard"
RECORDS_DIR   = ARTIFACT_ROOT / "records"
TP_POS = {'SUCCESS', 'AMBIGUOUS'}
# 전체 보고 기간(예: 32일) 기준 1회 산출. 도시당 단일 tier 고정 (보고 일관성).
# A = active_days ≥ 5 AND total_mentions ≥ 100 (단발 폭증 노이즈 자연 제외)
TIER_ACTIVE_MIN  = 5
TIER_MENT_MIN    = 100

_SAT_PASS_KEYS = (
    'satellite', 'norad_id', 'sensor_type', 'resolution_m', 'pass_time_utc',
    'max_elevation_deg', 'off_nadir_deg', 'off_nadir_limit_deg',
    'within_swath', 'cloud_cover_pct', 'cloud_status',
    'daylight', 'shootable', 'action_priority_label', 'urgency_label',
    'capture_condition_label', 'recommendation_reason',
)
_TOP_PASSES_PER_CITY = 3   # 7일 윈도우 내 도시당 가장 빠른 N개만 노출
_TOP_URLS_PER_CITY   = 5   # 도시당 노출할 기사 URL 개수 (LLM이 본 3~4개 모두 표시)


def _risk_label_for_z(z: float) -> str:
    """z → risk_label. RISK_THRESHOLDS는 내림차순 [5, 2, 0.5]."""
    for thr, lab in zip(RISK_THRESHOLDS, RISK_LABELS_LIST):
        if z >= thr:
            return lab
    return '정상'


def _build_tier_lookup(filtered_df: pd.DataFrame) -> dict:
    """전체 보고 기간 단위로 도시별 tier 1회 확정 → city → 'A'/'C' dict.
    같은 도시는 보고 안에서 같은 라벨 유지 (보고 일관성)."""
    if filtered_df.empty:
        return {}
    agg = (filtered_df.groupby('standard_name')
                      .agg(active_days=('date', 'nunique'),
                           total_mentions=('NumMentions', 'sum')))
    return {city: ('A' if (r.active_days >= TIER_ACTIVE_MIN and r.total_mentions >= TIER_MENT_MIN) else 'C')
            for city, r in agg.iterrows()}


def _passes_by_city(schedule: dict | None, primary_cutoff_iso: str | None = None) -> tuple[dict, dict]:
    """schedule['recommendations']를 city → (within_7d_passes[:3], delayed_marker) 로 그룹.
    primary_cutoff_iso: 사건일 +7일 시각. 이 이전 = 1차 윈도우, 이후 = 지연 마커 후보.
    sensor_type 'EO' 로 매핑."""
    if not schedule:
        return {}, {}
    grouped: dict[str, list] = {}
    for rec in schedule.get('recommendations', []):
        city = rec.get('city')
        if not city:
            continue
        p = {k: rec.get(k) for k in _SAT_PASS_KEYS}
        if p.get('sensor_type') == 'optical':
            p['sensor_type'] = 'EO'
        grouped.setdefault(city, []).append(p)

    passes_out: dict[str, list] = {}
    delayed_out: dict[str, dict] = {}
    for city, passes in grouped.items():
        passes.sort(key=lambda p: p.get('pass_time_utc') or '')
        if primary_cutoff_iso:
            within = [p for p in passes if (p.get('pass_time_utc') or '') < primary_cutoff_iso]
            after  = [p for p in passes if (p.get('pass_time_utc') or '') >= primary_cutoff_iso]
        else:
            within, after = passes, []

        if within:
            passes_out[city] = within[:_TOP_PASSES_PER_CITY]
        else:
            passes_out[city] = []
            if after:
                first = after[0]
                delayed_out[city] = {
                    'first_pass_utc': first.get('pass_time_utc'),
                    'satellite':      first.get('satellite'),
                    'message':        '사건 발생 7일 이내 촬영 가능 위성 없음, 1차 가능 통과는 7일 이후',
                }
    return passes_out, delayed_out


def _spaceeye_option_by_city(schedule: dict | None) -> dict:
    """SIA 자산 SpaceEye-T 가장 빠른 shootable 통과를 도시별 별도 옵션으로 추출.
    shootable=True이지만 시간순 top-3에는 못 든 케이스도 부각."""
    if not schedule:
        return {}
    out: dict[str, dict] = {}
    for rec in schedule.get('recommendations', []):
        if rec.get('satellite') != 'SpaceEye-T':
            continue
        city = rec.get('city')
        if not city:
            continue
        p = {k: rec.get(k) for k in _SAT_PASS_KEYS}
        if p.get('sensor_type') == 'optical':
            p['sensor_type'] = 'EO'
        prev = out.get(city)
        if prev is None or (p.get('pass_time_utc') or '') < (prev.get('pass_time_utc') or ''):
            out[city] = p
    return out


def _planetscope_option_by_city(schedule: dict | None, primary_cutoff_iso: str | None = None) -> dict:
    """PlanetScope 가장 빠른 shootable 통과를 도시별 별도 옵션으로 추출.
    primary_cutoff_iso 이후 통과는 None 처리 (7일 윈도우 내 통과만 표시)."""
    if not schedule:
        return {}
    out: dict[str, dict] = {}
    for rec in schedule.get('recommendations', []):
        sat = rec.get('satellite', '')
        if not sat.startswith('PlanetScope-'):
            continue
        if primary_cutoff_iso and (rec.get('pass_time_utc') or '') >= primary_cutoff_iso:
            continue
        city = rec.get('city')
        if not city:
            continue
        p = {k: rec.get(k) for k in _SAT_PASS_KEYS}
        if p.get('sensor_type') == 'optical':
            p['sensor_type'] = 'EO'
        prev = out.get(city)
        if prev is None or (p.get('pass_time_utc') or '') < (prev.get('pass_time_utc') or ''):
            out[city] = p
    return out


def _llm_report_obj(raw):
    """원본 llm_report (dict 또는 JSON string) → status/message 보존된 dict."""
    if not raw:
        return None
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return None


def write_daily_json(
    date: str,
    verified_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    schedule: dict | None = None,
    Q: float | None = None,
    R: float | None = None,
    city_daily: pd.DataFrame | None = None,    # 도시별 Kalman 결과 (z 시계열용)
) -> tuple[Path, Path]:
    """run_pipeline의 verified DataFrame을 대시보드 + 기록 JSON 두 개로 직렬화.

    Returns: (dashboard_path, records_path)
    """
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    RECORDS_DIR.mkdir(parents=True, exist_ok=True)

    tier_lookup = _build_tier_lookup(filtered_df)
    # 표시용 1차 윈도우 cutoff = 사건 발생일 12:00 UTC + 168h (사용자 기준 7일).
    # 실제 스케줄링은 다음날 11:00 UTC부터지만, 사용자 표기는 사건일 기준이라 분리.
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz
    _event_ref = _dt.strptime(date, "%Y%m%d").replace(tzinfo=_tz.utc) + _td(hours=12)
    _primary_cutoff_iso = (_event_ref + _td(hours=168)).strftime("%Y-%m-%dT%H:%M:%SZ")
    sat_passes, sat_delayed = _passes_by_city(schedule, primary_cutoff_iso=_primary_cutoff_iso)
    spaceeye_options    = _spaceeye_option_by_city(schedule)
    planetscope_options = _planetscope_option_by_city(schedule, primary_cutoff_iso=_primary_cutoff_iso)

    # 시계열 룩업: city → list[{date, z, sources, goldstein}] (날짜 오름차순)
    history_lookup = {}
    if city_daily is not None and not city_daily.empty and 'std_innov' in city_daily.columns:
        cd = city_daily.sort_values('date').copy()
        cd['ds'] = cd['date'].astype(str).str.replace('-', '')
        for city, sub in cd.groupby('city'):
            rows = []
            for r in sub.itertuples(index=False):
                rows.append({
                    'date': r.ds,
                    'z': round(float(r.std_innov), 3),
                    'sources': round(float(getattr(r, 'sources', 0) or 0), 2),
                    'goldstein': round(float(getattr(r, 'avg_goldstein', 0) or 0), 2),
                })
            history_lookup[city] = rows

    ANOM_Z = 2.0   # 연속 anomaly 일수 기준

    if not filtered_df.empty:
        cd_stats = (filtered_df[filtered_df['date'] == date]
                    .groupby('standard_name')
                    .agg(events=('GLOBALEVENTID', 'count'),
                         conflict_index=('weighted_index', 'sum'),
                         sources_total=('NumSources', 'sum'),
                         mentions_total=('NumMentions', 'sum'))
                    .to_dict('index'))
    else:
        cd_stats = {}

    def tier_of(city: str) -> str:
        return tier_lookup.get(city, 'C')

    eligible = verified_df[verified_df['llm_status'] != 'UNVERIFIED']
    unverified_count = int((verified_df['llm_status'] == 'UNVERIFIED').sum())

    targets, non_targets = [], []
    for _, row in eligible.sort_values('std_innov', ascending=False).iterrows():
        city = row['city']                          # standard_name (예: 'Tehran|IR')
        city_display = city.split('|', 1)[0]        # 사용자 표시용 (suffix 제거)
        z = float(row['std_innov'])
        llm_status = row.get('llm_status')
        llm_report = _llm_report_obj(row.get('llm_report'))
        message = (llm_report or {}).get('message') or (llm_report or {}).get('Summary')
        urls = list(row.get('source_urls') or [])[:_TOP_URLS_PER_CITY]
        needs_review = (z >= REVIEW_Z_THRESHOLD) and (llm_status in REVIEW_LLM_STATUSES)

        ce = filtered_df[(filtered_df['standard_name'] == city) & (filtered_df['date'] == date)]
        if not ce.empty and 'NumMentions' in ce.columns:
            nw = ce.groupby('ActionGeo_FullName')['NumMentions'].sum().sort_values(ascending=False)
            display_name = nw.index[0] if not nw.empty else city_display
        else:
            display_name = city_display

        stats = cd_stats.get(city, {})
        # 직전 30일 history (target_date 포함), z, sources, goldstein 한 번에
        history_30d = []
        delta_z = None
        consecutive_anom = 0
        volume_7d_ratio = None  # 최근 7일 sources 합 / 직전 7일 sources 합
        if city in history_lookup:
            full = history_lookup[city]
            up_to = [r for r in full if r['date'] <= date]
            history_30d = up_to[-30:] if len(up_to) > 30 else up_to[:]

            # Δz: 어제 z 대비 변화량
            today_row = next((r for r in up_to if r['date'] == date), None)
            if today_row and len(up_to) >= 2:
                prev_row = up_to[-2] if up_to[-1]['date'] == date else up_to[-1]
                delta_z = round(today_row['z'] - prev_row['z'], 3)

            # 연속 anomaly 일수: target_date부터 거꾸로 z >= ANOM_Z 인 연속 일수
            for r in reversed(up_to):
                if r['z'] >= ANOM_Z:
                    consecutive_anom += 1
                else:
                    break

            # 최근 7일 NumSources 합 / 직전 7일 합 (target_date 기준)
            if len(up_to) >= 14:
                recent7 = sum(r.get('sources', 0) for r in up_to[-7:])
                prior7  = sum(r.get('sources', 0) for r in up_to[-14:-7])
                if prior7 > 0:
                    volume_7d_ratio = round(recent7 / prior7, 2)

        # 사용자 친화적 anomaly 메시지, 트리거 만족시에만 추가
        anomaly_signals = []
        if volume_7d_ratio is not None and volume_7d_ratio >= 2.0:
            anomaly_signals.append({
                'type': 'volume_7d_boost',
                'value': volume_7d_ratio,
                'message': f"최근 7일 보도량 {volume_7d_ratio}배 급증",
            })
        if consecutive_anom >= 2:
            anomaly_signals.append({
                'type': 'anomaly_streak',
                'value': consecutive_anom,
                'message': f"{consecutive_anom}일 연속 이상 신호 탐지",
            })
        if delta_z is not None and abs(delta_z) >= 2.0:
            direction = "증가" if delta_z > 0 else "감소"
            anomaly_signals.append({
                'type': 'z_change',
                'value': delta_z,
                'message': f"어제 대비 이상 신호 {abs(delta_z):.1f} {direction}",
            })
        common = {
            'city': city_display,
            'display_name': display_name,
            'lat': float(row['lat']) if pd.notna(row.get('lat')) else None,
            'lon': float(row['lng']) if pd.notna(row.get('lng')) else None,
            'tier': tier_of(city),
            'std_innov': round(z, 3),
            'risk_label': _risk_label_for_z(z),
            'events_count': int(stats.get('events', 0)),
            'conflict_index': round(float(stats.get('conflict_index', 0.0)), 3),
            'sources_total': int(stats.get('sources_total', 0)),
            'mentions_total': int(stats.get('mentions_total', 0)),
            'llm_status': llm_status,
            'llm_message': message,
            'urls_sent': urls,
            'history_30d': history_30d,            # [{date, z, sources, goldstein}] 최대 30일
            'delta_z': delta_z,                    # 전일 대비 z 변화량 (float or null)
            'consecutive_anomaly_days': consecutive_anom,   # z>=2 연속 일수
            'volume_7d_ratio': volume_7d_ratio,    # 최근 7일 NumSources 합 / 직전 7일 합
            'anomaly_signals': anomaly_signals,    # 사용자 친화적 메시지 (트리거 만족시)
        }

        if llm_status in TP_POS:
            # 대시보드 표출용, satellite_passes 포함
            targets.append({
                **common,
                'satellite_passes':    sat_passes.get(city, []),
                'delayed_capture':     sat_delayed.get(city),       # 7일 이내 없을 때만 (null 가능)
                'spaceeye_option':     spaceeye_options.get(city),  # SpaceEye-T 가장 빠른 통과 (전체 윈도우)
                'planetscope_option':  planetscope_options.get(city),  # PlanetScope 가장 빠른 통과 (7일 이내만)
            })
        else:
            # 기록용, needs_review 플래그 포함
            non_targets.append({**common, 'needs_review': needs_review})

    base_cfg = {
        'model': MODEL_ID,
        'prompt_version': PROMPT_VERSION,
        'Q': Q if Q is not None else KALMAN_Q,
        'R': R if R is not None else KALMAN_R,
        'coefs': COEFS,
        'risk_thresholds': RISK_THRESHOLDS,
        'top_k': LLM_TOP_N,
    }
    now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    # 대시보드 JSON: SUCCESS/AMBIGUOUS만, review/reject 관련 config 제외
    dash_payload = {
        'date': date,
        'generated_at': now,
        'config': {**base_cfg, 'top_passes_per_city': _TOP_PASSES_PER_CITY},
        'summary': {
            'satellite_targets': len(targets),
            'total_passes':      sum(len(t['satellite_passes']) for t in targets),
        },
        'targets': targets,
    }
    dash_path = DASHBOARD_DIR / f"daily_{date}.json"
    dash_path.write_text(json.dumps(dash_payload, ensure_ascii=False, indent=2))

    # 기록 JSON: 그 외 + 검토 큐 + 미검증
    review_queue = [c for c in non_targets if c['needs_review']]
    rec_payload = {
        'date': date,
        'generated_at': now,
        'config': {**base_cfg,
                   'review_z_threshold': REVIEW_Z_THRESHOLD,
                   'review_llm_statuses': sorted(REVIEW_LLM_STATUSES)},
        'summary': {
            'rejected_by_llm':   len(non_targets),
            'needs_review':      len(review_queue),
            'unverified_kalman': unverified_count,
        },
        'rejected_candidates': non_targets,   # DROPPED, DATE_MISMATCH, NO_MENTION 등
        'review_queue':        review_queue,  # 높은 z + DROPPED/DATE_MISMATCH
    }
    rec_path = RECORDS_DIR / f"daily_{date}.json"
    rec_path.write_text(json.dumps(rec_payload, ensure_ascii=False, indent=2))

    return dash_path, rec_path


# CLI 진입점
if __name__ == '__main__':
    import argparse
    from pipeline.config import MAIN_PATH, URL_PATH
    from pipeline.kalman_filter import compute_conflict_index, detect_anomalies
    from pipeline.llm_verification import verify_anomalies_with_llm

    ap = argparse.ArgumentParser(description='특정 날짜의 daily JSON을 처음부터 생성 (위성 스케줄 미포함)')
    ap.add_argument('--date', required=True, help='YYYYMMDD')
    ap.add_argument('--top-k', type=int, default=LLM_TOP_N)
    args = ap.parse_args()

    raw = pd.read_parquet(MAIN_PATH); url_df = pd.read_parquet(URL_PATH)
    raw['date'] = raw['SQLDATE'].astype(str).str[:8]
    city_daily, filtered = compute_conflict_index(raw, is_train=False)
    anomalies = detect_anomalies(city_daily, args.date)
    verified = verify_anomalies_with_llm(anomalies, filtered, url_df, args.date, top_k=args.top_k)
    dash_path, rec_path = write_daily_json(args.date, verified, filtered)
    print(f"  dashboard: {dash_path}")
    print(f"  records:   {rec_path}")
