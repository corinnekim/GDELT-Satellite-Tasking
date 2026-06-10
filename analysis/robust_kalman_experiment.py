"""
Student-t Robust Kalman + per-city R 실험 (V0~V3 비교).

가설: §6.5.1의 per-city R 실패 원인은 (a) 가우시안 가정과 (b) 소도시 R≈0 폭주이지,
      per-city 자체가 잘못된 게 아니다.
검증: 4가지 변형 비교 — V0 baseline, V1/V2 Student-t with global R, V3 Student-t + per-city R + floor.

평가: top-K Recall (A tier / C tier / All) on Train + Test, LLM 단계 별도 실행.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from numba import jit

from pipeline.config import (
    KALMAN_Q, KALMAN_R, KALMAN_P0_RATIO, MIN_HISTORY, MAIN_PATH, PROJECT_ROOT, COEFS
)
from pipeline.preprocess import clean_gdelt_data, apply_standard_scaling
from pipeline.kalman_filter import kalman_innovation as kalman_gaussian
from pipeline.text_norm import _norm

GT_DIR = PROJECT_ROOT / "data" / "ground_truth"
OUT_DIR = PROJECT_ROOT / "logs" / "robust_kalman_experiment"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── Student-t Robust Kalman ────────────────────────────────────────────────
@jit(nopython=True)
def kalman_innovation_robust(signal: np.ndarray, Q: float, R: float, dof: float = 4.0) -> tuple:
    """Variational Student-t observation Kalman. std_innov은 Gaussian z-score (탐지용 임계치 호환)."""
    n = len(signal)
    x_est = np.zeros(n)
    P_est = np.zeros(n)
    innovation = np.zeros(n)
    std_innov = np.zeros(n)

    x_est[0] = signal[0]
    P_est[0] = R * KALMAN_P0_RATIO

    for t in range(1, n):
        x_pred = x_est[t - 1]
        P_pred = P_est[t - 1] + Q
        innov = signal[t] - x_pred
        S = max(P_pred + R, 1e-10)
        innovation[t] = innov
        std_innov[t] = innov / np.sqrt(S)

        m2 = (innov * innov) / S
        w = (dof + 1.0) / (dof + m2)
        w_safe = max(w, 1e-10)
        R_eff = R / w_safe
        S_eff = max(P_pred + R_eff, 1e-10)
        K = P_pred / S_eff
        x_est[t] = x_pred + K * innov
        P_est[t] = (1.0 - K) * P_pred

    return x_est, innovation, std_innov


# ─── 도시별 robust scale 추정 (V3용) ─────────────────────────────────────────
def estimate_city_R(signal: np.ndarray, R_floor: float = 10.0, R_ceiling: float = 5000.0,
                    init_window: int = 30, percentile: float = 50.0) -> float:
    """
    도시별 R: 첫 N일의 robust scale (MAD)로 변동성 추정 + floor.

    MAD = median(|x - median(x)|), σ ≈ 1.4826 × MAD (정규분포 가정 하 robust σ 추정).
    R = σ²이 통상적이므로 R = (1.4826 × MAD)².

    Floor로 소도시 R≈0 폭주 차단 (§6.5.1 실패 방지).
    Ceiling으로 단일 spike 도시의 과도한 R 차단.
    """
    init = signal[:init_window]
    if len(init) < 5:
        return R_floor
    med = np.median(init)
    mad = np.median(np.abs(init - med))
    sigma = 1.4826 * mad
    R = float(sigma * sigma)
    return float(np.clip(R, R_floor, R_ceiling))


# ─── 도시별 Kalman 적용 ──────────────────────────────────────────────────────
def apply_kalman_per_city(agg_df: pd.DataFrame, variant: str, Q: float, R: float, dof: float = 4.0,
                          R_floor: float = 5.0, R_ceiling: float = 5000.0):
    global_min = agg_df['date'].min()
    global_max = agg_df['date'].max()
    all_dates = pd.date_range(start=global_min, end=global_max, freq='D')

    out_rows = []
    R_used_log = []
    for city, grp in agg_df.groupby('city'):
        grp = grp.sort_values('date').set_index('date').reindex(all_dates).reset_index().rename(columns={'index': 'date'})
        for col in ['lat', 'lng']:
            if col in grp.columns:
                grp[col] = grp[col].ffill().bfill()
        if 'conflict_index' in grp.columns:
            grp['conflict_index'] = grp['conflict_index'].fillna(0)
        grp['city'] = city

        signal = np.ascontiguousarray(grp['conflict_index'].values, dtype=np.float64)
        if len(signal) < MIN_HISTORY:
            continue

        if variant == 'gaussian':
            x_est, innov, std_innov = kalman_gaussian(signal, Q, R)
            R_used = R
        elif variant == 'studentt':
            x_est, innov, std_innov = kalman_innovation_robust(signal, Q, R, dof)
            R_used = R
        elif variant == 'studentt_perR':
            R_city = estimate_city_R(signal, R_floor=R_floor, R_ceiling=R_ceiling)
            x_est, innov, std_innov = kalman_innovation_robust(signal, Q, R_city, dof)
            R_used = R_city
        else:
            raise ValueError(variant)

        R_used_log.append({'city': city, 'R': R_used, 'sig_max': float(signal.max()),
                           'sig_mean': float(signal.mean())})
        grp['kalman_est'] = x_est
        grp['innovation'] = innov
        grp['std_innov'] = std_innov
        grp['date'] = grp['date'].dt.strftime('%Y%m%d')
        out_rows.append(grp[['date', 'city', 'lat', 'lng', 'conflict_index',
                             'kalman_est', 'innovation', 'std_innov']])

    if not out_rows:
        return pd.DataFrame(), pd.DataFrame(R_used_log)
    return pd.concat(out_rows, ignore_index=True), pd.DataFrame(R_used_log)


# ─── 평가 ─────────────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    p = np.pi / 180.0
    a = 0.5 - np.cos((lat2 - lat1) * p) / 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 12742.0 * np.arcsin(np.sqrt(a))


def load_gt(gt_dir: Path) -> pd.DataFrame:
    files = sorted(gt_dir.rglob("*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.rename(columns={'SQLDATE': 'date', 'ActionGeo_FullName': 'city',
                            'Lat': 'lat', 'Long': 'lon', 'Latitude': 'lat', 'Longitude': 'lon'})
    df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d', errors='coerce').dt.strftime('%Y%m%d')
    df = df.dropna(subset=['date', 'lat', 'lon'])
    return df[['date', 'city', 'lat', 'lon']]


def compute_a_tier(filtered_df: pd.DataFrame) -> set:
    by_city = (filtered_df.groupby('standard_name')
               .agg(active_days=('date', 'nunique'),
                    total_mentions=('NumMentions', 'sum'))
               .reset_index())
    a = by_city[(by_city['active_days'] >= 10) & (by_city['total_mentions'] >= 100)]
    a_norms = set()
    for sn in a['standard_name']:
        base = str(sn).split('|')[0]
        a_norms.add(_norm(base))
    return a_norms


def is_a_tier(gt_city_norm: str, a_norms: set) -> bool:
    if not gt_city_norm:
        return False
    return any(gt_city_norm == an or (an and gt_city_norm in an) or (an and an in gt_city_norm) for an in a_norms)


def evaluate_recall(kalman_out: pd.DataFrame, gt_df: pd.DataFrame, dates: list,
                    top_k: int, a_norms: set, radius_km: float = 50.0) -> dict:
    a_num, a_den = 0, 0
    c_num, c_den = 0, 0
    all_num, all_den = 0, 0
    daily = []

    for d in dates:
        day = kalman_out[kalman_out['date'] == d].copy()
        day = day[day['std_innov'] > 0].sort_values('std_innov', ascending=False).head(top_k)
        gt_today = gt_df[gt_df['date'] == d]

        n_today_match = 0
        for _, gt in gt_today.iterrows():
            gt_norm = _norm(str(gt['city'])) if pd.notna(gt['city']) else ""
            matched = False
            for _, anom in day.iterrows():
                city_base = str(anom['city']).split('|')[0]
                anom_norm = _norm(city_base) if city_base else ""
                if gt_norm and anom_norm and (gt_norm == anom_norm or gt_norm in anom_norm or anom_norm in gt_norm):
                    matched = True; break
                if pd.notna(anom.get('lat')) and pd.notna(anom.get('lng')):
                    if haversine(float(anom['lat']), float(anom['lng']), float(gt['lat']), float(gt['lon'])) <= radius_km:
                        matched = True; break
            in_a = is_a_tier(gt_norm, a_norms)
            all_den += 1
            if in_a: a_den += 1
            else:    c_den += 1
            if matched:
                all_num += 1
                if in_a: a_num += 1
                else:    c_num += 1
                n_today_match += 1
        daily.append({'date': d, 'top_k': len(day), 'gt': len(gt_today), 'matched': n_today_match})

    return {
        'A_recall': a_num / a_den if a_den else 0.0,
        'A_n': f"{a_num}/{a_den}",
        'C_recall': c_num / c_den if c_den else 0.0,
        'C_n': f"{c_num}/{c_den}",
        'all_recall': all_num / all_den if all_den else 0.0,
        'all_n': f"{all_num}/{all_den}",
        'daily': daily,
    }


# ─── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    print("[LOAD] GDELT main parquet…")
    raw = pd.read_parquet(MAIN_PATH)
    print(f"  rows: {len(raw):,}")

    print("[PREPROCESS] clean_gdelt_data + apply_standard_scaling…")
    filtered = clean_gdelt_data(raw)
    if 'date' not in filtered.columns:
        filtered['date'] = filtered['SQLDATE'].astype(str).str[:8]
    filtered = apply_standard_scaling(filtered, is_train=False)

    key_to_col = {'w_log_mentions': 'Log_Mentions', 'w_log_sources': 'Log_Sources',
                  'w_avgtone': 'AvgTone', 'w_avgtone_sq': 'AvgTone_Sq', 'w_goldstein': 'GoldsteinScale'}
    filtered['weighted_index'] = 0.0
    for key, w in COEFS.items():
        filtered['weighted_index'] += w * filtered[key_to_col[key]]
    agg = (filtered.groupby(['date', 'standard_name'])
           .agg(conflict_index=('weighted_index', 'sum'),
                lat=('ActionGeo_Lat', 'median'),
                lng=('ActionGeo_Long', 'median'))
           .reset_index().rename(columns={'standard_name': 'city'}))
    agg['date'] = pd.to_datetime(agg['date'])
    print(f"  city-day rows: {len(agg):,}, unique cities: {agg['city'].nunique():,}")

    a_norms = compute_a_tier(filtered)
    print(f"  A tier 정규화 set: {len(a_norms)}")

    gt = load_gt(GT_DIR)
    train_dates = sorted(gt[gt['date'].between('20260228', '20260321')]['date'].unique())
    test_dates = sorted(gt[gt['date'].between('20260322', '20260331')]['date'].unique())

    # GT 도시 분류 검증
    gt_a_count = sum(1 for c in gt['city'] if is_a_tier(_norm(str(c)), a_norms))
    print(f"  GT obs total {len(gt)}: A={gt_a_count}, C={len(gt) - gt_a_count}")
    print(f"  Dates: train={len(train_dates)}, test={len(test_dates)}")

    variants = [
        ('V0_Gaussian',          'gaussian',       4.0,  None,  None),
        ('V1_Studentt_dof4',     'studentt',       4.0,  None,  None),
        ('V2_Studentt_dof2.5',   'studentt',       2.5,  None,  None),
        ('V3_perR_floor5_dof4',  'studentt_perR',  4.0,  5.0,   5000.0),
        ('V3b_perR_floor10_dof4','studentt_perR',  4.0,  10.0,  5000.0),
        ('V3c_perR_floor1_dof4', 'studentt_perR',  4.0,  1.0,   5000.0),
    ]

    results = {}
    for label, variant, dof, R_floor, R_ceil in variants:
        print(f"\n[RUN] {label} (Q={KALMAN_Q}, R={KALMAN_R}, dof={dof}, R_floor={R_floor})")
        out, R_log = apply_kalman_per_city(agg, variant, KALMAN_Q, KALMAN_R, dof,
                                           R_floor=R_floor or 0.0, R_ceiling=R_ceil or 1e9)
        if variant == 'studentt_perR':
            print(f"  Per-city R: median {R_log['R'].median():.2f}, p5 {R_log['R'].quantile(0.05):.2f}, "
                  f"p95 {R_log['R'].quantile(0.95):.2f}, max {R_log['R'].max():.0f}")

        train_m = evaluate_recall(out, gt, train_dates, 20, a_norms)
        test_m  = evaluate_recall(out, gt, test_dates,  20, a_norms)
        results[label] = {'train': train_m, 'test': test_m}
        print(f"  Train: A={train_m['A_recall']:.3f} ({train_m['A_n']})  "
              f"C={train_m['C_recall']:.3f} ({train_m['C_n']})  "
              f"All={train_m['all_recall']:.3f} ({train_m['all_n']})")
        print(f"  Test:  A={test_m['A_recall']:.3f} ({test_m['A_n']})  "
              f"C={test_m['C_recall']:.3f} ({test_m['C_n']})  "
              f"All={test_m['all_recall']:.3f} ({test_m['all_n']})")

    print("\n" + "=" * 80)
    print("[SUMMARY] top-K=20 Recall")
    print("=" * 80)
    print(f"{'Variant':<24} {'A_train':<10} {'C_train':<10} {'A_test':<10} {'C_test':<10} {'All_test':<10}")
    print("-" * 80)
    for label, *_ in variants:
        r = results[label]
        print(f"{label:<24} {r['train']['A_recall']:<10.3f} {r['train']['C_recall']:<10.3f} "
              f"{r['test']['A_recall']:<10.3f} {r['test']['C_recall']:<10.3f} {r['test']['all_recall']:<10.3f}")

    out_path = OUT_DIR / "results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=float))
    print(f"\n[OUT] {out_path}")


if __name__ == '__main__':
    main()
