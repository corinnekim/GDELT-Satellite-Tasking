"""
V0 vs V3 (Student-t per-city R floor=5) LLM 검증 비교 — Precision 측정.

test window 10일 (2026-03-22 ~ 03-31). 각 day별 top-K=20 anomalies → LLM 검증.
LLM 캐시가 도시·날짜·기사 텍스트 기반이라 V0/V3가 같은 도시를 top-K에 올리면 캐시 hit.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import KALMAN_Q, KALMAN_R, MAIN_PATH, URL_PATH, PROJECT_ROOT, COEFS
from pipeline.preprocess import clean_gdelt_data, apply_standard_scaling
from pipeline.llm_verification import verify_anomalies_with_llm
from analysis.evaluator import ModelEvaluator
from analysis.robust_kalman_experiment import (
    apply_kalman_per_city, compute_a_tier, load_gt
)

GT_DIR = PROJECT_ROOT / "data" / "ground_truth"
OUT_DIR = PROJECT_ROOT / "logs" / "robust_kalman_experiment"


def main():
    print("[LOAD]")
    raw = pd.read_parquet(MAIN_PATH)
    url_df = pd.read_parquet(URL_PATH)

    filtered = clean_gdelt_data(raw)
    if 'date' not in filtered.columns:
        filtered['date'] = filtered['SQLDATE'].astype(str).str[:8]
    filtered = apply_standard_scaling(filtered, is_train=False)

    key_to_col = {'w_log_mentions': 'Log_Mentions', 'w_log_sources': 'Log_Sources',
                  'w_avgtone': 'AvgTone', 'w_avgtone_sq': 'AvgTone_Sq', 'w_goldstein': 'GoldsteinScale'}
    filtered['weighted_index'] = 0.0
    for k, w in COEFS.items():
        filtered['weighted_index'] += w * filtered[key_to_col[k]]

    agg = (filtered.groupby(['date', 'standard_name'])
           .agg(conflict_index=('weighted_index', 'sum'),
                events=('EventCode', 'count'),
                mentions=('NumMentions', 'mean'),
                articles=('NumArticles', 'mean'),
                sources=('NumSources', 'sum'),
                avg_tone=('AvgTone', 'mean'),
                lat=('ActionGeo_Lat', 'median'),
                lng=('ActionGeo_Long', 'median'))
           .reset_index().rename(columns={'standard_name': 'city'}))
    agg['date'] = pd.to_datetime(agg['date'])
    print(f"  city-day rows: {len(agg):,}")

    a_norms = compute_a_tier(filtered)
    gt = load_gt(GT_DIR)
    test_dates = sorted(gt[gt['date'].between('20260322', '20260331')]['date'].unique())

    # name_to_fids for evaluator
    name_col, fid_col = 'ActionGeo_FullName', 'ActionGeo_FeatureID'
    tmp = raw[[name_col, fid_col]].dropna().copy()
    tmp[name_col] = tmp[name_col].astype(str).str.strip().str.lower()
    name_to_fids = {}
    for nm, sub in tmp.groupby(name_col):
        s = set()
        for x in sub[fid_col].unique():
            try: s.add(int(x))
            except (TypeError, ValueError): pass
        name_to_fids[nm] = s
    evaluator = ModelEvaluator(GT_DIR, name_to_fids=name_to_fids)

    variants = [
        ('V0_Gaussian',     'gaussian',       4.0,   0.0,   1e9),
        ('V3_perR_floor5',  'studentt_perR',  4.0,   5.0,   5000.0),
    ]

    out_summary = {}
    for label, variant, dof, R_floor, R_ceil in variants:
        print(f"\n{'='*70}\n[VARIANT] {label}\n{'='*70}")
        kal_out, _ = apply_kalman_per_city(agg, variant, KALMAN_Q, KALMAN_R, dof,
                                           R_floor=R_floor, R_ceiling=R_ceil)
        # add risk_level (use thresholds 0.5/2/5)
        z = kal_out['std_innov'].values
        rl = np.where(z >= 5, 3, np.where(z >= 2, 2, np.where(z >= 0.5, 1, 0)))
        kal_out['risk_level'] = rl
        kal_out['is_anomaly'] = kal_out['risk_level'] >= 1
        # Need columns evaluator expects from full pipeline
        for col in ['events', 'mentions', 'sources', 'avg_tone']:
            kal_out[col] = 0  # filler

        date_metrics = []
        for d in test_dates:
            day = kal_out[(kal_out['date'] == d) & (kal_out['is_anomaly'])].copy()
            day = day.sort_values('std_innov', ascending=False).head(20)
            if day.empty:
                metrics = evaluator.evaluate(day.assign(llm_status='UNVERIFIED'), d)
                print(f"  [{d}] anomalies=0  → P=N/A R={metrics['recall']:.2%}")
                date_metrics.append({'date': d, **metrics})
                continue
            print(f"  [{d}] sending {len(day)} anomalies to LLM...")
            verified = verify_anomalies_with_llm(day, filtered, url_df, d, top_k=20)
            metrics = evaluator.evaluate(verified, d)
            print(f"    → TP={metrics['n_tp']} FP={metrics['n_fp']} EXC={metrics.get('n_excluded',0)}  "
                  f"P={metrics['precision']:.2%} R={metrics['recall']:.2%}")
            date_metrics.append({'date': d, **metrics})

        # 평균
        avg_p = np.mean([m['precision'] for m in date_metrics if m.get('precision') is not None])
        avg_r = np.mean([m['recall']    for m in date_metrics if m.get('recall')    is not None])
        out_summary[label] = {'per_date': date_metrics, 'avg_precision': avg_p, 'avg_recall': avg_r}
        print(f"  >> {label}: avg P={avg_p:.2%}  avg R={avg_r:.2%}")

    print(f"\n{'='*70}\n[SUMMARY]\n{'='*70}")
    for label, s in out_summary.items():
        print(f"{label:<20} avg P={s['avg_precision']:.2%}  avg R={s['avg_recall']:.2%}")

    out_path = OUT_DIR / "llm_eval_results.json"
    out_path.write_text(json.dumps(out_summary, ensure_ascii=False, indent=2, default=float))
    print(f"\n[OUT] {out_path}")


if __name__ == '__main__':
    main()
