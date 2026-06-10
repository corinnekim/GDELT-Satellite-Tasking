"""
최종 비교: V0 (Gaussian Q=0.0001 R=1) vs V4_eb_prior10 (EB per-city R, dof=3) — LLM 검증 포함.

test 10일 window 평가.
지표: top-K Recall (LLM 단독), LLM 통과 후 Precision, F_β=1.5.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import KALMAN_P0_RATIO, MAIN_PATH, URL_PATH, PROJECT_ROOT, COEFS
from pipeline.preprocess import clean_gdelt_data, apply_standard_scaling
from pipeline.llm_verification import verify_anomalies_with_llm
from analysis.robust_kalman_experiment import (
    apply_kalman_per_city, compute_a_tier, load_gt, evaluate_recall, haversine
)
from analysis.robust_kalman_v4_v8 import build_agg, apply_method, compute_eb_R
from pipeline.text_norm import _norm

GT_DIR = PROJECT_ROOT / "data" / "ground_truth"
OUT_DIR = PROJECT_ROOT / "logs" / "robust_kalman_experiment"


def precision_from_llm(verified_df: pd.DataFrame) -> dict:
    """LLM 통과 후 Precision = TP / (TP + FP). EXC 제외."""
    TP = {'SUCCESS', 'AMBIGUOUS'}
    FP = {'DROPPED', 'NO_MENTION', 'DATE_MISMATCH'}
    EXC = {'ARTICLE_UNREACHABLE', 'ERROR'}
    counts = verified_df['llm_status'].value_counts().to_dict()
    n_tp = sum(counts.get(s, 0) for s in TP)
    n_fp = sum(counts.get(s, 0) for s in FP)
    n_exc = sum(counts.get(s, 0) for s in EXC)
    p = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else None
    return {'TP': n_tp, 'FP': n_fp, 'EXC': n_exc, 'precision': p, 'counts': counts}


def main():
    print("[LOAD]")
    raw = pd.read_parquet(MAIN_PATH)
    url_df = pd.read_parquet(URL_PATH)
    filtered = clean_gdelt_data(raw)
    if 'date' not in filtered.columns:
        filtered['date'] = filtered['SQLDATE'].astype(str).str[:8]
    filtered = apply_standard_scaling(filtered, is_train=False)

    agg = build_agg(filtered, multivariate=False)
    a_norms = compute_a_tier(filtered)
    gt = load_gt(GT_DIR)
    test_dates = sorted(gt[gt['date'].between('20260322', '20260331')]['date'].unique())
    print(f"  test_dates: {len(test_dates)}")

    variants = [
        ('V0_baseline',     'gaussian',  dict(Q=0.0001, R=1.0)),
        ('V4_eb_prior10',   'V4_eb',     dict(prior_strength=10.0, Q=0.0001, dof=3.0)),
    ]

    results = {}
    for label, kind, cfg in variants:
        print(f"\n{'='*70}\n[{label}]  {cfg}\n{'='*70}")
        if kind == 'gaussian':
            kal_out, _ = apply_kalman_per_city(agg, 'gaussian', cfg['Q'], cfg['R'])
        elif kind == 'V4_eb':
            kal_out = apply_method(agg, 'V4_eb', **cfg)

        # threshold for sending to LLM: 그대로 0.5 사용 + top_k=20
        z = kal_out['std_innov'].values
        kal_out['risk_level'] = np.where(z >= 5, 3, np.where(z >= 2, 2, np.where(z >= 0.5, 1, 0)))
        kal_out['is_anomaly'] = kal_out['risk_level'] >= 1
        for col in ['events', 'mentions', 'sources', 'avg_tone']:
            kal_out[col] = 0  # filler

        per_date = []
        for d in test_dates:
            day_all = kal_out[kal_out['date'] == d].copy()
            day = day_all[day_all['is_anomaly']].sort_values('std_innov', ascending=False).head(20)
            if day.empty:
                per_date.append({'date': d, 'top_k': 0, 'precision': None, 'recall': None})
                print(f"  [{d}] no anomalies")
                continue
            print(f"  [{d}] {len(day)} anomalies → LLM")
            verified = verify_anomalies_with_llm(day, filtered, url_df, d, top_k=20)
            p = precision_from_llm(verified)

            # Recall (this date) — same evaluation logic as before
            local = pd.DataFrame({'date': [d]*len(day), 'city': day['city'].values,
                                  'std_innov': day['std_innov'].values,
                                  'lat': day.get('lat', pd.Series([np.nan]*len(day))).values,
                                  'lng': day.get('lng', pd.Series([np.nan]*len(day))).values})
            r = evaluate_recall(local, gt, [d], 20, a_norms)
            print(f"    TP={p['TP']} FP={p['FP']} EXC={p['EXC']}  P={p['precision']:.2%}  "
                  f"R(top-K)={r['all_recall']:.2%}  status={p['counts']}")
            per_date.append({'date': d, 'top_k': len(day), **p,
                             'recall_topk': r['all_recall'], 'recall_A': r['A_recall'], 'recall_C': r['C_recall']})

        # Aggregate (entire test window)
        # full Recall via evaluate_recall on the whole test window
        full = evaluate_recall(kal_out.assign(std_innov=lambda d: d['std_innov'].fillna(0)),
                               gt, test_dates, 20, a_norms)
        # Aggregate precision over verified rows only
        agg_tp = sum(d.get('TP', 0) or 0 for d in per_date)
        agg_fp = sum(d.get('FP', 0) or 0 for d in per_date)
        avg_prec = agg_tp / (agg_tp + agg_fp) if (agg_tp + agg_fp) > 0 else None

        beta = 1.5
        if avg_prec is not None and full['all_recall'] > 0:
            f_beta = (1 + beta**2) * (avg_prec * full['all_recall']) / (beta**2 * avg_prec + full['all_recall'])
        else:
            f_beta = None

        results[label] = {
            'config': cfg,
            'agg_precision': avg_prec,
            'recall_top_k_all': full['all_recall'],
            'recall_top_k_A':   full['A_recall'],
            'recall_top_k_C':   full['C_recall'],
            'F_beta_1.5': f_beta,
            'TP_total': agg_tp, 'FP_total': agg_fp,
            'per_date': per_date,
        }
        print(f"\n  >> {label}: P={avg_prec:.2%}  R={full['all_recall']:.2%} (A={full['A_recall']:.2%}, C={full['C_recall']:.2%})  F={f_beta:.3f}")

    print(f"\n{'='*70}\n[FINAL COMPARISON]\n{'='*70}")
    print(f"{'Variant':<22} {'P':<10} {'R(All)':<10} {'R(A)':<10} {'R(C)':<10} {'F1.5':<10}")
    print("-"*70)
    for label, r in results.items():
        print(f"{label:<22} {r['agg_precision']:<10.3%} {r['recall_top_k_all']:<10.3%} "
              f"{r['recall_top_k_A']:<10.3%} {r['recall_top_k_C']:<10.3%} {r['F_beta_1.5']:<10.3f}")

    out_path = OUT_DIR / "llm_final_comparison.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=float))
    print(f"\n[OUT] {out_path}")


if __name__ == '__main__':
    main()
