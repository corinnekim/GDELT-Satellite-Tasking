"""
공정 비교: V0(Gaussian) vs V3(Student-t per-city R) — 각자 자기 hyperparameter로 그리드서치.

V0 그리드: Q ∈ {0.0001, 0.001, 0.01}, R ∈ {1, 5, 10, 50}
V3 그리드: Q ∈ {0.0001, 0.001, 0.01}, R_floor ∈ {1, 5, 10, 50}, dof ∈ {3, 4, 6}

평가: train Recall (top-K=20, A/C/All).
선택: 각 변형의 best (Q,R)를 train Recall로 결정 → test Recall 보고.
"""
from __future__ import annotations
import json
from pathlib import Path
import itertools

import numpy as np
import pandas as pd

from pipeline.config import KALMAN_P0_RATIO, MAIN_PATH, PROJECT_ROOT, COEFS
from pipeline.preprocess import clean_gdelt_data, apply_standard_scaling
from analysis.robust_kalman_experiment import (
    apply_kalman_per_city, compute_a_tier, load_gt, evaluate_recall
)

GT_DIR = PROJECT_ROOT / "data" / "ground_truth"
OUT_DIR = PROJECT_ROOT / "logs" / "robust_kalman_experiment"


def main():
    print("[LOAD]")
    raw = pd.read_parquet(MAIN_PATH)
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
                lat=('ActionGeo_Lat', 'median'),
                lng=('ActionGeo_Long', 'median'))
           .reset_index().rename(columns={'standard_name': 'city'}))
    agg['date'] = pd.to_datetime(agg['date'])

    a_norms = compute_a_tier(filtered)
    gt = load_gt(GT_DIR)
    train_dates = sorted(gt[gt['date'].between('20260228', '20260321')]['date'].unique())
    test_dates  = sorted(gt[gt['date'].between('20260322', '20260331')]['date'].unique())
    print(f"  cities: {agg['city'].nunique()}, train_dates {len(train_dates)}, test_dates {len(test_dates)}")

    # ── V0 grid ─────────────────────────────────────
    print(f"\n{'='*70}\n[GRID] V0 Gaussian — Q × R\n{'='*70}")
    v0_grid = list(itertools.product([0.0001, 0.001, 0.01], [1.0, 5.0, 10.0, 50.0]))
    v0_results = []
    for Q, R in v0_grid:
        out, _ = apply_kalman_per_city(agg, 'gaussian', Q, R)
        tr = evaluate_recall(out, gt, train_dates, 20, a_norms)
        te = evaluate_recall(out, gt, test_dates, 20, a_norms)
        v0_results.append({'Q': Q, 'R': R, 'train': tr, 'test': te})
        print(f"  Q={Q:<7} R={R:<6}  train: A={tr['A_recall']:.3f} C={tr['C_recall']:.3f} All={tr['all_recall']:.3f}  "
              f"|  test: A={te['A_recall']:.3f} C={te['C_recall']:.3f} All={te['all_recall']:.3f}")

    # ── V3 grid ─────────────────────────────────────
    print(f"\n{'='*70}\n[GRID] V3 Student-t per-city R — Q × R_floor × dof\n{'='*70}")
    v3_grid = list(itertools.product([0.0001, 0.001, 0.01], [1.0, 5.0, 10.0, 50.0], [3.0, 4.0, 6.0]))
    v3_results = []
    for Q, Rf, dof in v3_grid:
        # global R는 의미 없음 (per-city), 그냥 floor로 통일
        out, R_log = apply_kalman_per_city(agg, 'studentt_perR', Q, Rf, dof, R_floor=Rf, R_ceiling=5000.0)
        tr = evaluate_recall(out, gt, train_dates, 20, a_norms)
        te = evaluate_recall(out, gt, test_dates,  20, a_norms)
        v3_results.append({'Q': Q, 'R_floor': Rf, 'dof': dof, 'train': tr, 'test': te,
                           'R_median': float(R_log['R'].median())})
        print(f"  Q={Q:<7} Rf={Rf:<5} dof={dof}  train: A={tr['A_recall']:.3f} C={tr['C_recall']:.3f} All={tr['all_recall']:.3f}  "
              f"|  test: A={te['A_recall']:.3f} C={te['C_recall']:.3f} All={te['all_recall']:.3f}")

    # ── best ─────────────────────────────────────
    v0_best = max(v0_results, key=lambda r: r['train']['all_recall'])
    v3_best = max(v3_results, key=lambda r: r['train']['all_recall'])
    v3_best_a = max(v3_results, key=lambda r: r['train']['A_recall'])
    print(f"\n{'='*70}\n[BEST]")
    print(f"V0 best by train All Recall: Q={v0_best['Q']} R={v0_best['R']}")
    print(f"  train: A={v0_best['train']['A_recall']:.3f} C={v0_best['train']['C_recall']:.3f} All={v0_best['train']['all_recall']:.3f}")
    print(f"  test:  A={v0_best['test']['A_recall']:.3f} C={v0_best['test']['C_recall']:.3f} All={v0_best['test']['all_recall']:.3f}")
    print(f"V3 best by train All Recall: Q={v3_best['Q']} Rf={v3_best['R_floor']} dof={v3_best['dof']}")
    print(f"  train: A={v3_best['train']['A_recall']:.3f} C={v3_best['train']['C_recall']:.3f} All={v3_best['train']['all_recall']:.3f}")
    print(f"  test:  A={v3_best['test']['A_recall']:.3f} C={v3_best['test']['C_recall']:.3f} All={v3_best['test']['all_recall']:.3f}")
    print(f"V3 best by train A-tier Recall: Q={v3_best_a['Q']} Rf={v3_best_a['R_floor']} dof={v3_best_a['dof']}")
    print(f"  train: A={v3_best_a['train']['A_recall']:.3f} C={v3_best_a['train']['C_recall']:.3f}")
    print(f"  test:  A={v3_best_a['test']['A_recall']:.3f} C={v3_best_a['test']['C_recall']:.3f}")

    out_path = OUT_DIR / "grid_results.json"
    out_path.write_text(json.dumps({'V0': v0_results, 'V3': v3_results}, ensure_ascii=False, indent=2, default=float))
    print(f"\n[OUT] {out_path}")


if __name__ == '__main__':
    main()
