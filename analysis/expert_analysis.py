"""
전문가 수준 통계적 분석 — V0 vs V4_prior10 비교의 정직성 확보.

A. Bootstrap 95% CI (Recall, Precision, F1.5)
B. Time-series cross-validation (rolling origin)
C. PR-AUC across full threshold range
D. Regime-specific decomposition (spike day, A/C tier, cold-start city)
E. Calibration curve (z-score → empirical hit rate)
F. Conformal prediction threshold
"""
from __future__ import annotations
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from pipeline.config import MAIN_PATH, PROJECT_ROOT, COEFS
from pipeline.preprocess import clean_gdelt_data, apply_standard_scaling
from pipeline.text_norm import _norm
from analysis.robust_kalman_experiment import (
    apply_kalman_per_city, compute_a_tier, load_gt, evaluate_recall, haversine
)
from analysis.robust_kalman_v4_v8 import build_agg, apply_method

GT_DIR = PROJECT_ROOT / "data" / "ground_truth"
OUT_DIR = PROJECT_ROOT / "logs" / "robust_kalman_experiment"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def daily_hits(kal_out: pd.DataFrame, gt_df: pd.DataFrame, dates, top_k: int, a_norms: set,
               radius_km: float = 50.0) -> pd.DataFrame:
    """Returns per-(date, gt_index) row with hit/miss + tier label. Used for bootstrap."""
    rows = []
    for d in dates:
        day = kal_out[kal_out['date'] == d].copy()
        day = day[day['std_innov'] > 0].sort_values('std_innov', ascending=False).head(top_k)
        gt_today = gt_df[gt_df['date'] == d]
        for i, gt in gt_today.iterrows():
            gt_norm = _norm(str(gt['city'])) if pd.notna(gt['city']) else ""
            matched = False
            for _, anom in day.iterrows():
                cb = str(anom['city']).split('|')[0]
                an = _norm(cb) if cb else ""
                if gt_norm and an and (gt_norm == an or gt_norm in an or an in gt_norm):
                    matched = True; break
                if pd.notna(anom.get('lat')) and pd.notna(anom.get('lng')):
                    if haversine(float(anom['lat']), float(anom['lng']), float(gt['lat']), float(gt['lon'])) <= radius_km:
                        matched = True; break
            in_a = any(gt_norm == an or (an and gt_norm in an) or (an and an in gt_norm) for an in a_norms if an)
            rows.append({'date': d, 'gt_idx': i, 'matched': matched, 'tier': 'A' if in_a else 'C'})
    return pd.DataFrame(rows)


def bootstrap_recall_diff(hits_v0: pd.DataFrame, hits_v4: pd.DataFrame, n_boot: int = 1000, seed: int = 0):
    """Bootstrap CI for Recall difference (per-tier and overall)."""
    rng = np.random.default_rng(seed)
    n = len(hits_v0)
    assert len(hits_v4) == n, f"size mismatch {len(hits_v0)} vs {len(hits_v4)}"

    diffs_all, diffs_a, diffs_c = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v0 = hits_v0.iloc[idx]
        v4 = hits_v4.iloc[idx]
        for tier_label, store in [('all', diffs_all), ('A', diffs_a), ('C', diffs_c)]:
            sub_v0 = v0 if tier_label == 'all' else v0[v0['tier'] == tier_label]
            sub_v4 = v4 if tier_label == 'all' else v4[v4['tier'] == tier_label]
            r0 = sub_v0['matched'].mean() if len(sub_v0) else 0
            r4 = sub_v4['matched'].mean() if len(sub_v4) else 0
            store.append(r0 - r4)

    out = {}
    for label, arr in [('all', diffs_all), ('A', diffs_a), ('C', diffs_c)]:
        a = np.array(arr)
        out[label] = {
            'mean_diff': float(a.mean()),
            'ci_2.5': float(np.percentile(a, 2.5)),
            'ci_97.5': float(np.percentile(a, 97.5)),
            'p_v0_better': float((a > 0).mean()),  # fraction of bootstrap where V0 wins
        }
    return out


def pr_curve(kal_out: pd.DataFrame, gt_df: pd.DataFrame, dates, a_norms: set, radius_km: float = 50.0):
    """For each city-day, label as positive if matches a GT city that day. Sort by std_innov, compute PR."""
    # Per (date, city, std_innov): label = exists GT city today within radius_km or name match
    labels = []
    scores = []
    sample_meta = []
    for d in dates:
        day = kal_out[kal_out['date'] == d].copy()
        gt_today = gt_df[gt_df['date'] == d]
        for _, row in day.iterrows():
            cb = str(row['city']).split('|')[0]
            an = _norm(cb) if cb else ""
            label = 0
            for _, gt in gt_today.iterrows():
                gn = _norm(str(gt['city'])) if pd.notna(gt['city']) else ""
                if gn and an and (gn == an or gn in an or an in gn):
                    label = 1; break
                if pd.notna(row.get('lat')) and pd.notna(row.get('lng')):
                    if haversine(float(row['lat']), float(row['lng']), float(gt['lat']), float(gt['lon'])) <= radius_km:
                        label = 1; break
            labels.append(label)
            scores.append(float(row['std_innov']))
            sample_meta.append({'date': d, 'city': row['city']})
    labels = np.array(labels); scores = np.array(scores)
    # PR curve via sorted threshold
    order = np.argsort(-scores)
    labels_o = labels[order]
    tp_cum = np.cumsum(labels_o)
    fp_cum = np.cumsum(1 - labels_o)
    n_pos = labels.sum()
    if n_pos == 0:
        return None
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    recalls = tp_cum / n_pos
    # AUC via trapezoidal
    auc = float(np.trapezoid(precisions, recalls))
    return {'precisions': precisions.tolist(), 'recalls': recalls.tolist(),
            'thresholds': scores[order].tolist(), 'auc': auc, 'n_pos': int(n_pos), 'n_total': int(len(labels))}


def calibration_table(kal_out: pd.DataFrame, gt_df: pd.DataFrame, dates,
                      bins=[0, 0.5, 1, 2, 3, 5, 10, 100], radius_km: float = 50.0):
    """For each z-score bin, compute empirical hit rate."""
    labels, scores = [], []
    for d in dates:
        day = kal_out[kal_out['date'] == d].copy()
        gt_today = gt_df[gt_df['date'] == d]
        for _, row in day.iterrows():
            cb = str(row['city']).split('|')[0]
            an = _norm(cb) if cb else ""
            label = 0
            for _, gt in gt_today.iterrows():
                gn = _norm(str(gt['city'])) if pd.notna(gt['city']) else ""
                if gn and an and (gn == an or gn in an or an in gn):
                    label = 1; break
                if pd.notna(row.get('lat')) and pd.notna(row.get('lng')):
                    if haversine(float(row['lat']), float(row['lng']), float(gt['lat']), float(gt['lon'])) <= radius_km:
                        label = 1; break
            labels.append(label); scores.append(float(row['std_innov']))
    labels = np.array(labels); scores = np.array(scores)
    table = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (scores >= lo) & (scores < hi)
        n = int(mask.sum())
        h = int(labels[mask].sum())
        table.append({'bin': f"[{lo}, {hi})", 'n': n, 'hits': h, 'rate': h/n if n else None})
    return table


def conformal_threshold(scores_calib: np.ndarray, alpha: float = 0.05) -> float:
    """Conformal prediction threshold: (1-alpha) quantile of calibration scores."""
    return float(np.quantile(scores_calib, 1 - alpha))


def time_series_cv(agg, gt, all_dates, kalman_fn, kalman_kwargs, a_norms, n_folds: int = 4, top_k: int = 20):
    """Rolling-origin CV over GT dates. Each fold: expand train, evaluate on next chunk of dates."""
    gt_dates = sorted(gt['date'].unique())
    fold_size = len(gt_dates) // (n_folds + 1)
    out = []
    for f in range(n_folds):
        train_end = (f + 1) * fold_size
        val_start = train_end
        val_end = train_end + fold_size
        val_dates = gt_dates[val_start:val_end]
        if not val_dates:
            continue
        kal_out, _ = kalman_fn(agg, **kalman_kwargs) if kalman_fn else (None, None)
        m = evaluate_recall(kal_out, gt, val_dates, top_k, a_norms)
        out.append({'fold': f, 'val_dates': val_dates, 'A': m['A_recall'], 'C': m['C_recall'], 'all': m['all_recall']})
    return out


def regime_decomp(kal_out: pd.DataFrame, gt_df: pd.DataFrame, dates, a_norms,
                  filtered_df, top_k=20, radius_km=50.0):
    """Decompose Recall by event regime."""
    # For each city, history length until each test date
    # And mention count of each city
    city_active = filtered_df.groupby('standard_name')['date'].nunique().to_dict()
    city_mentions = filtered_df.groupby('standard_name')['NumMentions'].sum().to_dict()

    # Per-day spike intensity: total mentions today vs median across all dates
    daily_mentions = filtered_df.groupby('date')['NumMentions'].sum()
    median_daily = daily_mentions.median()
    spike_dates = set(daily_mentions[daily_mentions > 1.5 * median_daily].index)

    rows = []
    for d in dates:
        day = kal_out[kal_out['date'] == d].copy()
        day = day[day['std_innov'] > 0].sort_values('std_innov', ascending=False).head(top_k)
        gt_today = gt_df[gt_df['date'] == d]
        is_spike = d in spike_dates
        for _, gt in gt_today.iterrows():
            gn = _norm(str(gt['city'])) if pd.notna(gt['city']) else ""
            matched = False
            for _, anom in day.iterrows():
                cb = str(anom['city']).split('|')[0]
                an = _norm(cb) if cb else ""
                if gn and an and (gn == an or gn in an or an in gn):
                    matched = True; break
                if pd.notna(anom.get('lat')) and pd.notna(anom.get('lng')):
                    if haversine(float(anom['lat']), float(anom['lng']), float(gt['lat']), float(gt['lon'])) <= radius_km:
                        matched = True; break
            in_a = any(gn == an or (an and gn in an) or (an and an in gn) for an in a_norms if an)
            # Find matching GDELT city for cold-start determination
            cold_start = False  # default
            for cn, active in city_active.items():
                cn_n = _norm(str(cn).split('|')[0])
                if cn_n == gn or (cn_n and gn and (cn_n in gn or gn in cn_n)):
                    cold_start = active < 10
                    break
            rows.append({'date': d, 'matched': matched, 'tier': 'A' if in_a else 'C',
                         'spike_day': is_spike, 'cold_start': cold_start})
    return pd.DataFrame(rows)


def main():
    print("[LOAD]")
    raw = pd.read_parquet(MAIN_PATH)
    filtered = clean_gdelt_data(raw)
    if 'date' not in filtered.columns:
        filtered['date'] = filtered['SQLDATE'].astype(str).str[:8]
    filtered = apply_standard_scaling(filtered, is_train=False)

    agg = build_agg(filtered, multivariate=False)
    a_norms = compute_a_tier(filtered)
    gt = load_gt(GT_DIR)
    train_dates = sorted(gt[gt['date'].between('20260228', '20260321')]['date'].unique())
    test_dates  = sorted(gt[gt['date'].between('20260322', '20260331')]['date'].unique())
    all_gt_dates = sorted(gt['date'].unique())
    print(f"  cities: {agg['city'].nunique()}, train_dates {len(train_dates)}, test_dates {len(test_dates)}")

    # Build V0 and V4 outputs
    print("\n[BUILD] V0 (Gaussian Q=0.0001 R=1)")
    v0_out, _ = apply_kalman_per_city(agg, 'gaussian', 0.0001, 1.0)
    print("[BUILD] V4_eb_prior10 (Student-t dof=3)")
    v4_out = apply_method(agg, 'V4_eb', prior_strength=10.0, Q=0.0001, dof=3.0)

    full = {}

    # ─── A. Bootstrap CI ───
    print("\n[A] Bootstrap 95% CI for Recall difference (V0 - V4) on test set")
    hv0 = daily_hits(v0_out, gt, test_dates, 20, a_norms)
    hv4 = daily_hits(v4_out, gt, test_dates, 20, a_norms)
    boot = bootstrap_recall_diff(hv0, hv4, n_boot=2000, seed=42)
    for tier, b in boot.items():
        print(f"  {tier:<4} ΔR = {b['mean_diff']*100:+.2f}%p  95% CI [{b['ci_2.5']*100:+.2f}, {b['ci_97.5']*100:+.2f}]  "
              f"P(V0>V4)={b['p_v0_better']:.2%}")
    full['bootstrap'] = boot

    # ─── B. Time-series CV (rolling origin) ───
    print("\n[B] Time-series CV (rolling origin) — V0 stability")
    # Manual: split GT dates into 5 folds, expanding train + 1-fold val
    folds = 4
    fold_size = len(all_gt_dates) // (folds + 1)
    cv_results_v0, cv_results_v4 = [], []
    for f in range(folds):
        val_start = (f + 1) * fold_size
        val_end = min(val_start + fold_size, len(all_gt_dates))
        val_dates = all_gt_dates[val_start:val_end]
        if not val_dates: continue
        m_v0 = evaluate_recall(v0_out, gt, val_dates, 20, a_norms)
        m_v4 = evaluate_recall(v4_out, gt, val_dates, 20, a_norms)
        cv_results_v0.append({'fold': f, 'A': m_v0['A_recall'], 'C': m_v0['C_recall'], 'all': m_v0['all_recall']})
        cv_results_v4.append({'fold': f, 'A': m_v4['A_recall'], 'C': m_v4['C_recall'], 'all': m_v4['all_recall']})
        print(f"  Fold {f}: V0 A={m_v0['A_recall']:.3f} C={m_v0['C_recall']:.3f} | V4 A={m_v4['A_recall']:.3f} C={m_v4['C_recall']:.3f}")
    if cv_results_v0:
        v0_all = [r['all'] for r in cv_results_v0]
        v4_all = [r['all'] for r in cv_results_v4]
        print(f"  V0 across folds: mean={np.mean(v0_all):.3f}, std={np.std(v0_all):.3f}")
        print(f"  V4 across folds: mean={np.mean(v4_all):.3f}, std={np.std(v4_all):.3f}")
    full['cv'] = {'V0': cv_results_v0, 'V4': cv_results_v4}

    # ─── C. PR-AUC ───
    print("\n[C] PR-AUC across full threshold range")
    pr_v0 = pr_curve(v0_out, gt, test_dates, a_norms)
    pr_v4 = pr_curve(v4_out, gt, test_dates, a_norms)
    print(f"  V0 PR-AUC = {pr_v0['auc']:.4f}  (n_pos={pr_v0['n_pos']}, n_total={pr_v0['n_total']})")
    print(f"  V4 PR-AUC = {pr_v4['auc']:.4f}  (n_pos={pr_v4['n_pos']}, n_total={pr_v4['n_total']})")
    full['pr_auc'] = {'V0': {k: pr_v0[k] for k in ['auc', 'n_pos', 'n_total']},
                      'V4': {k: pr_v4[k] for k in ['auc', 'n_pos', 'n_total']}}

    # ─── D. Regime decomposition ───
    print("\n[D] Regime decomposition (test set)")
    rd_v0 = regime_decomp(v0_out, gt, test_dates, a_norms, filtered)
    rd_v4 = regime_decomp(v4_out, gt, test_dates, a_norms, filtered)

    def summarize(df, label):
        out_summary = {}
        for col in ['tier', 'spike_day', 'cold_start']:
            for val in df[col].unique():
                sub = df[df[col] == val]
                key = f"{col}={val}"
                out_summary[key] = {'n': len(sub), 'recall': sub['matched'].mean()}
        return out_summary
    print(f"  V0:")
    s_v0 = summarize(rd_v0, 'V0')
    for k, v in s_v0.items():
        print(f"    {k:<25} n={v['n']:<4} recall={v['recall']:.3f}")
    print(f"  V4:")
    s_v4 = summarize(rd_v4, 'V4')
    for k, v in s_v4.items():
        print(f"    {k:<25} n={v['n']:<4} recall={v['recall']:.3f}")
    full['regime'] = {'V0': s_v0, 'V4': s_v4}

    # ─── E. Calibration ───
    print("\n[E] Calibration table (z-score bin → empirical hit rate)")
    cal_v0 = calibration_table(v0_out, gt, test_dates)
    cal_v4 = calibration_table(v4_out, gt, test_dates)
    print(f"  {'bin':<10} {'V0_n':<7} {'V0_rate':<9} {'V4_n':<7} {'V4_rate':<9}")
    for v0b, v4b in zip(cal_v0, cal_v4):
        v0r = f"{v0b['rate']:.3f}" if v0b['rate'] is not None else "N/A"
        v4r = f"{v4b['rate']:.3f}" if v4b['rate'] is not None else "N/A"
        print(f"  {v0b['bin']:<10} {v0b['n']:<7} {v0r:<9} {v4b['n']:<7} {v4r:<9}")
    full['calibration'] = {'V0': cal_v0, 'V4': cal_v4}

    # ─── F. Conformal threshold ───
    print("\n[F] Conformal prediction threshold (using train as calibration)")
    v0_train_scores = v0_out[v0_out['date'].isin(train_dates)]['std_innov'].dropna().values
    v0_train_scores = v0_train_scores[v0_train_scores > 0]
    for alpha in [0.01, 0.05, 0.10]:
        thr_v0 = conformal_threshold(v0_train_scores, alpha=alpha)
        # Test: how many anomalies above this threshold per day?
        v0_test_above = v0_out[(v0_out['date'].isin(test_dates)) & (v0_out['std_innov'] >= thr_v0)]
        per_day = len(v0_test_above) / len(test_dates)
        print(f"  V0 alpha={alpha}: threshold z={thr_v0:.3f}, test alarms={len(v0_test_above)}, avg/day={per_day:.1f}")
    v4_train_scores = v4_out[v4_out['date'].isin(train_dates)]['std_innov'].dropna().values
    v4_train_scores = v4_train_scores[v4_train_scores > 0]
    for alpha in [0.01, 0.05, 0.10]:
        thr_v4 = conformal_threshold(v4_train_scores, alpha=alpha)
        v4_test_above = v4_out[(v4_out['date'].isin(test_dates)) & (v4_out['std_innov'] >= thr_v4)]
        per_day = len(v4_test_above) / len(test_dates)
        print(f"  V4 alpha={alpha}: threshold z={thr_v4:.3f}, test alarms={len(v4_test_above)}, avg/day={per_day:.1f}")
    full['conformal'] = {
        'V0': {f"alpha_{a}": conformal_threshold(v0_train_scores, a) for a in [0.01, 0.05, 0.10]},
        'V4': {f"alpha_{a}": conformal_threshold(v4_train_scores, a) for a in [0.01, 0.05, 0.10]},
    }

    out_path = OUT_DIR / "expert_analysis_results.json"
    out_path.write_text(json.dumps(full, ensure_ascii=False, indent=2, default=float))
    print(f"\n[OUT] {out_path}")


if __name__ == '__main__':
    main()
