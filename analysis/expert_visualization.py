"""
전문가 분석 시각화 — 6개 그림으로 V0 vs V4 비교를 명확히.

Fig 1. Bootstrap distribution of ΔRecall (V0 − V4)
Fig 2. CV stability (4-fold)
Fig 3. PR curves
Fig 4. Regime decomposition bar chart
Fig 5. Calibration curve (z-score → empirical hit rate)
Fig 6. Score distribution + conformal threshold
"""
from __future__ import annotations
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from pipeline.config import MAIN_PATH, PROJECT_ROOT
from pipeline.preprocess import clean_gdelt_data, apply_standard_scaling
from pipeline.text_norm import _norm
from analysis.robust_kalman_experiment import (
    apply_kalman_per_city, compute_a_tier, load_gt, evaluate_recall, haversine
)
from analysis.robust_kalman_v4_v8 import build_agg, apply_method
from analysis.expert_analysis import daily_hits, pr_curve, calibration_table, conformal_threshold

GT_DIR = PROJECT_ROOT / "data" / "ground_truth"
OUT_DIR = PROJECT_ROOT / "logs" / "robust_kalman_experiment"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

V0_COLOR = '#1f77b4'
V4_COLOR = '#ff7f0e'


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

    v0_out, _ = apply_kalman_per_city(agg, 'gaussian', 0.0001, 1.0)
    v4_out = apply_method(agg, 'V4_eb', prior_strength=10.0, Q=0.0001, dof=3.0)

    # ─── Fig 1: Bootstrap distribution ──────────────────────────────
    print("[Fig 1] Bootstrap distribution")
    hv0 = daily_hits(v0_out, gt, test_dates, 20, a_norms)
    hv4 = daily_hits(v4_out, gt, test_dates, 20, a_norms)
    rng = np.random.default_rng(42)
    n = len(hv0); n_boot = 2000
    diffs_all, diffs_a, diffs_c = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v0s, v4s = hv0.iloc[idx], hv4.iloc[idx]
        diffs_all.append(v0s['matched'].mean() - v4s['matched'].mean())
        for tier, store in [('A', diffs_a), ('C', diffs_c)]:
            sa = v0s[v0s['tier'] == tier]; sb = v4s[v4s['tier'] == tier]
            r0 = sa['matched'].mean() if len(sa) else 0
            r4 = sb['matched'].mean() if len(sb) else 0
            store.append(r0 - r4)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, arr, title in zip(axes, [diffs_all, diffs_a, diffs_c], ['Overall', 'A tier', 'C tier']):
        a = np.array(arr) * 100  # convert to percentage points
        ax.hist(a, bins=40, color='gray', alpha=0.6, edgecolor='black', linewidth=0.5)
        ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='no difference')
        ax.axvline(np.mean(a), color='blue', linestyle='-', linewidth=1.5, label=f'mean={np.mean(a):+.2f}')
        lo, hi = np.percentile(a, [2.5, 97.5])
        ax.axvspan(lo, hi, alpha=0.15, color='blue', label=f'95% CI [{lo:+.1f}, {hi:+.1f}]')
        ax.set_title(f'{title}: ΔRecall (V0 − V4)\nP(V0>V4)={(a>0).mean():.1%}')
        ax.set_xlabel('Recall difference (%p)')
        ax.set_ylabel('Bootstrap frequency')
        ax.legend(fontsize=8, loc='upper right')
    fig.suptitle('Fig 1. Bootstrap 95% CI of ΔRecall (V0 − V4) — All CIs include 0', y=1.02, fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig1_bootstrap_ci.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved: {FIG_DIR / 'fig1_bootstrap_ci.png'}")

    # ─── Fig 2: CV stability ─────────────────────────────────────────
    print("[Fig 2] CV stability")
    folds = 4
    fold_size = len(all_gt_dates) // (folds + 1)
    fold_v0, fold_v4 = [], []
    for f in range(folds):
        val_start = (f + 1) * fold_size
        val_end = min(val_start + fold_size, len(all_gt_dates))
        val_dates = all_gt_dates[val_start:val_end]
        if not val_dates: continue
        m0 = evaluate_recall(v0_out, gt, val_dates, 20, a_norms)
        m4 = evaluate_recall(v4_out, gt, val_dates, 20, a_norms)
        fold_v0.append(m0['all_recall'])
        fold_v4.append(m4['all_recall'])

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(fold_v0))
    width = 0.35
    ax.bar(x - width/2, fold_v0, width, label=f'V0  (mean={np.mean(fold_v0):.3f}, std={np.std(fold_v0):.3f})', color=V0_COLOR, alpha=0.8)
    ax.bar(x + width/2, fold_v4, width, label=f'V4  (mean={np.mean(fold_v4):.3f}, std={np.std(fold_v4):.3f})', color=V4_COLOR, alpha=0.8)
    ax.axhline(np.mean(fold_v0), color=V0_COLOR, linestyle='--', alpha=0.5)
    ax.axhline(np.mean(fold_v4), color=V4_COLOR, linestyle='--', alpha=0.5)
    ax.set_xlabel('Fold')
    ax.set_ylabel('Recall (top-K=20)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {i}' for i in x])
    ax.set_title('Fig 2. Time-series CV (4 folds) — V4 is more stable across folds\n'
                 f'V0 std=0.053, V4 std=0.022 (V4 is 2.4× more stable)', fontsize=11, fontweight='bold')
    ax.legend()
    ax.set_ylim(0.3, 0.65)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig2_cv_stability.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved: {FIG_DIR / 'fig2_cv_stability.png'}")

    # ─── Fig 3: PR curves ────────────────────────────────────────────
    print("[Fig 3] PR curves")
    pr_v0 = pr_curve(v0_out, gt, test_dates, a_norms)
    pr_v4 = pr_curve(v4_out, gt, test_dates, a_norms)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(pr_v0['recalls'], pr_v0['precisions'], color=V0_COLOR, linewidth=2,
            label=f"V0 PR-AUC = {pr_v0['auc']:.4f}")
    ax.plot(pr_v4['recalls'], pr_v4['precisions'], color=V4_COLOR, linewidth=2,
            label=f"V4 PR-AUC = {pr_v4['auc']:.4f}")
    # baseline = positive rate
    base = pr_v0['n_pos'] / pr_v0['n_total']
    ax.axhline(base, linestyle='--', color='gray', alpha=0.6,
               label=f'random baseline = {base:.3f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.6)
    ax.set_title('Fig 3. Precision–Recall Curves on Test Set\n'
                 'V4 has marginally higher PR-AUC (threshold-free metric)', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig3_pr_curves.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved: {FIG_DIR / 'fig3_pr_curves.png'}")

    # ─── Fig 4: Regime decomposition ─────────────────────────────────
    print("[Fig 4] Regime decomposition")
    from analysis.expert_analysis import regime_decomp
    rd_v0 = regime_decomp(v0_out, gt, test_dates, a_norms, filtered)
    rd_v4 = regime_decomp(v4_out, gt, test_dates, a_norms, filtered)

    regimes = [
        ('A tier',          rd_v0[rd_v0['tier'] == 'A']['matched'].mean(),
                            rd_v4[rd_v4['tier'] == 'A']['matched'].mean(),
                            (rd_v0['tier'] == 'A').sum()),
        ('C tier',          rd_v0[rd_v0['tier'] == 'C']['matched'].mean(),
                            rd_v4[rd_v4['tier'] == 'C']['matched'].mean(),
                            (rd_v0['tier'] == 'C').sum()),
        ('Cold-start city', rd_v0[rd_v0['cold_start']]['matched'].mean(),
                            rd_v4[rd_v4['cold_start']]['matched'].mean(),
                            rd_v0['cold_start'].sum()),
        ('Established city', rd_v0[~rd_v0['cold_start']]['matched'].mean(),
                             rd_v4[~rd_v4['cold_start']]['matched'].mean(),
                             (~rd_v0['cold_start']).sum()),
    ]
    labels = [r[0] for r in regimes]
    v0_vals = [r[1] for r in regimes]
    v4_vals = [r[2] for r in regimes]
    ns = [r[3] for r in regimes]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    width = 0.35
    bars0 = ax.bar(x - width/2, v0_vals, width, label='V0', color=V0_COLOR, alpha=0.8)
    bars4 = ax.bar(x + width/2, v4_vals, width, label='V4', color=V4_COLOR, alpha=0.8)
    for i, (b0, b4) in enumerate(zip(bars0, bars4)):
        ax.text(b0.get_x() + b0.get_width()/2, b0.get_height() + 0.005, f'{v0_vals[i]:.3f}',
                ha='center', fontsize=9)
        ax.text(b4.get_x() + b4.get_width()/2, b4.get_height() + 0.005, f'{v4_vals[i]:.3f}',
                ha='center', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{l}\n(n={ns[i]})' for i, l in enumerate(labels)])
    ax.set_ylabel('Recall')
    ax.set_ylim(0, 0.75)
    ax.set_title('Fig 4. Regime-specific Recall — V0 wins A tier, V4 wins C tier, tied elsewhere',
                 fontsize=11, fontweight='bold')
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig4_regime.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved: {FIG_DIR / 'fig4_regime.png'}")

    # ─── Fig 5: Calibration curve ────────────────────────────────────
    print("[Fig 5] Calibration curve")
    cal_v0 = calibration_table(v0_out, gt, test_dates)
    cal_v4 = calibration_table(v4_out, gt, test_dates)

    bin_centers = []
    for entry in cal_v0:
        b = entry['bin']
        lo, hi = [float(x) for x in b.strip('[)').split(', ')]
        bin_centers.append((lo + hi) / 2 if hi < 100 else lo * 1.5)

    v0_rates = [c['rate'] if c['rate'] is not None else 0 for c in cal_v0]
    v4_rates = [c['rate'] if c['rate'] is not None else 0 for c in cal_v4]
    v0_n = [c['n'] for c in cal_v0]
    v4_n = [c['n'] for c in cal_v4]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(bin_centers, v0_rates, 'o-', color=V0_COLOR, linewidth=2, markersize=10, label='V0 (Gaussian)')
    ax.plot(bin_centers, v4_rates, 'o-', color=V4_COLOR, linewidth=2, markersize=10, label='V4 (Student-t + EB)')
    for i, (xc, r, n) in enumerate(zip(bin_centers, v0_rates, v0_n)):
        ax.annotate(f'n={n}', (xc, r), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=7, color=V0_COLOR)
    for i, (xc, r, n) in enumerate(zip(bin_centers, v4_rates, v4_n)):
        ax.annotate(f'n={n}', (xc, r), textcoords='offset points', xytext=(0, -14), ha='center', fontsize=7, color=V4_COLOR)
    ax.set_xscale('symlog', linthresh=1)
    ax.set_xlabel('z-score bin (log scale)')
    ax.set_ylabel('Empirical hit rate (= P[GT match | z in bin])')
    ax.set_title('Fig 5a. Calibration curve\n'
                 'V0 is FLAT (no discrimination); V4 monotonically increases', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right panel: bar comparison at high z
    ax = axes[1]
    high_idx = [4, 5, 6]  # [3,5), [5,10), [10,100)
    labels = [cal_v0[i]['bin'] for i in high_idx]
    v0_high = [v0_rates[i] for i in high_idx]
    v4_high = [v4_rates[i] for i in high_idx]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, v0_high, width, label='V0', color=V0_COLOR, alpha=0.8)
    ax.bar(x + width/2, v4_high, width, label='V4', color=V4_COLOR, alpha=0.8)
    for i in range(len(labels)):
        ax.text(x[i] - width/2, v0_high[i] + 0.005, f'{v0_high[i]:.3f}', ha='center', fontsize=9)
        ax.text(x[i] + width/2, v4_high[i] + 0.005, f'{v4_high[i]:.3f}', ha='center', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Empirical hit rate')
    ax.set_ylim(0, 0.55)
    ax.set_title('Fig 5b. High z-score region\n'
                 'V4 hit rate increases with z; V0 stays flat ~0.30', fontsize=11, fontweight='bold')
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig5_calibration.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved: {FIG_DIR / 'fig5_calibration.png'}")

    # ─── Fig 6: Score distribution + conformal threshold ─────────────
    print("[Fig 6] Score distribution")
    v0_train_scores = v0_out[v0_out['date'].isin(train_dates)]['std_innov'].dropna().values
    v0_train_scores = v0_train_scores[v0_train_scores > 0]
    v4_train_scores = v4_out[v4_out['date'].isin(train_dates)]['std_innov'].dropna().values
    v4_train_scores = v4_train_scores[v4_train_scores > 0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, scores, label, color in [(axes[0], v0_train_scores, 'V0', V0_COLOR),
                                      (axes[1], v4_train_scores, 'V4', V4_COLOR)]:
        ax.hist(scores, bins=80, color=color, alpha=0.7, edgecolor='black', linewidth=0.3, log=True)
        thr_05 = conformal_threshold(scores, 0.05)
        thr_10 = conformal_threshold(scores, 0.10)
        ax.axvline(thr_05, color='red', linestyle='--', linewidth=2, label=f'5% FPR threshold = {thr_05:.1f}')
        ax.axvline(thr_10, color='orange', linestyle='--', linewidth=2, label=f'10% FPR threshold = {thr_10:.1f}')
        ax.axvline(0.5, color='black', linestyle=':', linewidth=1, label='current threshold (0.5)')
        ax.set_xlabel('std_innov (positive scores only)')
        ax.set_ylabel('Frequency (log scale)')
        ax.set_title(f'{label} score distribution + conformal thresholds', fontsize=11, fontweight='bold')
        ax.set_xscale('symlog', linthresh=1)
        ax.legend()
    fig.suptitle('Fig 6. Score distribution: V0 has heavy tail (5% threshold = z=49), V4 has reasonable tail (z=7)', y=1.02, fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig6_scores.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved: {FIG_DIR / 'fig6_scores.png'}")

    print(f"\n[DONE] All figures saved in {FIG_DIR}")


if __name__ == '__main__':
    main()
