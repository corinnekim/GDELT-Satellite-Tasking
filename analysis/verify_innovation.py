"""우선순위 1: 평시 innovation z의 정규성 검증.

논리:
  Kalman은 H0(평시) = 가우시안 baseline → innovation z ~ N(0, 1)
  H0 위반(사건) → z 이상치 = anomaly signal

검증 절차:
  1. 22-day train window (2/28 ~ 3/21) Kalman 전체 city-day 시계열 → std_innov 추출
  2. (city, date) ∈ GT positives = 사건일, 그 외 = 평시
  3. 평시 z 분포 vs N(0, 1) 적합도
     - 히스토그램 + 정규분포 overlay
     - QQ plot
     - Shapiro-Wilk (n≤5000) or Anderson-Darling, KS test
     - mean, std, skew, kurtosis
  4. 결과 numpy + 그림 저장 + console summary
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from pipeline.config import MAIN_PATH, PROJECT_ROOT
from pipeline.preprocess import clean_gdelt_data, apply_standard_scaling
from pipeline.kalman_filter import apply_kalman_group
from pipeline.training.fit_coefs import load_window_gdelt, map_gt_to_positives

OUT = PROJECT_ROOT / "analysis" / "results"
OUT.mkdir(parents=True, exist_ok=True)

WIN_START, WIN_END = '20260228', '20260321'
Q, R = 0.001, 10.0   # slow (production)


def main():
    print(f"[1/4] Loading GDELT...")
    raw = pd.read_parquet(MAIN_PATH)
    raw['date'] = raw['SQLDATE'].astype(str).str[:8]

    filtered = clean_gdelt_data(raw)
    filtered = apply_standard_scaling(filtered, is_train=False)
    filtered['weighted_index'] = filtered['Log_Sources']

    agg = (filtered.groupby(['date', 'standard_name']).agg(
        conflict_index=('weighted_index', 'sum'),
        events=('EventCode', 'count'),
        mentions=('NumMentions', 'sum'),
        articles=('NumArticles', 'mean'),
        sources=('NumSources', 'mean'),
        avg_tone=('AvgTone', 'mean'),
    ).reset_index().rename(columns={'standard_name': 'city'}))
    agg['date'] = pd.to_datetime(agg['date'])

    print(f"      {len(agg):,} city-day rows, {agg['city'].nunique()} cities")

    print(f"[2/4] Kalman filter (Q={Q}, R={R})...")
    res = agg.groupby('city', group_keys=False).apply(
        lambda x: apply_kalman_group(x, agg['date'].min(), agg['date'].max(), Q, R))
    res = res.dropna(subset=['std_innov']).reset_index(drop=True)
    print(f"      {len(res):,} std_innov values")

    print(f"[3/4] GT positives + window filter...")
    gd = load_window_gdelt()
    positives = map_gt_to_positives(gd)
    res['gt'] = res.apply(lambda r: (r['date'], r['city']) in positives, axis=1)

    # train window 안만
    in_window = (res['date'] >= WIN_START) & (res['date'] <= WIN_END)
    res = res[in_window].reset_index(drop=True)
    print(f"      window rows: {len(res):,} | GT: {res['gt'].sum()} | peace: {(~res['gt']).sum()}")

    z_peace = res.loc[~res['gt'], 'std_innov'].to_numpy()
    z_event = res.loc[res['gt'],  'std_innov'].to_numpy()

    # === peacetime stats ===
    print("\n[4/4] Peacetime z-score 정규성 검정\n" + "─"*60)
    print(f"{'metric':22s} {'value':>14s}   {'expected if N(0,1)':>22s}")
    print("─"*60)
    print(f"{'n':22s} {len(z_peace):>14d}")
    print(f"{'mean':22s} {z_peace.mean():>14.4f}   {0:>22.4f}")
    print(f"{'std':22s} {z_peace.std():>14.4f}   {1:>22.4f}")
    print(f"{'skewness':22s} {stats.skew(z_peace):>14.4f}   {0:>22.4f}")
    print(f"{'kurtosis (excess)':22s} {stats.kurtosis(z_peace):>14.4f}   {0:>22.4f}")
    # KS test against N(0, 1)
    ks_stat, ks_p = stats.kstest(z_peace, 'norm', args=(0, 1))
    print(f"{'KS test stat':22s} {ks_stat:>14.4f}")
    print(f"{'KS test p-value':22s} {ks_p:>14.2e}   {'>0.05 = N(0,1)':>22s}")
    # AD test
    ad = stats.anderson(z_peace, dist='norm')
    print(f"{'AD stat':22s} {ad.statistic:>14.4f}")
    print(f"{'AD critical (5%)':22s} {ad.critical_values[2]:>14.4f}   {'stat<crit = N':>22s}")
    print("─"*60)

    print(f"\n[event vs peace]")
    print(f"  peacetime z   range:  {z_peace.min():.2f} … {z_peace.max():.2f}  (mean={z_peace.mean():.3f})")
    print(f"  event   z     range:  {z_event.min():.2f} … {z_event.max():.2f}  (mean={z_event.mean():.3f})")
    print(f"  > 3σ peace fraction:  {(np.abs(z_peace) > 3).mean()*100:.2f}%   (theory: 0.27%)")
    print(f"  > 3σ event fraction:  {(np.abs(z_event) > 3).mean()*100:.2f}%")

    # === plots ===
    print(f"\n[plots] saving...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) histogram + N(0,1)
    ax = axes[0]
    ax.hist(z_peace, bins=80, density=True, alpha=0.55, color='#5fdfa8', label=f'평시 (n={len(z_peace):,})', edgecolor='none')
    xs = np.linspace(-5, 5, 200)
    ax.plot(xs, stats.norm.pdf(xs, 0, 1), 'r--', lw=1.5, label='N(0, 1)')
    ax.set_xlim(-6, 6); ax.set_xlabel('innovation z'); ax.set_ylabel('density')
    ax.set_title(f'(a) 평시 innovation 분포 vs N(0,1)\nKS p={ks_p:.2e}')
    ax.legend(); ax.grid(alpha=0.3)

    # (b) QQ plot
    ax = axes[1]
    stats.probplot(z_peace, dist='norm', plot=ax)
    ax.get_lines()[0].set_markerfacecolor('#5fdfa8')
    ax.get_lines()[0].set_markeredgecolor('#5fdfa8')
    ax.get_lines()[0].set_markersize(2.5)
    ax.get_lines()[1].set_color('red')
    ax.get_lines()[1].set_linestyle('--')
    ax.set_title('(b) QQ plot — 평시 z vs N(0,1)')
    ax.grid(alpha=0.3)

    # (c) peace vs event distribution
    ax = axes[2]
    ax.hist(z_peace, bins=60, density=True, alpha=0.55, color='#5fdfa8', label=f'평시 (n={len(z_peace):,})')
    ax.hist(z_event, bins=30, density=True, alpha=0.55, color='#e25a4a', label=f'사건 (n={len(z_event):,})')
    ax.set_xlabel('innovation z'); ax.set_ylabel('density')
    ax.set_title('(c) 평시 vs 사건 z 분포')
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = OUT / 'innovation_normality.png'
    plt.savefig(fig_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"      saved: {fig_path}")

    # save raw arrays
    np.savez(OUT / 'innovation_arrays.npz', z_peace=z_peace, z_event=z_event,
             ks_stat=ks_stat, ks_p=ks_p, ad_stat=ad.statistic, ad_crit5=ad.critical_values[2])
    print(f"      saved: {OUT / 'innovation_arrays.npz'}")


if __name__ == '__main__':
    main()
