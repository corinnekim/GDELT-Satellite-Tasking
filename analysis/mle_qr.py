"""우선순위 4: Q, R의 MLE 검증.

Kalman log-likelihood:
  ℓ(Q,R) = Σ_t -½ [log(2π S_t) + ε_t² / S_t]
  where  S_t = P_pred,t + R,  ε_t = y_t - x_pred,t

도시별로 Kalman recursion 돌리며 likelihood 누적.
Q, R 격자 위에서 MLE 추정 → grid search 결과(slow=0.001, 10)와 비교.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pipeline.config import MAIN_PATH
from pipeline.preprocess import clean_gdelt_data, apply_standard_scaling

OUT = Path('/Users/corinnekim/gdelt/analysis/results')


def kalman_loglik_per_series(signal: np.ndarray, Q: float, R: float, P0_ratio: float = 2.0) -> float:
    n = len(signal)
    if n < 2:
        return 0.0
    x = signal[0]
    P = R * P0_ratio
    ll = 0.0
    for t in range(1, n):
        x_pred = x
        P_pred = P + Q
        eps = signal[t] - x_pred
        S = P_pred + R
        if S <= 0 or not np.isfinite(S):
            return -np.inf
        ll += -0.5 * (np.log(2 * np.pi * S) + eps * eps / S)
        K = P_pred / S
        x = x_pred + K * eps
        P = (1 - K) * P_pred
    return ll


def total_loglik(signals_list, Q, R):
    return sum(kalman_loglik_per_series(s, Q, R) for s in signals_list)


def main():
    print("[1/3] Loading + aggregate (city-day signals)...")
    raw = pd.read_parquet(MAIN_PATH)
    raw['date'] = raw['SQLDATE'].astype(str).str[:8]
    f = clean_gdelt_data(raw)
    f = apply_standard_scaling(f, is_train=False)
    f['weighted_index'] = f['Log_Sources']
    agg = (f.groupby(['date','standard_name'])
           .agg(conflict_index=('weighted_index','sum'))
           .reset_index().rename(columns={'standard_name':'city'}))
    agg['date'] = pd.to_datetime(agg['date'])

    all_dates = pd.date_range(agg['date'].min(), agg['date'].max(), freq='D')
    cities = sorted(agg['city'].unique())
    signals = []
    for city in cities:
        s = (agg[agg['city']==city].set_index('date')['conflict_index']
             .reindex(all_dates).fillna(0).values.astype(np.float64))
        if len(s) >= 2:
            signals.append(s)
    print(f"      {len(signals):,} city series, mean length {np.mean([len(s) for s in signals]):.1f}")

    # log-grid
    Q_grid = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1e0])
    R_grid = np.array([0.5, 1, 2, 5, 10, 20, 50, 100])
    print(f"[2/3] Grid eval ({len(Q_grid)} × {len(R_grid)} = {len(Q_grid)*len(R_grid)} pts)...")
    LL = np.full((len(Q_grid), len(R_grid)), -np.inf)
    for i, Q in enumerate(Q_grid):
        for j, R in enumerate(R_grid):
            LL[i, j] = total_loglik(signals, Q, R)
        print(f"      Q={Q:8.4f}  best R={R_grid[np.argmax(LL[i])]:7.2f}  ℓ={LL[i].max():12.1f}")

    # argmax
    flat_idx = np.unravel_index(LL.argmax(), LL.shape)
    Q_mle, R_mle = Q_grid[flat_idx[0]], R_grid[flat_idx[1]]
    ll_mle = LL[flat_idx]
    # production grid: slow / fast
    slow_idx_q = np.argmin(np.abs(Q_grid - 0.001));  slow_idx_r = np.argmin(np.abs(R_grid - 10.0))
    fast_idx_q = np.argmin(np.abs(Q_grid - 0.1));    fast_idx_r = np.argmin(np.abs(R_grid - 1.0))
    ll_slow = LL[slow_idx_q, slow_idx_r]
    ll_fast = LL[fast_idx_q, fast_idx_r]

    print("\n[3/3] MLE 결과")
    print("─"*60)
    print(f"{'config':25s} {'Q':>8s} {'R':>8s} {'log-lik':>14s}")
    print("─"*60)
    print(f"{'MLE optimum':25s} {Q_mle:>8.4f} {R_mle:>8.2f} {ll_mle:>14.1f}")
    print(f"{'production slow':25s} {0.001:>8.4f} {10.0:>8.2f} {ll_slow:>14.1f}  Δ={ll_slow-ll_mle:+.1f}")
    print(f"{'production fast':25s} {0.1:>8.4f} {1.0:>8.2f} {ll_fast:>14.1f}  Δ={ll_fast-ll_mle:+.1f}")
    print("─"*60)

    # plot likelihood heatmap
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
    im = ax.imshow(LL, aspect='auto', origin='lower', cmap='magma',
                   extent=[np.log10(R_grid[0]), np.log10(R_grid[-1]),
                           np.log10(Q_grid[0]), np.log10(Q_grid[-1])])
    cbar = plt.colorbar(im, ax=ax); cbar.set_label('log-likelihood')
    # mark points
    ax.scatter(np.log10(R_mle), np.log10(Q_mle), color='cyan', marker='*', s=300, edgecolors='white',
               linewidth=1.5, label=f'MLE (Q={Q_mle:.4f}, R={R_mle:.1f})', zorder=5)
    ax.scatter(np.log10(10.0), np.log10(0.001), color='#5fdfa8', marker='o', s=180, edgecolors='white',
               linewidth=1.5, label='slow (Q=0.001, R=10)', zorder=5)
    ax.scatter(np.log10(1.0), np.log10(0.1), color='#c4702c', marker='s', s=160, edgecolors='white',
               linewidth=1.5, label='fast (Q=0.1, R=1)', zorder=5)
    ax.set_xlabel('log10(R)'); ax.set_ylabel('log10(Q)')
    ax.set_title('Kalman log-likelihood surface  (sum over all cities)')
    ax.legend(loc='lower right')
    fig_path = OUT / 'qr_mle.png'
    plt.savefig(fig_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"saved: {fig_path}")

    np.savez(OUT / 'mle_arrays.npz', Q_grid=Q_grid, R_grid=R_grid, LL=LL,
             Q_mle=Q_mle, R_mle=R_mle, ll_mle=ll_mle, ll_slow=ll_slow, ll_fast=ll_fast)


if __name__ == '__main__':
    main()
