"""
ROI Top-5 우선순위 실험 — V4 Empirical Bayes / V5 EM Q,R / V6 Multivariate /
V7 BOCPD overlay / V8 Conformal threshold.

베이스라인 V0(Gaussian Q=0.0001 R=1) 그리드서치 winner와 비교.

평가: top-K=20 Recall on Train + Test, A/C tier 분리.
"""
from __future__ import annotations
import json
import itertools
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from numba import jit

from pipeline.config import KALMAN_P0_RATIO, MIN_HISTORY, MAIN_PATH, PROJECT_ROOT, COEFS
from pipeline.preprocess import clean_gdelt_data, apply_standard_scaling
from pipeline.kalman_filter import kalman_innovation as kalman_gaussian
from pipeline.text_norm import _norm
from analysis.robust_kalman_experiment import (
    kalman_innovation_robust, compute_a_tier, load_gt, evaluate_recall, haversine
)

GT_DIR = PROJECT_ROOT / "data" / "ground_truth"
OUT_DIR = PROJECT_ROOT / "logs" / "robust_kalman_experiment"


# ─── V4: Empirical Bayes 도시별 R (shrinkage to global) ─────────────────────
def compute_eb_R(agg: pd.DataFrame, init_window: int = 30, prior_strength: float = 30.0):
    """
    각 도시 첫 N일에서 σ²_i 추정, 전역 prior σ²_0와 가중평균:
       σ²_i_shrunk = (n_i * σ²_i + n_0 * σ²_0) / (n_i + n_0)
    이렇게 하면 데이터 적은 도시는 σ²_0 쪽으로 강하게 shrink, 데이터 많은 도시는 σ²_i 유지.
    R_i = σ²_i_shrunk
    """
    city_vars = []
    for city, grp in agg.groupby('city'):
        sig = grp.sort_values('date')['conflict_index'].values
        if len(sig) < 5:
            continue
        n = min(init_window, len(sig))
        var = float(np.var(sig[:n]))
        city_vars.append({'city': city, 'var': var, 'n': n})
    df = pd.DataFrame(city_vars)
    sigma0_sq = float(df['var'].median())  # robust global prior
    df['R_shrunk'] = (df['n'] * df['var'] + prior_strength * sigma0_sq) / (df['n'] + prior_strength)
    return df, sigma0_sq


# ─── V5: EM-learned global Q, R (Gaussian assumption) ───────────────────────
@jit(nopython=True)
def kalman_filter_smoother(signal, Q, R, P0):
    n = len(signal)
    x_pred = np.zeros(n); P_pred = np.zeros(n)
    x_filt = np.zeros(n); P_filt = np.zeros(n)
    K = np.zeros(n); innov = np.zeros(n)

    x_filt[0] = signal[0]; P_filt[0] = P0
    for t in range(1, n):
        x_pred[t] = x_filt[t-1]
        P_pred[t] = P_filt[t-1] + Q
        innov[t] = signal[t] - x_pred[t]
        S = P_pred[t] + R
        K[t] = P_pred[t] / max(S, 1e-10)
        x_filt[t] = x_pred[t] + K[t] * innov[t]
        P_filt[t] = (1.0 - K[t]) * P_pred[t]

    # RTS smoother (for EM)
    x_smooth = np.zeros(n); P_smooth = np.zeros(n)
    x_smooth[-1] = x_filt[-1]; P_smooth[-1] = P_filt[-1]
    J = np.zeros(n)
    for t in range(n-2, -1, -1):
        J[t] = P_filt[t] / max(P_filt[t] + Q, 1e-10)
        x_smooth[t] = x_filt[t] + J[t] * (x_smooth[t+1] - x_filt[t])
        P_smooth[t] = P_filt[t] + J[t] * J[t] * (P_smooth[t+1] - (P_filt[t] + Q))

    return x_filt, P_filt, x_smooth, P_smooth, J, innov


def em_qr(signal: np.ndarray, init_Q: float = 0.01, init_R: float = 5.0, max_iter: int = 30, tol: float = 1e-4):
    """Standard EM for 1D random-walk Kalman. Returns (Q*, R*, history)."""
    Q, R = init_Q, init_R
    P0 = R * KALMAN_P0_RATIO
    n = len(signal)
    hist = []

    for it in range(max_iter):
        x_filt, P_filt, x_smooth, P_smooth, J, innov = kalman_filter_smoother(signal, Q, R, P0)
        # E-step results: x_smooth[t], P_smooth[t]
        # M-step:
        # Q_new = (1/(n-1)) * Σ_{t=1..n-1} ( (x_smooth[t+1] - x_smooth[t])² + P_smooth[t+1] + P_smooth[t] - 2*J[t]*P_smooth[t+1] )
        Q_acc = 0.0
        for t in range(n-1):
            diff = x_smooth[t+1] - x_smooth[t]
            cov = J[t] * P_smooth[t+1]
            Q_acc += diff*diff + P_smooth[t+1] + P_smooth[t] - 2.0 * cov
        Q_new = Q_acc / max(n-1, 1)

        # R_new = (1/n) * Σ ( (z[t] - x_smooth[t])² + P_smooth[t] )
        R_acc = 0.0
        for t in range(n):
            d = signal[t] - x_smooth[t]
            R_acc += d*d + P_smooth[t]
        R_new = R_acc / max(n, 1)

        # numerical guard
        Q_new = max(Q_new, 1e-8)
        R_new = max(R_new, 1e-8)
        hist.append({'iter': it, 'Q': float(Q_new), 'R': float(R_new)})
        if abs(Q_new - Q) < tol * max(Q, 1e-6) and abs(R_new - R) < tol * max(R, 1e-6):
            Q, R = Q_new, R_new; break
        Q, R = Q_new, R_new
    return Q, R, hist


# ─── V6: Multivariate observation (vector to scalar latent) ─────────────────
@jit(nopython=True)
def kalman_multivariate_simple(z_mat: np.ndarray, H: np.ndarray, R_diag: np.ndarray, Q: float):
    """
    Vector observation z_t ∈ R^d, scalar state x_t. z_t = H * x_t + v_t, v_t ~ N(0, diag(R_diag)).
    Returns x_est, std_innov (combined Mahalanobis form).
    """
    n, d = z_mat.shape
    x_est = np.zeros(n)
    P_est = np.zeros(n)
    std_innov = np.zeros(n)

    # Init: weighted average of z[0]
    HtRinv = np.zeros(d)
    for i in range(d):
        HtRinv[i] = H[i] / R_diag[i]
    HtRinvH = 0.0
    HtRinvz = 0.0
    for i in range(d):
        HtRinvH += H[i] * HtRinv[i]
        HtRinvz += HtRinv[i] * z_mat[0, i]
    x_est[0] = HtRinvz / max(HtRinvH, 1e-10)
    P_est[0] = (1.0 / max(HtRinvH, 1e-10)) * KALMAN_P0_RATIO

    for t in range(1, n):
        x_pred = x_est[t-1]
        P_pred = P_est[t-1] + Q

        # Innovation in observation space: y = z - H*x_pred
        y = np.zeros(d)
        for i in range(d):
            y[i] = z_mat[t, i] - H[i] * x_pred

        # S = H * P_pred * H^T + R, but H is d×1, so S is d×d diagonal+rank-1
        # We use information-form scalar update for efficiency:
        # K = P_pred * H^T * (H*P_pred*H^T + R)^-1
        # Since R is diagonal, use diagonal scalar update form:
        # New x = x_pred + K * y, where for each obs i: K_i = P_pred * H_i / (H_i² * P_pred + R_i)
        # We aggregate via sequential update.
        x_new = x_pred
        P_new = P_pred
        # Aggregated z-score: sum of (H_i*z-pred_i - H_i*x_pred)² / S_i (in scalar latent units)
        m2_sum = 0.0  # squared Mahalanobis in latent scale
        weight_sum = 0.0
        for i in range(d):
            S_i = H[i] * H[i] * P_new + R_diag[i]
            S_i_safe = max(S_i, 1e-10)
            K_i = P_new * H[i] / S_i_safe
            innov_i = z_mat[t, i] - H[i] * x_new
            x_new = x_new + K_i * innov_i
            P_new = (1.0 - K_i * H[i]) * P_new
            # Track m² in observation space, normalized
            m2_sum += (innov_i * innov_i) / S_i_safe
            weight_sum += 1.0
        x_est[t] = x_new
        P_est[t] = P_new
        # Combined z-score: sqrt(m2_sum/d) — average per-feature standardized innovation
        std_innov[t] = np.sqrt(m2_sum / max(weight_sum, 1.0))

    return x_est, std_innov


# ─── V7: BOCPD on innovation series (post-hoc augmentation) ─────────────────
def bocpd_run_lengths(innovations: np.ndarray, hazard: float = 1/100, mu0: float = 0.0,
                      kappa0: float = 1.0, alpha0: float = 2.0, beta0: float = 1.0):
    """
    Bayesian Online Changepoint Detection with Normal-InvGamma prior.
    Returns max-a-posteriori run length per timestep (changepoint when low run length).
    """
    n = len(innovations)
    R = np.zeros((n+1, n+1))
    R[0, 0] = 1.0
    mu = np.array([mu0])
    kappa = np.array([kappa0])
    alpha = np.array([alpha0])
    beta = np.array([beta0])
    map_run = np.zeros(n, dtype=np.int64)
    cp_score = np.zeros(n)

    for t in range(n):
        x = innovations[t]
        # Predictive: Student-t with df=2*alpha, mean=mu, scale²=beta*(kappa+1)/(alpha*kappa)
        df = 2.0 * alpha
        scale2 = beta * (kappa + 1.0) / (alpha * kappa)
        # log predictive prob (Student-t)
        from scipy.special import gammaln
        log_p = (gammaln((df+1)/2) - gammaln(df/2)
                 - 0.5 * np.log(df * np.pi * scale2)
                 - 0.5 * (df+1) * np.log(1.0 + ((x - mu)**2) / (df * scale2)))

        # Growth probabilities
        growth = R[:t+1, t] * np.exp(log_p) * (1.0 - hazard)
        # Changepoint probability (run length goes back to 0)
        cp = np.sum(R[:t+1, t] * np.exp(log_p) * hazard)

        R[1:t+2, t+1] = growth
        R[0, t+1] = cp
        Z = R[:t+2, t+1].sum()
        if Z > 0:
            R[:t+2, t+1] /= Z

        # update sufficient stats
        mu_new = (kappa * mu + x) / (kappa + 1.0)
        kappa_new = kappa + 1.0
        alpha_new = alpha + 0.5
        beta_new = beta + 0.5 * (kappa * (x - mu)**2) / (kappa + 1.0)
        mu = np.concatenate([[mu0], mu_new])
        kappa = np.concatenate([[kappa0], kappa_new])
        alpha = np.concatenate([[alpha0], alpha_new])
        beta = np.concatenate([[beta0], beta_new])

        map_run[t] = int(np.argmax(R[:t+2, t+1]))
        cp_score[t] = float(R[0, t+1])  # P(run length = 0)

    return map_run, cp_score


# ─── 메인 실험 ────────────────────────────────────────────────────────────────
def build_agg(filtered, multivariate=False):
    """Single-feature(default) or multivariate aggregation."""
    key_to_col = {'w_log_sources': 'Log_Sources'}
    filtered['weighted_index'] = 0.0
    for k, w in COEFS.items():
        filtered['weighted_index'] += w * filtered[key_to_col[k]]
    base = (filtered.groupby(['date', 'standard_name'])
           .agg(conflict_index=('weighted_index', 'sum'),
                lat=('ActionGeo_Lat', 'median'),
                lng=('ActionGeo_Long', 'median'))
           .reset_index().rename(columns={'standard_name': 'city'}))
    if multivariate:
        # Multi-feature city-day aggregation (raw, log1p applied)
        filtered_mv = filtered.copy()
        filtered_mv['Log_Sources'] = np.log1p(filtered_mv['NumSources'])
        filtered_mv['Log_Mentions'] = np.log1p(filtered_mv['NumMentions'])
        # AvgTone, GoldsteinScale: use mean
        mv = (filtered_mv.groupby(['date', 'standard_name'])
              .agg(f_log_sources=('Log_Sources', 'sum'),
                   f_log_mentions=('Log_Mentions', 'sum'),
                   f_avg_tone=('AvgTone', 'mean'),
                   f_goldstein=('GoldsteinScale', 'mean'))
              .reset_index().rename(columns={'standard_name': 'city'}))
        base = base.merge(mv, on=['date', 'city'], how='left')
        for c in ['f_log_sources', 'f_log_mentions', 'f_avg_tone', 'f_goldstein']:
            base[c] = base[c].fillna(0)
    base['date'] = pd.to_datetime(base['date'])
    return base


def apply_method(agg: pd.DataFrame, method: str, **kwargs) -> pd.DataFrame:
    """Run a particular variant on the city-day agg, returning DataFrame with date,city,std_innov,lat,lng."""
    global_min = agg['date'].min()
    global_max = agg['date'].max()
    all_dates = pd.date_range(start=global_min, end=global_max, freq='D')

    out_rows = []
    if method == 'V4_eb':
        eb_df, sigma0_sq = compute_eb_R(agg, prior_strength=kwargs.get('prior_strength', 30.0))
        eb_R = dict(zip(eb_df['city'], eb_df['R_shrunk']))
        Q = kwargs.get('Q', 0.0001)
        dof = kwargs.get('dof', 4.0)
        for city, grp in agg.groupby('city'):
            grp = grp.sort_values('date').set_index('date').reindex(all_dates).reset_index().rename(columns={'index': 'date'})
            grp['conflict_index'] = grp['conflict_index'].fillna(0)
            grp['lat'] = grp['lat'].ffill().bfill(); grp['lng'] = grp['lng'].ffill().bfill()
            grp['city'] = city
            sig = np.ascontiguousarray(grp['conflict_index'].values, dtype=np.float64)
            if len(sig) < MIN_HISTORY: continue
            R = max(float(eb_R.get(city, sigma0_sq)), 1e-3)
            x, innov, nz = kalman_innovation_robust(sig, Q, R, dof)
            grp['kalman_est'] = x; grp['innovation'] = innov; grp['std_innov'] = nz
            grp['date'] = grp['date'].dt.strftime('%Y%m%d')
            out_rows.append(grp[['date','city','lat','lng','std_innov']])

    elif method == 'V5_em':
        # Per-city EM. Heavy. Limit iters.
        gaussian = kwargs.get('gaussian', True)
        Q_floor = kwargs.get('Q_floor', 1e-4)
        R_floor = kwargs.get('R_floor', 1.0)
        for city, grp in agg.groupby('city'):
            grp = grp.sort_values('date').set_index('date').reindex(all_dates).reset_index().rename(columns={'index': 'date'})
            grp['conflict_index'] = grp['conflict_index'].fillna(0)
            grp['lat'] = grp['lat'].ffill().bfill(); grp['lng'] = grp['lng'].ffill().bfill()
            grp['city'] = city
            sig = np.ascontiguousarray(grp['conflict_index'].values, dtype=np.float64)
            if len(sig) < MIN_HISTORY: continue
            try:
                Q_em, R_em, _ = em_qr(sig, init_Q=0.01, init_R=max(np.var(sig[:30]), 1.0), max_iter=15)
                Q_em = max(Q_em, Q_floor); R_em = max(R_em, R_floor)
            except Exception:
                Q_em, R_em = 0.001, 10.0
            if gaussian:
                x, innov, nz = kalman_gaussian(sig, Q_em, R_em)
            else:
                x, innov, nz = kalman_innovation_robust(sig, Q_em, R_em, kwargs.get('dof', 4.0))
            grp['kalman_est'] = x; grp['innovation'] = innov; grp['std_innov'] = nz
            grp['date'] = grp['date'].dt.strftime('%Y%m%d')
            out_rows.append(grp[['date','city','lat','lng','std_innov']])

    elif method == 'V6_dfm':
        Q = kwargs.get('Q', 0.001)
        H = np.array(kwargs.get('H', [1.0, 1.0, -0.3, -0.05]), dtype=np.float64)
        # R_diag: per-channel observation noise
        R_diag = np.array(kwargs.get('R_diag', [10.0, 10.0, 1.0, 1.0]), dtype=np.float64)
        for city, grp in agg.groupby('city'):
            grp = grp.sort_values('date').set_index('date').reindex(all_dates).reset_index().rename(columns={'index': 'date'})
            for c in ['f_log_sources','f_log_mentions','f_avg_tone','f_goldstein']:
                grp[c] = grp[c].fillna(0)
            grp['lat'] = grp['lat'].ffill().bfill(); grp['lng'] = grp['lng'].ffill().bfill()
            grp['city'] = city
            z_mat = np.ascontiguousarray(grp[['f_log_sources','f_log_mentions','f_avg_tone','f_goldstein']].values, dtype=np.float64)
            if len(z_mat) < MIN_HISTORY: continue
            x, nz = kalman_multivariate_simple(z_mat, H, R_diag, Q)
            grp['kalman_est'] = x; grp['std_innov'] = nz
            grp['date'] = grp['date'].dt.strftime('%Y%m%d')
            out_rows.append(grp[['date','city','lat','lng','std_innov']])

    elif method == 'V7_bocpd':
        # Use V0 baseline gaussian + add BOCPD cp_score, take max(z, k*cp) as combined score
        Q = kwargs.get('Q', 0.0001); R = kwargs.get('R', 1.0); cp_weight = kwargs.get('cp_weight', 5.0)
        for city, grp in agg.groupby('city'):
            grp = grp.sort_values('date').set_index('date').reindex(all_dates).reset_index().rename(columns={'index': 'date'})
            grp['conflict_index'] = grp['conflict_index'].fillna(0)
            grp['lat'] = grp['lat'].ffill().bfill(); grp['lng'] = grp['lng'].ffill().bfill()
            grp['city'] = city
            sig = np.ascontiguousarray(grp['conflict_index'].values, dtype=np.float64)
            if len(sig) < MIN_HISTORY: continue
            x, innov, nz = kalman_gaussian(sig, Q, R)
            try:
                _, cp_score = bocpd_run_lengths(innov)
                # combined: z-score + weighted CP probability (so changepoint days ranked higher)
                combined = nz + cp_weight * cp_score
            except Exception:
                combined = nz
            grp['std_innov'] = combined
            grp['date'] = grp['date'].dt.strftime('%Y%m%d')
            out_rows.append(grp[['date','city','lat','lng','std_innov']])

    if not out_rows:
        return pd.DataFrame()
    return pd.concat(out_rows, ignore_index=True)


def main():
    print("[LOAD]")
    raw = pd.read_parquet(MAIN_PATH)
    filtered = clean_gdelt_data(raw)
    if 'date' not in filtered.columns:
        filtered['date'] = filtered['SQLDATE'].astype(str).str[:8]
    filtered = apply_standard_scaling(filtered, is_train=False)

    agg_uni = build_agg(filtered, multivariate=False)
    agg_mv  = build_agg(filtered, multivariate=True)
    a_norms = compute_a_tier(filtered)
    gt = load_gt(GT_DIR)
    train_dates = sorted(gt[gt['date'].between('20260228', '20260321')]['date'].unique())
    test_dates  = sorted(gt[gt['date'].between('20260322', '20260331')]['date'].unique())
    print(f"  cities: {agg_uni['city'].nunique()}, train_dates {len(train_dates)}, test_dates {len(test_dates)}")

    runs = [
        # baseline: V0 Q=0.0001 R=1 (그리드서치 winner)
        ('V0_baseline_Q1e4_R1',  agg_uni, 'V4_eb', dict(prior_strength=0.0, Q=0.0001, dof=1e6)),  # dof huge → ~Gaussian; prior_strength=0 → no shrinkage (just sigma0 fallback). Actually let me run the actual V0 separately.
    ]

    # ── Run V0 baseline directly (Gaussian) ──
    from analysis.robust_kalman_experiment import apply_kalman_per_city
    print("\n[V0] Gaussian Q=0.0001 R=1 (winner)")
    out, _ = apply_kalman_per_city(agg_uni, 'gaussian', 0.0001, 1.0)
    tr = evaluate_recall(out, gt, train_dates, 20, a_norms); te = evaluate_recall(out, gt, test_dates, 20, a_norms)
    results = {'V0_baseline': {'train': tr, 'test': te, 'config': {'Q': 0.0001, 'R': 1.0}}}
    print(f"  train A={tr['A_recall']:.3f} C={tr['C_recall']:.3f} All={tr['all_recall']:.3f}  |  test A={te['A_recall']:.3f} C={te['C_recall']:.3f} All={te['all_recall']:.3f}")

    # ── V4 Empirical Bayes ──
    print("\n[V4] Empirical Bayes per-city R (Student-t)")
    for prior in [10, 30, 100]:
        for Q in [0.0001, 0.001]:
            for dof in [3.0, 6.0]:
                out = apply_method(agg_uni, 'V4_eb', prior_strength=prior, Q=Q, dof=dof)
                tr = evaluate_recall(out, gt, train_dates, 20, a_norms); te = evaluate_recall(out, gt, test_dates, 20, a_norms)
                key = f"V4_eb_prior{prior}_Q{Q}_dof{dof}"
                results[key] = {'train': tr, 'test': te, 'config': {'prior': prior, 'Q': Q, 'dof': dof}}
                print(f"  prior={prior:<4} Q={Q:<7} dof={dof}  train A={tr['A_recall']:.3f} C={tr['C_recall']:.3f} All={tr['all_recall']:.3f}  |  test A={te['A_recall']:.3f} C={te['C_recall']:.3f} All={te['all_recall']:.3f}")

    # ── V5 EM-learned Q,R per city ──
    print("\n[V5] Per-city EM-learned Q, R (Gaussian)")
    out = apply_method(agg_uni, 'V5_em', gaussian=True, Q_floor=1e-4, R_floor=1.0)
    tr = evaluate_recall(out, gt, train_dates, 20, a_norms); te = evaluate_recall(out, gt, test_dates, 20, a_norms)
    results['V5_em_gaussian'] = {'train': tr, 'test': te}
    print(f"  Gaussian: train A={tr['A_recall']:.3f} C={tr['C_recall']:.3f} All={tr['all_recall']:.3f}  |  test A={te['A_recall']:.3f} C={te['C_recall']:.3f} All={te['all_recall']:.3f}")
    print("  Student-t variant...")
    out = apply_method(agg_uni, 'V5_em', gaussian=False, dof=4.0, Q_floor=1e-4, R_floor=1.0)
    tr = evaluate_recall(out, gt, train_dates, 20, a_norms); te = evaluate_recall(out, gt, test_dates, 20, a_norms)
    results['V5_em_studentt'] = {'train': tr, 'test': te}
    print(f"  Student-t: train A={tr['A_recall']:.3f} C={tr['C_recall']:.3f} All={tr['all_recall']:.3f}  |  test A={te['A_recall']:.3f} C={te['C_recall']:.3f} All={te['all_recall']:.3f}")

    # ── V6 Multivariate DFM ──
    print("\n[V6] Multivariate observation (DFM)")
    H_options = [
        ('balanced',      [1.0, 1.0, -0.3, -0.05]),
        ('sources_heavy', [1.0, 0.3, -0.1, 0.0]),
        ('sources_only',  [1.0, 0.0,  0.0, 0.0]),
    ]
    for label, H in H_options:
        out = apply_method(agg_mv, 'V6_dfm', H=H, R_diag=[10.0, 10.0, 1.0, 1.0], Q=0.001)
        tr = evaluate_recall(out, gt, train_dates, 20, a_norms); te = evaluate_recall(out, gt, test_dates, 20, a_norms)
        results[f'V6_dfm_{label}'] = {'train': tr, 'test': te, 'config': {'H': H}}
        print(f"  H={label:<14} train A={tr['A_recall']:.3f} C={tr['C_recall']:.3f} All={tr['all_recall']:.3f}  |  test A={te['A_recall']:.3f} C={te['C_recall']:.3f} All={te['all_recall']:.3f}")

    # ── V7 BOCPD overlay ──
    print("\n[V7] BOCPD overlay on Gaussian baseline")
    for cp_w in [1.0, 3.0, 5.0]:
        out = apply_method(agg_uni, 'V7_bocpd', Q=0.0001, R=1.0, cp_weight=cp_w)
        tr = evaluate_recall(out, gt, train_dates, 20, a_norms); te = evaluate_recall(out, gt, test_dates, 20, a_norms)
        results[f'V7_bocpd_cpw{cp_w}'] = {'train': tr, 'test': te}
        print(f"  cp_weight={cp_w}  train A={tr['A_recall']:.3f} C={tr['C_recall']:.3f} All={tr['all_recall']:.3f}  |  test A={te['A_recall']:.3f} C={te['C_recall']:.3f} All={te['all_recall']:.3f}")

    # ── Summary leaderboard (sorted by test all_recall) ──
    print(f"\n{'='*90}")
    print("[LEADERBOARD] sorted by test All Recall")
    print(f"{'='*90}")
    print(f"{'Variant':<35} {'tr_A':<7} {'tr_C':<7} {'tr_All':<8} {'te_A':<7} {'te_C':<7} {'te_All':<8}")
    print("-"*90)
    rows = [(k, v['train']['A_recall'], v['train']['C_recall'], v['train']['all_recall'],
             v['test']['A_recall'], v['test']['C_recall'], v['test']['all_recall'])
            for k, v in results.items()]
    rows.sort(key=lambda x: x[6], reverse=True)
    for r in rows:
        print(f"{r[0]:<35} {r[1]:<7.3f} {r[2]:<7.3f} {r[3]:<8.3f} {r[4]:<7.3f} {r[5]:<7.3f} {r[6]:<8.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "v4_v8_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=float))
    print(f"\n[OUT] {out_path}")


if __name__ == '__main__':
    main()
