"""
Debug: 1) A tier 매칭, 2) Student-t robust weight 분포 확인.
"""
import numpy as np
import pandas as pd
from pipeline.config import KALMAN_Q, KALMAN_R, KALMAN_P0_RATIO, MIN_HISTORY, MAIN_PATH, PROJECT_ROOT, COEFS
from pipeline.preprocess import clean_gdelt_data, apply_standard_scaling
from pipeline.text_norm import _norm

GT_DIR = PROJECT_ROOT / "data" / "ground_truth"


def main():
    raw = pd.read_parquet(MAIN_PATH)
    filtered = clean_gdelt_data(raw)
    if 'date' not in filtered.columns:
        filtered['date'] = filtered['SQLDATE'].astype(str).str[:8]
    filtered = apply_standard_scaling(filtered, is_train=False)

    # A tier 산출
    by_city = (filtered.groupby('standard_name')
               .agg(active_days=('date', 'nunique'),
                    total_mentions=('NumMentions', 'sum'))
               .reset_index())
    a_tier = by_city[(by_city['active_days'] >= 10) & (by_city['total_mentions'] >= 100)]
    print(f"A tier 도시 수: {len(a_tier)}")
    print("샘플 (standard_name 형태):")
    print(a_tier.head(10)[['standard_name']].to_string(index=False))

    # GT 로드
    gt_files = sorted((GT_DIR).rglob("*.csv"))
    gt = pd.concat([pd.read_csv(f) for f in gt_files], ignore_index=True)
    gt = gt.rename(columns={'SQLDATE': 'date', 'ActionGeo_FullName': 'city'})
    print(f"\nGT 총 관측: {len(gt)}, unique 도시: {gt['city'].nunique()}")
    print("GT 도시 샘플:")
    print(gt['city'].value_counts().head(15))

    # 매칭 점검 — country suffix 떼고 정규화
    a_norms = set()
    for sn in a_tier['standard_name']:
        base = str(sn).split('|')[0]
        a_norms.add(_norm(base))
    print(f"\nA tier _norm set 크기: {len(a_norms)}, 예: {list(a_norms)[:8]}")

    # GT 도시 매칭 시도
    matched_a, matched_c = 0, 0
    for c, n in gt['city'].value_counts().items():
        cn = _norm(str(c))
        if any(cn == an or cn in an or an in cn for an in a_norms if an):
            matched_a += n
        else:
            matched_c += n
    print(f"\nGT 관측 → A tier {matched_a}, C tier {matched_c}")

    # Student-t weight 분포 확인 (Tehran 시계열)
    print("\n[Student-t weight 분포 디버그] Tehran")
    key_to_col = {'w_log_sources': 'Log_Sources'}
    filtered['weighted_index'] = 0.0
    for k, w in COEFS.items():
        filtered['weighted_index'] += w * filtered[key_to_col[k]]
    agg = (filtered.groupby(['date', 'standard_name'])
           .agg(conflict_index=('weighted_index', 'sum')).reset_index()
           .rename(columns={'standard_name': 'city'}))
    agg['date'] = pd.to_datetime(agg['date'])
    cities_sample = ['Tehran|IR', 'Beirut|LE', 'Khan Yunis|GZ', 'Najaf|IZ', 'Sanaa|YM']
    for cname in cities_sample:
        ts = agg[agg['city'] == cname].sort_values('date')['conflict_index'].values
        if len(ts) < 30:
            continue
        # 간단 시뮬레이션
        n = len(ts)
        x = np.zeros(n); P = np.zeros(n)
        x[0] = ts[0]; P[0] = KALMAN_R * KALMAN_P0_RATIO
        weights_dof4 = []
        weights_dof2_5 = []
        m2_list = []
        for t in range(1, n):
            xpred = x[t-1]; Ppred = P[t-1] + KALMAN_Q
            innov = ts[t] - xpred
            S = max(Ppred + KALMAN_R, 1e-10)
            m2 = innov*innov/S
            m2_list.append(m2)
            for dof, store in [(4.0, weights_dof4), (2.5, weights_dof2_5)]:
                w = (dof + 1.0) / (dof + m2)
                store.append(w)
            # gaussian update for trajectory
            K = Ppred / S
            x[t] = xpred + K*innov
            P[t] = (1-K)*Ppred
        print(f"  {cname:<18} signal range [{ts.min():.1f}, {ts.max():.1f}] mean {ts.mean():.1f}")
        print(f"      m² distribution: median {np.median(m2_list):.3f}, p95 {np.percentile(m2_list,95):.2f}, max {np.max(m2_list):.2f}")
        print(f"      w(dof=4): median {np.median(weights_dof4):.3f}, min {np.min(weights_dof4):.3f}")
        print(f"      w(dof=2.5): median {np.median(weights_dof2_5):.3f}, min {np.min(weights_dof2_5):.3f}")


if __name__ == '__main__':
    main()
