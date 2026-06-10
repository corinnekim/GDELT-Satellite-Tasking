"""
A/C/D tier threshold analysis over the 22-day train window (2026-02-28 to 2026-03-21).

Baseline per standard_name:
  - days_present   : number of distinct dates with ≥1 GDELT row
  - sum_mentions   : total NumMentions over window
  - avg_mentions_per_day_active  : mean per active day (excludes zero days)
  - n_events       : total rows

A/C split logic ('major vs minor'):
  - A (major/active) : frequently and widely reported
  - C (minor)        : rarely/sparsely reported (rare-spike candidates)
  - D                : GT city not present in GDELT window → transparency (excluded from recall)

GT mapping:
  - Each GT entry's (city, lat, lon) is mapped to a standard_name via:
    (a) name-exact match on _norm() against GDELT {FullName, standard_name}
    (b) geo: dominant standard_name within 30km (≥60% of mentions)
    (c) else D-tier
"""
from pathlib import Path
import numpy as np
import pandas as pd

from pipeline.config import CONFIRMED_CODES, MONITORED_COUNTRIES, CITY_BLACKLIST
from pipeline.preprocess import apply_name_map
from pipeline.text_norm import _norm

ROOT = Path(__file__).resolve().parent.parent
GDELT = ROOT / "data" / "gdelt_main_2026.parquet"
GT_DIR = ROOT / "data" / "ground_truth" / "train"
OUT_CITY = ROOT / "data" / "review" / "ac_city_stats.csv"
OUT_GT = ROOT / "data" / "review" / "ac_gt_report.csv"

WIN_START = "20260228"
WIN_END = "20260321"
RADIUS_KM = 30.0
GEO_DOM = 0.60


def haversine(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = (0.5 - np.cos((lat2 - lat1) * p) / 2
         + np.cos(lat1 * p) * np.cos(lat2 * p)
         * (1 - np.cos((lon2 - lon1) * p)) / 2)
    return 12742 * np.arcsin(np.sqrt(a))


def load_window_gdelt():
    raw = pd.read_parquet(GDELT)
    raw = raw[raw.EventCode.astype(int).isin(CONFIRMED_CODES)]
    raw = raw[raw.ActionGeo_CountryCode.isin(MONITORED_COUNTRIES)]
    raw = raw[raw.ActionGeo_Type == 4]
    raw = raw.dropna(subset=['ActionGeo_FullName', 'ActionGeo_Lat', 'ActionGeo_Long'])
    raw['date'] = raw['SQLDATE'].astype(str).str[:8]
    raw = raw[(raw.date >= WIN_START) & (raw.date <= WIN_END)]
    raw = raw[~raw['ActionGeo_FullName'].isin(CITY_BLACKLIST)]
    raw = apply_name_map(raw)
    raw = raw[~raw['standard_name'].isin(CITY_BLACKLIST)]
    return raw


def per_city_stats(gdelt):
    stats = (gdelt.groupby('standard_name')
                  .agg(days_present=('date', 'nunique'),
                       sum_mentions=('NumMentions', 'sum'),
                       n_events=('GLOBALEVENTID', 'size'),
                       lat=('ActionGeo_Lat', 'median'),
                       lon=('ActionGeo_Long', 'median'))
                  .reset_index())
    stats['avg_mentions_active'] = (stats.sum_mentions / stats.days_present).round(1)
    stats['avg_events_active'] = (stats.n_events / stats.days_present).round(2)
    return stats.sort_values('sum_mentions', ascending=False)


def map_gt_to_standard(gdelt):
    """Returns: list of dicts per GT entry with standard_name (or None for D)."""
    # Index: norm key (both FullName and standard_name are candidates)
    name_to_std = {}
    for _, r in gdelt[['ActionGeo_FullName', 'standard_name']].drop_duplicates().iterrows():
        name_to_std[_norm(r['ActionGeo_FullName'])] = r['standard_name']
        name_to_std[_norm(r['standard_name'])] = r['standard_name']
    name_to_std.pop('', None)

    # Precompute mentions per (date, standard_name) → for geo-dominance checks we use window totals
    std_mentions = gdelt.groupby('standard_name')['NumMentions'].sum().to_dict()

    rows = []
    for fp in sorted(GT_DIR.glob('*.csv')):
        date = fp.stem
        try:
            gt = pd.read_csv(fp, skipinitialspace=True)
        except Exception:
            gt = pd.read_csv(fp, skipinitialspace=True, on_bad_lines='skip')
        gt = gt.rename(columns={'ActionGeo_FullName': 'city', 'Lat': 'lat', 'Long': 'lon'})
        for _, r in gt.iterrows():
            name = r.get('city')
            lat, lon = r.get('lat'), r.get('lon')
            key = _norm(name)
            std = name_to_std.get(key)
            mtype = 'name_exact' if std else None

            if std is None and pd.notna(lat) and pd.notna(lon):
                near = gdelt[gdelt.ActionGeo_Lat.between(lat - 0.5, lat + 0.5)
                             & gdelt.ActionGeo_Long.between(lon - 0.5, lon + 0.5)].copy()
                if not near.empty:
                    near['dist'] = haversine(lat, lon, near.ActionGeo_Lat.values, near.ActionGeo_Long.values)
                    near = near[near.dist <= RADIUS_KM]
                if not near.empty:
                    tot = near.groupby('standard_name')['NumMentions'].sum().sort_values(ascending=False)
                    frac = tot.iloc[0] / tot.sum()
                    if frac >= GEO_DOM:
                        std = tot.index[0]
                        mtype = f'geo_dominant({frac:.0%})'
                    else:
                        mtype = f'ambiguous top={tot.index[0]}({frac:.0%})'

            if std is None and mtype is None:
                mtype = 'not_in_gdelt'

            rows.append({
                'date': date, 'gt_city': name, 'gt_lat': lat, 'gt_lon': lon,
                'standard_name': std, 'match_type': mtype,
                'std_sum_mentions': std_mentions.get(std) if std else None,
            })
    return pd.DataFrame(rows)


def propose_thresholds(stats, gt_df):
    """Print A/C threshold candidates."""
    total_days = 22
    print("\n=== GDELT city baseline distribution (22-day window) ===")
    print(f"Cities total: {len(stats)}")
    for cut in [1, 2, 3, 5, 8, 11, 15, 20]:
        n = (stats.days_present >= cut).sum()
        print(f"  days_present ≥ {cut:2d} : {n:4d} cities")

    print("\navg_mentions_active distribution:")
    print(stats.avg_mentions_active.describe().to_string())

    # GT-anchored view
    gt_cities = gt_df.dropna(subset=['standard_name']).standard_name.unique()
    d_entries = gt_df[gt_df.standard_name.isna()]
    print(f"\n=== GT coverage ===")
    print(f"  Unique GT standard_names matched: {len(gt_cities)}")
    print(f"  GT entries D-tier (no GDELT): {len(d_entries)}")
    print(gt_df['match_type'].value_counts().to_string())

    # Check where GT cities fall in baseline
    gt_stats = stats[stats.standard_name.isin(gt_cities)].copy()
    non_gt_stats = stats[~stats.standard_name.isin(gt_cities)]
    print(f"\n=== GT cities baseline stats (n={len(gt_stats)}) ===")
    print(gt_stats[['days_present', 'avg_mentions_active', 'sum_mentions']].describe().to_string())
    print(f"\n=== Non-GT cities (n={len(non_gt_stats)}) ===")
    print(non_gt_stats[['days_present', 'avg_mentions_active', 'sum_mentions']].describe().to_string())

    # Candidate A/C boundary sweeps
    print("\n=== A/C threshold sweep ===")
    print("criterion: A if (days_present ≥ D) AND (avg_mentions_active ≥ M)")
    print(f"{'D':>3}  {'M':>5}  {'A-cities':>9}  {'C-cities':>9}  {'GT in A':>8}  {'GT in C':>8}")
    for D in [3, 5, 8, 11]:
        for M in [5, 10, 20, 50]:
            A = stats[(stats.days_present >= D) & (stats.avg_mentions_active >= M)]
            C = stats[~stats.index.isin(A.index)]
            gA = A[A.standard_name.isin(gt_cities)]
            gC = C[C.standard_name.isin(gt_cities)]
            print(f"{D:>3}  {M:>5}  {len(A):>9}  {len(C):>9}  {len(gA):>8}  {len(gC):>8}")


def main():
    print(f"[1/3] Loading GDELT {WIN_START}–{WIN_END}...")
    g = load_window_gdelt()
    dates = sorted(g.date.unique())
    print(f"      {len(g):,} rows, {len(dates)} days ({dates[0]} .. {dates[-1]})")
    if len(dates) != 22:
        print(f"      [WARN] expected 22 days, got {len(dates)}")

    print("[2/3] Per-standard_name stats...")
    stats = per_city_stats(g)
    stats.to_csv(OUT_CITY, index=False)
    print(f"      saved {OUT_CITY}")

    print("[3/3] GT mapping to standard_name...")
    gt_df = map_gt_to_standard(g)
    gt_df.to_csv(OUT_GT, index=False)
    print(f"      saved {OUT_GT}")

    propose_thresholds(stats, gt_df)


if __name__ == '__main__':
    main()
