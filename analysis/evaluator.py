"""
Evaluator for filter+LLM pipeline.

TP/FP rules (precision):
  TP  : GT match within radius (regardless of LLM)   OR   LLM ∈ {SUCCESS, AMBIGUOUS}
  FP  : LLM ∈ {DATE_MISMATCH, DROPPED, NO_MENTION}
  EXC : LLM ∈ {ARTICLE_UNREACHABLE, ERROR}           → excluded from precision denominator
Recall: fraction of daily GT cities with ≥1 verified anomaly within radius.

Ground truth: directory of per-date CSVs with columns
  SQLDATE, ActionGeo_FullName, Lat, Long, Source, Event_Description
"""

from pathlib import Path
import numpy as np
import pandas as pd
from pipeline.text_norm import _norm

TP_STATUSES = {'SUCCESS', 'AMBIGUOUS'}
FP_STATUSES = {'DROPPED', 'NO_MENTION', 'DATE_MISMATCH'}
EXCLUDED_STATUSES = {'ARTICLE_UNREACHABLE', 'ERROR'}


def get_distance(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = 0.5 - np.cos((lat2 - lat1) * p) / 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 12742 * np.arcsin(np.sqrt(a))


def _load_ground_truth(gt_path) -> pd.DataFrame:
    """Accept either a single CSV or a directory of per-date CSVs. Returns unified df with standard columns: date, city, lat, lon."""
    p = Path(gt_path)
    if p.is_dir():
        files = sorted(p.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSVs in {p}")
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    else:
        df = pd.read_csv(p)

    # Standardize column names
    rename_map = {}
    for src, dst in [('SQLDATE', 'date'), ('ActionGeo_FullName', 'city'),
                     ('Lat', 'lat'), ('Long', 'lon'),
                     ('Longitude', 'lon'), ('Latitude', 'lat')]:
        if src in df.columns and dst not in df.columns:
            rename_map[src] = dst
    df = df.rename(columns=rename_map)

    df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d', errors='coerce').dt.strftime('%Y%m%d')
    df = df.dropna(subset=['date', 'lat', 'lon'])
    return df[['date', 'city', 'lat', 'lon']]


class ModelEvaluator:
    def __init__(self, gt_path, radius_km: float = 50.0, beta: float = 1.5, name_to_fids: dict = None):
        """
        name_to_fids: optional {lowercase_city_name: set(FeatureID)} built from raw GDELT.
                      Enables FeatureID-based fallback for alias cases like Isfahan↔Esfahan
                      (same FeatureID after preprocess consolidation).
        """
        self.gt_df = _load_ground_truth(gt_path)
        self.radius_km = radius_km
        self.beta = beta
        self.name_to_fids = name_to_fids or {}

    def available_dates(self) -> list:
        return sorted(self.gt_df['date'].unique().tolist())

    def _gt_match(self, lat, lon, target_city, target_fid, daily_gt: pd.DataFrame):
        """Return matched GT city name or None. Tiers: name substring → FeatureID alias → geo within radius."""
        if daily_gt.empty:
            return None

        # 1) City-name match via _norm (strips diacritics, punctuation, country suffix).
        # e.g. "El Khiâm|LE" → _norm → "elkhiam"; GT "Khiam" → _norm → "khiam" → substring match.
        if isinstance(target_city, str) and target_city.strip():
            # Strip country suffix before normalising so "|LE" doesn't pollute the token.
            city_base = target_city.split('|')[0]
            t = _norm(city_base)
            if t:
                for _, gt in daily_gt.iterrows():
                    gt_name = _norm(str(gt['city'])) if pd.notna(gt['city']) else ""
                    if gt_name and (gt_name == t or gt_name in t or t in gt_name):
                        return gt['city']

        # 2) FeatureID alias match: GT city name → GDELT FeatureIDs (pre-consolidation).
        # Catches Esfahan↔Isfahan where preprocess merged aliases under one standard name+FID.
        if target_fid is not None and not pd.isna(target_fid) and self.name_to_fids:
            try:
                tfid = int(target_fid)
            except (TypeError, ValueError):
                tfid = None
            if tfid is not None:
                for _, gt in daily_gt.iterrows():
                    gt_name = _norm(str(gt['city'])) if pd.notna(gt['city']) else ""
                    if gt_name and tfid in self.name_to_fids.get(gt_name, set()):
                        return gt['city']

        # 3) Geo match (haversine ≤ radius_km)
        if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
            return None
        for _, gt in daily_gt.iterrows():
            if get_distance(lat, lon, gt['lat'], gt['lon']) <= self.radius_km:
                return gt['city']
        return None

    def evaluate(self, anomalies: pd.DataFrame, target_date: str) -> dict:
        """Precision from LLM verdicts on top-k. Recall = GT cities present in the same top-k (LLM status ignored)."""
        verified = anomalies[anomalies['llm_status'] != 'UNVERIFIED'].copy()
        daily_gt = self.gt_df[self.gt_df['date'] == target_date]
        n_gt = len(daily_gt)

        n_tp = n_fp = n_excluded = 0
        hit_cities = set()
        status_counts = {}
        per_row = []

        for _, row in verified.iterrows():
            status = row['llm_status']
            lat = row.get('lat')
            lon = row.get('lng')
            status_counts[status] = status_counts.get(status, 0) + 1

            fid = row.get('ActionGeo_FeatureID')
            gt_match = self._gt_match(lat, lon, row.get('city'), fid, daily_gt)
            # Recall credit: GT match inside the top-k that reached LLM, regardless of verdict.
            if gt_match is not None:
                hit_cities.add(gt_match)

            if gt_match is not None:
                judgement = 'TP'
            elif status in TP_STATUSES:
                judgement = 'TP'
            elif status in FP_STATUSES:
                judgement = 'FP'
            elif status in EXCLUDED_STATUSES:
                judgement = 'EXCLUDED'
            else:
                judgement = 'EXCLUDED'

            if judgement == 'TP':
                n_tp += 1
            elif judgement == 'FP':
                n_fp += 1
            else:
                n_excluded += 1

            # Extract LLM message + URLs for review (especially for FP/EXCLUDED rows)
            llm_msg = None
            report_raw = row.get('llm_report')
            if isinstance(report_raw, str) and report_raw:
                try:
                    import json as _json
                    llm_msg = _json.loads(report_raw).get('Summary')
                except Exception:
                    llm_msg = report_raw[:500]
            urls = row.get('source_urls')
            if isinstance(urls, list):
                urls_str = " | ".join(urls)
            else:
                urls_str = urls if isinstance(urls, str) else None

            per_row.append({
                'city': row.get('city'),
                'feature_id': int(fid) if (fid is not None and pd.notna(fid)) else None,
                'std_innov': float(row['std_innov']) if pd.notna(row.get('std_innov')) else None,
                'lat': float(lat) if pd.notna(lat) else None,
                'lng': float(lon) if pd.notna(lon) else None,
                'llm_status': status,
                'llm_message': llm_msg,
                'source_urls': urls_str,
                'scrape_stats': row.get('scrape_stats'),
                'gt_match': gt_match,
                'judgement': judgement,
            })

        denom = n_tp + n_fp
        precision = n_tp / denom if denom > 0 else 0.0
        recall = len(hit_cities) / n_gt if n_gt > 0 else 0.0

        def _f(beta):
            b2 = beta ** 2
            return ((1 + b2) * precision * recall / (b2 * precision + recall)) if (precision + recall) > 0 else 0.0

        f1     = _f(1.0)
        f_beta = _f(self.beta)

        return {
            'date': target_date,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'f_beta': f_beta,
            'beta': self.beta,
            'n_tp': n_tp,
            'n_fp': n_fp,
            'n_excluded': n_excluded,
            'n_verified': len(verified),
            'n_gt': n_gt,
            'n_gt_hit': len(hit_cities),
            'status_counts': status_counts,
            'per_row': per_row,
        }

    # Backwards-compat shim for run_experiment.py which expects (score, recall)
    def run_scoring(self, results, target_date):
        m = self.evaluate(results, target_date)
        return m['f_beta'], m['recall']
