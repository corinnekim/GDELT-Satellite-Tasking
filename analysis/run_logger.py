"""
Per-run structured logger for Q/R experiments.

Layout:
  logs/runs/{timestamp}_Q{q}_R{r}/
    ├── config.json                # hyperparams for the run
    ├── summary.json               # aggregated metrics across all dates
    └── by_date/{YYYYMMDD}/
         ├── metrics.json          # evaluator.evaluate() output (minus per_row)
         ├── verified.csv          # top-K rows that hit LLM stage
         └── anomalies_all.csv     # all anomalies that day (with risk_level)
"""

import json
import statistics
from datetime import datetime
from pathlib import Path

import pandas as pd

from pipeline.config import PROJECT_ROOT


class RunLogger:
    def __init__(self, q: float, r: float, extra_config: dict = None):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.root = PROJECT_ROOT / "logs" / "runs" / f"{ts}_Q{q}_R{r}"
        (self.root / "by_date").mkdir(parents=True, exist_ok=True)

        self.q = q
        self.r = r
        self.per_date_metrics = []  # list of metrics dicts

        config = {"q": q, "r": r, "timestamp": ts}
        if extra_config:
            config.update(extra_config)
        (self.root / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2)
        )

    def log_date(self, target_date: str, metrics: dict, anomalies_all: pd.DataFrame) -> None:
        """
        metrics: output from ModelEvaluator.evaluate() — dict with per_row list.
        anomalies_all: full anomaly df for the date (pre-LLM, includes all risk_levels).
        """
        d = self.root / "by_date" / target_date
        d.mkdir(parents=True, exist_ok=True)

        # 1. verified.csv — top-K rows that went through LLM stage
        verified_df = pd.DataFrame(metrics.get("per_row", []))
        verified_df.to_csv(d / "verified.csv", index=False)

        # 2. metrics.json — everything except per_row (already in verified.csv)
        slim = {k: v for k, v in metrics.items() if k != "per_row"}
        (d / "metrics.json").write_text(
            json.dumps(slim, ensure_ascii=False, indent=2, default=float)
        )

        # 3. anomalies_all.csv — all risk-level anomalies (pre-LLM filter output)
        if not anomalies_all.empty:
            keep_cols = [c for c in
                         ['city', 'date', 'std_innov',
                          'risk_level', 'risk_label', 'lat', 'lng', 'is_anomaly']
                         if c in anomalies_all.columns]
            anomalies_all[keep_cols].to_csv(d / "anomalies_all.csv", index=False)

        self.per_date_metrics.append(slim)

        # Console one-liner for live tracking
        print(
            f"  [{target_date}] "
            f"P={metrics['precision']:.2%} "
            f"R={metrics['recall']:.2%} "
            f"F1={metrics.get('f1', 0):.3f} F{metrics['beta']}={metrics['f_beta']:.3f} "
            f"TP={metrics['n_tp']} FP={metrics['n_fp']} EXC={metrics['n_excluded']} "
            f"GT={metrics['n_gt_hit']}/{metrics['n_gt']}"
        )

    def finalize(self) -> dict:
        """Aggregate per-date metrics and write summary.json. Returns the summary dict."""
        if not self.per_date_metrics:
            summary = {"q": self.q, "r": self.r, "n_dates": 0}
        else:
            def _avg(key):
                vals = [m[key] for m in self.per_date_metrics if m.get(key) is not None]
                return statistics.mean(vals) if vals else 0.0

            def _sum(key):
                return sum(m.get(key, 0) for m in self.per_date_metrics)

            summary = {
                "q": self.q,
                "r": self.r,
                "n_dates": len(self.per_date_metrics),
                "avg_precision": _avg("precision"),
                "avg_recall": _avg("recall"),
                "avg_f1": _avg("f1"),
                "avg_f_beta": _avg("f_beta"),
                "median_f_beta": statistics.median(m["f_beta"] for m in self.per_date_metrics),
                "min_f_beta": min(m["f_beta"] for m in self.per_date_metrics),
                "total_tp": _sum("n_tp"),
                "total_fp": _sum("n_fp"),
                "total_excluded": _sum("n_excluded"),
                "total_gt": _sum("n_gt"),
                "total_gt_hit": _sum("n_gt_hit"),
                "per_date": [
                    {"date": m["date"], "precision": m["precision"], "recall": m["recall"],
                     "f_beta": m["f_beta"], "n_tp": m["n_tp"], "n_fp": m["n_fp"],
                     "n_excluded": m["n_excluded"], "n_gt_hit": m["n_gt_hit"], "n_gt": m["n_gt"]}
                    for m in self.per_date_metrics
                ],
            }

        (self.root / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=float)
        )

        print(
            f"\n[SUMMARY Q={self.q} R={self.r}] "
            f"avg P={summary.get('avg_precision', 0):.2%} "
            f"avg R={summary.get('avg_recall', 0):.2%} "
            f"avg F={summary.get('avg_f_beta', 0):.3f} "
            f"(min F={summary.get('min_f_beta', 0):.3f}) "
            f"across {summary['n_dates']} dates"
        )
        print(f"  → logs at {self.root.relative_to(PROJECT_ROOT)}")

        return summary
