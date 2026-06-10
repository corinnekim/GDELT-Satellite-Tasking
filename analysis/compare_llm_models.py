"""
LLM 모델 A/B 비교: gemini-2.5-flash (baseline, 캐시) vs gemini-3.1-flash-lite (신규).

캐시된 2.5-flash 응답을 baseline으로 사용하고, 같은 프롬프트를
3.1-flash-lite로 직접 호출해 llm_status 판정이 얼마나 바뀌는지 확인한다.

실행: python -m pipeline.compare_llm_models --date 20260228 --top-k 20
"""
from __future__ import annotations

import argparse

import pandas as pd

from pipeline.config import MAIN_PATH, URL_PATH
from pipeline.kalman_filter import compute_conflict_index, detect_anomalies
from pipeline import cache as cache_mod
from pipeline import llm_verification as lv


def run(target_date: str, top_k: int = 20,
        baseline_model: str = "gemini-2.5-flash",
        candidate_model: str = "gemini-3.1-flash-lite") -> pd.DataFrame:
    print(f"\n=== LLM A/B: baseline(cache of 2.5-flash) vs {candidate_model} ===\n")

    print("[SYSTEM] 데이터 로드...")
    raw_df = pd.read_parquet(MAIN_PATH)
    url_df = pd.read_parquet(URL_PATH)
    raw_df["date"] = raw_df["SQLDATE"].astype(str).str[:8]

    print("[TRACK 1] Kalman + anomaly detection...")
    city_daily, filtered = compute_conflict_index(raw_df)
    anomalies_df = detect_anomalies(city_daily, target_date)
    if anomalies_df.empty:
        print("이상징후 없음.")
        return pd.DataFrame()

    # Baseline: 이전에 2.5-flash로 채워진 캐시 재사용. 캐시 미스 시 baseline_model 호출.
    print(f"\n[BASELINE] {baseline_model} (cache 우선)...")
    original_model = lv.MODEL_ID
    lv.MODEL_ID = baseline_model
    try:
        baseline = lv.verify_anomalies_with_llm(
            anomalies_df.copy(), filtered, url_df, target_date, top_k=top_k
        )
    finally:
        lv.MODEL_ID = original_model
    baseline_verified = baseline[baseline["llm_status"] != "UNVERIFIED"][
        ["city", "std_innov", "llm_status", "llm_report"]
    ].rename(columns={"llm_status": "status_baseline", "llm_report": "report_baseline"})

    # Candidate: 3.1-flash-lite로 직접 호출, 캐시 우회 (오염 방지)
    print(f"\n[CANDIDATE] {candidate_model} (cache bypass)...")
    original_get = cache_mod.get_llm
    original_set = cache_mod.set_llm
    lv.MODEL_ID = candidate_model
    cache_mod.get_llm = lambda key: None
    cache_mod.set_llm = lambda key, resp: None
    try:
        candidate = lv.verify_anomalies_with_llm(
            anomalies_df.copy(), filtered, url_df, target_date, top_k=top_k
        )
    finally:
        lv.MODEL_ID = original_model
        cache_mod.get_llm = original_get
        cache_mod.set_llm = original_set

    candidate_verified = candidate[candidate["llm_status"] != "UNVERIFIED"][
        ["city", "llm_status", "llm_report"]
    ].rename(columns={"llm_status": "status_candidate", "llm_report": "report_candidate"})

    merged = baseline_verified.merge(candidate_verified, on="city", how="outer")
    merged = merged.sort_values("std_innov", ascending=False).reset_index(drop=True)

    # 비교 표
    print("\n" + "═" * 100)
    print(f"  A/B 비교: 2.5-flash(baseline) vs {candidate_model}")
    print("═" * 100)
    print(f"  {'도시':<18} | {'z':>6} | {'baseline':<14} | {'candidate':<14} | {'일치':<5}")
    print("  " + "─" * 96)

    agree = 0
    schedulable_match = 0
    both_verified = 0
    SCHED = {"SUCCESS", "AMBIGUOUS"}
    for _, row in merged.iterrows():
        b = str(row.get("status_baseline") or "N/A")
        c = str(row.get("status_candidate") or "N/A")
        match = b == c
        both_v = b not in ("N/A", "ERROR") and c not in ("N/A", "ERROR")
        if both_v:
            both_verified += 1
            if match:
                agree += 1
            if (b in SCHED) == (c in SCHED):
                schedulable_match += 1
        mark = "✅" if match else "❌"
        print(
            f"  {row['city']:<18} | {row['std_innov']:>6.2f} | {b:<14} | {c:<14} | {mark}"
        )

    print("  " + "─" * 96)
    if both_verified:
        print(
            f"\n  · 정확 일치: {agree}/{both_verified} ({agree/both_verified*100:.1f}%)"
        )
        print(
            f"  · 스케줄 판정(SUCCESS/AMBIGUOUS 여부) 일치: "
            f"{schedulable_match}/{both_verified} ({schedulable_match/both_verified*100:.1f}%)"
        )
    print()

    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM 모델 A/B 비교")
    parser.add_argument("--date", type=str, required=True, help="타깃 날짜 YYYYMMDD")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--baseline", type=str, default="gemini-2.5-flash")
    parser.add_argument("--candidate", type=str, default="gemini-3.1-flash-lite")
    args = parser.parse_args()
    run(args.date, top_k=args.top_k,
        baseline_model=args.baseline, candidate_model=args.candidate)
