"""
Ground Truth 지명 정규화.

각 GT 항목의 ActionGeo_FullName을 표준 GDELT FullName으로 매핑한다
(필터링된 모니터링 대상국 분쟁 코드 부분집합에 나타나는 형태).

티어 (순서대로):
  1. name_exact    , casefold/strip/구분자 제거 후 GDELT에서 정확히 일치
  2. featureid_geo , GT 좌표 radius_km 내 GDELT 이벤트, dominant FeatureID 사용
  3. geo_unique    , radius_km 내 단일 dominant GDELT 도시 (국소 멘션의 ≥60%)
  4. AMBIGUOUS     , 후보 여럿이고 우세 도시 없음 → drop
  5. NOT_IN_GDELT  , 반경 내 GDELT 없음 → D tier

출력:
  - data/review/ground_truth_normalized/*.csv  (날짜별, 모호 행 제거,
                                                + standard_name, match_type, confidence 컬럼)
  - data/review/ground_truth_report.csv         (리포트: 모든 GT 항목의 처리 결과)

파이프라인 연동:
  evaluator / tier_assigner / trainer는 정규화된 GT를 읽는다.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd

# 설정
ROOT = Path(__file__).resolve().parent.parent
GT_IN_DIR  = ROOT / "data" / "ground_truth"
GT_OUT_DIR = ROOT / "data" / "review" / "ground_truth_normalized"
REPORT_PATH = ROOT / "data" / "review" / "ground_truth_report.csv"
GDELT_PATH = ROOT / "data" / "gdelt_main_2026.parquet"

RADIUS_KM = 50.0
GEO_DOMINANCE = 0.60   # 반경 내 멘션의 ≥60%를 차지해야 dominant

from pipeline.config import CONFIRMED_CODES, MONITORED_COUNTRIES, CITY_BLACKLIST
from pipeline.preprocess import apply_name_map


def _norm_str(s: str) -> str:
    """비교용: 소문자화, 양끝 공백 제거, 하이픈/어포스트로피/공백 삭제."""
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"[\s\-'’]+", "", s)
    return s


def _haversine(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = (0.5 - np.cos((lat2 - lat1) * p) / 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2)
    return 12742 * np.arcsin(np.sqrt(a))


def load_filtered_gdelt():
    raw = pd.read_parquet(GDELT_PATH)
    raw = raw[raw["EventCode"].astype(int).isin(CONFIRMED_CODES)]
    raw = raw[raw["ActionGeo_CountryCode"].isin(MONITORED_COUNTRIES)]
    raw = raw[raw["ActionGeo_Type"] == 4]
    raw = raw.dropna(subset=["ActionGeo_FullName", "ActionGeo_Lat", "ActionGeo_Long"])
    # GeoNames 기반 name map 적용 (Bay Of Haifa → Haifa 등)
    raw = apply_name_map(raw)
    # 블랙리스트 지명 제거 (Hebron, Gaza, 조직명 등)
    raw = raw[~raw["ActionGeo_FullName"].isin(CITY_BLACKLIST)]
    raw["norm"] = raw["ActionGeo_FullName"].apply(_norm_str)
    return raw


def build_indices(gdelt):
    """조회용 사전 미리 계산:
       norm → 표준 FullName (여럿이면 가장 많이 멘션된 변형)
       도시별 총 멘션 수  (geo dominance 검사용)
    """
    # 표준명: norm 키별로 NumMentions 합이 가장 큰 FullName 선택
    grp = gdelt.groupby(["norm", "ActionGeo_FullName"])["NumMentions"].sum().reset_index()
    norm_to_name = (grp.sort_values("NumMentions", ascending=False)
                        .drop_duplicates("norm")
                        .set_index("norm")["ActionGeo_FullName"].to_dict())
    return norm_to_name


def match_gt_entry(gt_row, gdelt, norm_to_name):
    """반환 dict: {standard_name, match_type, confidence, candidate_info}"""
    gt_name_norm = _norm_str(gt_row["city"])
    lat, lon = gt_row["lat"], gt_row["lon"]

    # norm 정확히 일치
    if gt_name_norm and gt_name_norm in norm_to_name:
        return {
            "standard_name": norm_to_name[gt_name_norm],
            "match_type": "name_exact",
            "confidence": 1.0,
            "note": "",
        }

    # 2·3) 좌표 기반: 반경 내 GDELT 이벤트 탐색
    if pd.isna(lat) or pd.isna(lon):
        return {"standard_name": None, "match_type": "no_coords", "confidence": 0.0, "note": ""}

    # 공간 사전 필터 (bbox ~0.7° ≈ 적도에서 약 77km, 여유 있게)
    bb = gdelt[
        gdelt["ActionGeo_Lat"].between(lat - 0.7, lat + 0.7) &
        gdelt["ActionGeo_Long"].between(lon - 0.7, lon + 0.7)
    ].copy()
    if bb.empty:
        return {"standard_name": None, "match_type": "not_in_gdelt",
                "confidence": 0.0, "note": "no events in bbox"}
    bb["dist"] = _haversine(lat, lon, bb["ActionGeo_Lat"].values, bb["ActionGeo_Long"].values)
    near = bb[bb["dist"] <= RADIUS_KM]
    if near.empty:
        return {"standard_name": None, "match_type": "not_in_gdelt",
                "confidence": 0.0, "note": f"no events in {RADIUS_KM}km"}

    # 표준 도시별 멘션 집계
    city_tot = (near.groupby("ActionGeo_FullName")["NumMentions"].sum()
                    .sort_values(ascending=False))
    total = city_tot.sum()
    top_city = city_tot.index[0]
    top_frac = city_tot.iloc[0] / total

    # FeatureID / dominant 도시 매칭
    if top_frac >= GEO_DOMINANCE:
        return {
            "standard_name": top_city,
            "match_type": "geo_dominant",
            "confidence": float(top_frac),
            "note": f"{len(city_tot)} cities in radius, top '{top_city}' holds {top_frac:.0%}",
        }

    # 모호 → drop
    return {
        "standard_name": None,
        "match_type": "ambiguous",
        "confidence": float(top_frac),
        "note": f"candidates: {city_tot.head(3).to_dict()}",
    }


def main():
    print(f"[1/4] Loading filtered GDELT from {GDELT_PATH.name}...")
    gdelt = load_filtered_gdelt()
    print(f"      {len(gdelt):,} rows, {gdelt['ActionGeo_FullName'].nunique():,} unique cities")

    print("[2/4] Building norm → standard-name index...")
    norm_to_name = build_indices(gdelt)
    print(f"      {len(norm_to_name):,} norm keys")

    print("[3/4] Processing GT files...")
    GT_OUT_DIR.mkdir(exist_ok=True)
    report = []
    kept = dropped_amb = dropped_d = 0

    for fp in sorted(GT_IN_DIR.glob("*.csv")):
        date = fp.stem
        gt = pd.read_csv(fp)
        # GT 컬럼명을 표준 형태로 변경
        gt = gt.rename(columns={"ActionGeo_FullName": "city", "Lat": "lat", "Long": "lon"})
        out_rows = []
        for _, r in gt.iterrows():
            m = match_gt_entry(r, gdelt, norm_to_name)
            report.append({
                "date": date,
                "gt_city": r["city"],
                "gt_lat": r["lat"],
                "gt_lon": r["lon"],
                **m,
            })
            if m["standard_name"] is not None:
                out_rows.append({
                    **r.to_dict(),
                    "standard_name": m["standard_name"],
                    "match_type": m["match_type"],
                    "confidence": m["confidence"],
                })
                kept += 1
            elif m["match_type"] == "ambiguous":
                dropped_amb += 1
            else:
                dropped_d += 1

        if out_rows:
            out_df = pd.DataFrame(out_rows)
            # 원본 GDELT 스타일 컬럼 유지 + standard_name 추가
            out_df.to_csv(GT_OUT_DIR / fp.name, index=False)

    report_df = pd.DataFrame(report)
    report_df.to_csv(REPORT_PATH, index=False)

    print("\n" + "=" * 60)
    print(f"[4/4] Summary")
    print("=" * 60)
    print(f"  Total GT entries:      {len(report_df)}")
    print(f"  Kept (matched):        {kept}")
    print(f"  Dropped (ambiguous):   {dropped_amb}")
    print(f"  D tier (not in GDELT): {dropped_d}")
    print()
    print("Match type breakdown:")
    print(report_df["match_type"].value_counts().to_string())
    print()
    print(f"Normalized GT:  {GT_OUT_DIR}/")
    print(f"Report log:     {REPORT_PATH}")


if __name__ == "__main__":
    main()
