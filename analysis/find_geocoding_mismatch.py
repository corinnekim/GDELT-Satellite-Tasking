"""GDELT geocoding 의심 케이스 탐지.

GDELT의 ActionGeo_FullName 좌표 vs GeoNames의 동명 인구 최대 도시 좌표 비교.
50km 이상 차이 + 후보 인구 ≥1000인 도시를 의심 목록으로 출력.

용도: 분기별 실행 → 새로 등장한 의심 케이스를 사람이 검증 후
data/gdelt_latlon_correction.csv 에 추가 결정.

주의: 결과는 "확정 오류"가 아니라 "의심 후보". 이름 충돌(같은 이름의 작은 마을 vs
큰 도시)이 자연스러운 케이스도 포함됨. 기사 URL 샘플 확인 필요.

출력: data/review/geocoding_mismatch_review.csv
"""
import pandas as pd
import numpy as np
from pipeline.config import MONITORED_COUNTRIES, PROJECT_ROOT
from pipeline.text_norm import _norm

DIST_THRESHOLD_KM = 50.0
MIN_POP_THRESHOLD = 1000
SAMPLE_URLS = 5
GDELT_PATH = PROJECT_ROOT / "data" / "gdelt_main_2026.parquet"
URL_PATH = PROJECT_ROOT / "data" / "gdelt_url_2026.parquet"
GN_PATH = PROJECT_ROOT / "data" / "geonames" / "_all.parquet"
CORRECTION_PATH = PROJECT_ROOT / "data" / "gdelt_latlon_correction.csv"
OUT_PATH = PROJECT_ROOT / "data" / "review" / "geocoding_mismatch_review.csv"

FIPS2ISO = {'IR':'IR','IS':'IL','IZ':'IQ','LE':'LB','SY':'SY','YM':'YE',
            'AE':'AE','SA':'SA','QA':'QA','KU':'KW','BA':'BH','MU':'OM'}


def haversine(la1, lo1, la2, lo2):
    p = np.pi / 180
    a = (0.5 - np.cos((la2 - la1) * p) / 2
         + np.cos(la1 * p) * np.cos(la2 * p) * (1 - np.cos((lo2 - lo1) * p)) / 2)
    return 12742 * np.arcsin(np.sqrt(a))


def main():
    print(f"[1/3] GDELT 로딩 ({GDELT_PATH.name})")
    df = pd.read_parquet(GDELT_PATH)
    df = df[df.ActionGeo_CountryCode.isin(MONITORED_COUNTRIES)]
    df = df[df.ActionGeo_Type == 4]
    df = df.dropna(subset=['ActionGeo_FullName', 'ActionGeo_Lat', 'ActionGeo_Long'])
    g = (df.groupby(['ActionGeo_FullName', 'ActionGeo_CountryCode'])
           .agg(glat=('ActionGeo_Lat', 'median'),
                glon=('ActionGeo_Long', 'median'),
                n=('GLOBALEVENTID', 'size'))
           .reset_index())
    g['iso'] = g.ActionGeo_CountryCode.map(FIPS2ISO).fillna(g.ActionGeo_CountryCode)
    g['nkey'] = g.ActionGeo_FullName.apply(_norm)
    print(f"      GDELT 후보 (FullName, country): {len(g):,}")

    print(f"[2/3] GeoNames 로딩 ({GN_PATH.name})")
    gn = pd.read_parquet(GN_PATH)
    gn = gn.dropna(subset=['lat', 'lon', 'country']).copy()
    gn['nkey'] = gn.name.fillna('').apply(_norm)
    gn['askey'] = gn.asciiname.fillna('').apply(_norm)
    gn_P = gn[(gn.fclass == 'P') & (gn.population >= MIN_POP_THRESHOLD)]
    print(f"      GeoNames P-fclass, pop≥{MIN_POP_THRESHOLD}: {len(gn_P):,}")

    print(f"[3/3] 의심 케이스 탐지 (>{DIST_THRESHOLD_KM:.0f}km)")
    url_df = pd.read_parquet(URL_PATH) if URL_PATH.exists() else None

    rows = []
    for _, r in g.iterrows():
        if not r.nkey:
            continue
        sub = gn_P[(gn_P.country == r.iso)
                   & ((gn_P.nkey == r.nkey) | (gn_P.askey == r.nkey))]
        if sub.empty:
            continue
        top = sub.loc[sub.population.idxmax()]
        d = haversine(r.glat, r.glon, top.lat, top.lon)
        if d > DIST_THRESHOLD_KM:
            sample_urls = ''
            if url_df is not None:
                eids = df[(df.ActionGeo_FullName == r.ActionGeo_FullName) &
                          (df.ActionGeo_CountryCode == r.ActionGeo_CountryCode)
                          ]['GLOBALEVENTID']
                urls = (url_df[url_df.GLOBALEVENTID.isin(eids)]
                        ['SOURCEURL'].dropna().drop_duplicates().head(SAMPLE_URLS).tolist())
                sample_urls = ' | '.join(urls)
            rows.append({
                'gdelt_fullname': r.ActionGeo_FullName,
                'country': r.ActionGeo_CountryCode,
                'gdelt_lat': round(r.glat, 4), 'gdelt_lon': round(r.glon, 4),
                'true_lat': round(float(top.lat), 4), 'true_lon': round(float(top.lon), 4),
                'dist_km': round(float(d), 1),
                'pop': int(top.population),
                'gdelt_events': int(r.n),
                'geonames_name': top['name'],
                'sample_urls': sample_urls,
            })

    out = pd.DataFrame(rows).sort_values('gdelt_events', ascending=False)

    # 이미 보정 등재된 항목 표시
    if CORRECTION_PATH.exists():
        existing = pd.read_csv(CORRECTION_PATH)
        existing_keys = set(zip(existing.gdelt_fullname, existing.country))
        out['already_corrected'] = [(n, c) in existing_keys
                                    for n, c in zip(out.gdelt_fullname, out.country)]
    else:
        out['already_corrected'] = False

    out['verified'] = ''   # 빈칸/✓/✗
    out['notes'] = ''

    out.to_csv(OUT_PATH, index=False)
    print(f"\n결과: {len(out)} 의심 도시, {out.gdelt_events.sum():,} 이벤트")
    print(f"  이미 보정된 것: {int(out.already_corrected.sum())}")
    print(f"  신규 검토 필요: {int((~out.already_corrected).sum())}")
    print(f"\n저장: {OUT_PATH}")
    print(f"\n다음 단계: CSV 열어서 신규 행 검토 → 확정 건은")
    print(f"          data/gdelt_latlon_correction.csv 로 옮기기")


if __name__ == '__main__':
    main()
