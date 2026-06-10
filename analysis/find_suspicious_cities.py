"""
GDELT 이름 매핑 결과에서 의심스러운 이름을 자동 플래그.

기준:
  A. GeoNames 국가에 아예 없음 (remap_reason='no geonames country') + 언급 많음
  B. 뉴스사/조직/무기/인명 키워드 부분일치

제외:
  - 이미 CITY_BLACKLIST에 있는 이름
  - base / airport / university 키워드 (build_name_map가 이미 도시로 통합했거나
    통합 실패한 경우 self로 유지하기로 함 - 사용자 정책)
  - GeoNames 인구=0 규칙 (GeoNames population 결측이 흔해서 신뢰도 낮음)

출력: data/review/city_blacklist_candidates.csv
사용법:
  python -m analysis.find_suspicious_cities
  → 결과 훑어보고 data/blacklist_manual.txt 에 추가
"""
from pathlib import Path
import pandas as pd

from pipeline.config import CITY_BLACKLIST

ROOT = Path(__file__).resolve().parent.parent
MAP_PATH = ROOT / "data" / "gdelt_name_map.csv"
OUT_PATH = ROOT / "data" / "review" / "city_blacklist_candidates.csv"

MIN_MENTIONS = 100

# 뉴스사/조직/무기/인명 키워드 (부분일치 시 플래그)
SUSPICIOUS_KEYWORDS = [
    'ministry','news','times','jazeera','reuters','afp','post',
    'brigade','guard','corps','militia',
    'hezbollah','hamas','taliban','houthi','basij','sepah','quds',
    'shahed','drone','missile',
    'red crescent','red cross','embassy','consulate',
    'palestin','gaza','west bank','hebron',
]


def main():
    df = pd.read_csv(MAP_PATH)
    df = df[~df['ActionGeo_FullName'].isin(CITY_BLACKLIST)]
    flagged = []

    # A. GeoNames 매칭 실패 + 언급 많음
    unmatched = df[(df.remap_reason == 'no geonames country') & (df.n_mentions >= MIN_MENTIONS)]
    for _, r in unmatched.iterrows():
        flagged.append({**r.to_dict(), 'flag': 'no_geonames_match',
                        'note': f'{r.n_mentions} mentions, no GeoNames match'})

    # B. 키워드 부분일치
    for _, r in df.iterrows():
        name_lc = str(r.ActionGeo_FullName).lower()
        for kw in SUSPICIOUS_KEYWORDS:
            if kw in name_lc:
                flagged.append({**r.to_dict(), 'flag': f'keyword:{kw}',
                                'note': f'contains "{kw}"'})
                break

    if not flagged:
        print('No suspicious candidates found.')
        return

    out = pd.DataFrame(flagged).drop_duplicates(
        subset=['ActionGeo_FullName','ActionGeo_CountryCode','flag'])
    out = out.sort_values('n_mentions', ascending=False)
    out[['ActionGeo_FullName','ActionGeo_CountryCode','n_mentions',
         'standard_name','flag','note']].to_csv(OUT_PATH, index=False)

    print(f'Flagged {len(out)} suspicious entries → {OUT_PATH}')
    print('\n=== TOP 30 by mentions ===')
    print(out[['ActionGeo_FullName','ActionGeo_CountryCode','n_mentions',
               'standard_name','flag','note']].head(30).to_string(index=False))


if __name__ == '__main__':
    main()
