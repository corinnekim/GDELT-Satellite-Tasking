"""
GeoNames(좌표 + fclass 규칙)로 GDELT 지명 → 표준 도시명 매핑.

각 고유 (ActionGeo_FullName, country, lat, lon)마다:
  같은 국가 GeoNames에서 공용 _norm()으로 이름 조회:
  _norm(FullName)이 { _norm(name), _norm(asciiname), _norm(각 alternatenames) }에 있으면 매칭
    - P (도시) → GeoNames.name (Esfahan↔Isfahan은 alternatenames로 통합)
    - S (시설·공항·기지·대학) → 30km 내 가장 가까운 행정 중심지
      (PPLA/PPLC+); 없으면 인구 있는 PPL (≥1k); 없으면 self
    - H/T/L (수역·지형·지역) → _norm(지명)이 P-도시의 _norm(name)을 포함하고
      좌표가 30km 내면 그 P-도시로; 아니면 self
      (Bay Of Haifa → Haifa; Gulf Of Oman은 self 유지)
    - A/R/V (행정·도로·포도밭) → self
    - 매칭 없음 → self  (좌표 fallback 없음: "Gulf Of Oman"이 도시로 매핑되는 것 방지)

출력: data/gdelt_name_map.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

from pipeline.config import CONFIRMED_CODES, MONITORED_COUNTRIES as MONITORED, PROJECT_ROOT, MAIN_PATH, ARTIFACT_ROOT, TEST_MODE
from pipeline.text_norm import _norm

GDELT = MAIN_PATH
GN_PATH = PROJECT_ROOT / "data" / "geonames" / "_all.parquet"
OUT = ARTIFACT_ROOT / "data" / "lookups" / "gdelt_name_map.csv"

# FIPS → ISO 국가코드 (GeoNames join용)
FIPS2ISO = {'IR':'IR','IS':'IL','IZ':'IQ','LE':'LB','SY':'SY','YM':'YE',
            'AE':'AE','SA':'SA','QA':'QA','KU':'KW','BA':'BH','MU':'OM'}

NEAR_P_RADIUS_KM = 30.0
NAME_MATCH_RADIUS_KM = 30.0

# 자동 매핑 후 강제 통합. {ActionGeo_FullName: 표준명}
# GeoNames 자동 매핑이 같은 장소를 다른 표준명으로 분리한 경우 수동 보정.
MANUAL_OVERRIDES = {
    'Kharg': 'Kharg Island',         # P-fclass entry → 통합
    'Kharg Island': 'Kharg Island',  # T-fclass entry → manual로 같은 표준명 강제 (suffix 일치시키기)
}

# 행정 중심지 + 인구 있는 도시 (S 시설의 remap 대상).
# 제외: PPLX (구역), PPLL (소규모 지역), PPLF (농촌),
# PPLQ (폐기), PPLH (역사), PPLW (소멸), PPLR (종교).
ADMIN_FCODES = {'PPLC', 'PPLA', 'PPLA2', 'PPLA3', 'PPLA4', 'STLMT'}
MIN_PPL_POPULATION = 1000


def haversine(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = (0.5 - np.cos((lat2 - lat1) * p) / 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2)
    return 12742 * np.arcsin(np.sqrt(a))


def main():
    g = pd.read_parquet(GDELT)
    if not TEST_MODE:
        g = g[g.EventCode.astype(int).isin(CONFIRMED_CODES)]
    else:
        print('[TEST_MODE] CONFIRMED_CODES 필터 OFF, 전체 CAMEO 코드 포함')
    g = g[g.ActionGeo_CountryCode.isin(MONITORED)]
    g = g[g.ActionGeo_Type == 4]
    g = g.dropna(subset=['ActionGeo_FullName', 'ActionGeo_Lat', 'ActionGeo_Long'])
    print(f'GDELT Type=4 filtered rows: {len(g):,}')

    places = (g.groupby(['ActionGeo_FullName', 'ActionGeo_CountryCode'])
                .agg(lat=('ActionGeo_Lat', 'median'),
                     lon=('ActionGeo_Long', 'median'),
                     n_events=('GLOBALEVENTID', 'size'),
                     n_mentions=('NumMentions', 'sum'))
                .reset_index())
    print(f'Unique (FullName, country) places: {len(places):,}')

    gn = pd.read_parquet(GN_PATH)
    gn = gn.dropna(subset=['lat', 'lon']).copy()

    # 행별 정규화 이름 집합 미리 계산 (name + asciiname + alternatenames)
    def _name_keys(row):
        keys = {_norm(row['name']), _norm(row['asciiname'])}
        alt = row['alternatenames']
        if isinstance(alt, str) and alt:
            for a in alt.split(','):
                k = _norm(a)
                if k:
                    keys.add(k)
        keys.discard('')
        return keys

    gn['_keys'] = gn.apply(_name_keys, axis=1)
    gn_P = gn[gn.fclass == 'P'].copy()

    # Tier 1: 행정 중심지 (PPLC/PPLA/PPLA2+/STLMT). Tier 2: 인구 ≥1k인 다른 PPL.
    # PPLX (동네), PPLL (소지역), PPLF (농촌)은 대상에서 제외 —
    # 안 그러면 작은 거리 차이로 진짜 행정 도시를 이긴다.
    gn_admin = gn_P[gn_P.fcode.isin(ADMIN_FCODES)].copy()
    gn_ppl = gn_P[~gn_P.fcode.isin(ADMIN_FCODES) & (gn_P.population >= MIN_PPL_POPULATION)].copy()
    print(f'GeoNames rows: {len(gn):,}  P: {len(gn_P):,}  admin: {len(gn_admin):,}  populous PPL: {len(gn_ppl):,}')

    gn_by_cc = {cc: sub for cc, sub in gn.groupby('country')}
    gnA_by_cc = {cc: sub for cc, sub in gn_admin.groupby('country')}
    gnPop_by_cc = {cc: sub for cc, sub in gn_ppl.groupby('country')}
    gnP_by_cc = {cc: sub for cc, sub in gn_P.groupby('country')}

    def _closest(sub, lat, lon, radius):
        if sub is None or sub.empty:
            return None, None
        d = haversine(lat, lon, sub.lat.values, sub.lon.values)
        mask = d <= radius
        if not mask.any():
            return None, None
        sub2 = sub[mask].copy()
        sub2['dist'] = d[mask]
        sub2 = sub2.sort_values(['dist', 'population'], ascending=[True, False])
        row = sub2.iloc[0]
        return row, float(row['dist'])

    def nearest_admin_city(iso, lat, lon, radius=NEAR_P_RADIUS_KM):
        """Tier 1: 가장 가까운 행정 중심지. Tier 2: 가장 가까운 인구 있는 PPL."""
        row, d = _closest(gnA_by_cc.get(iso), lat, lon, radius)
        if row is not None:
            return row, d, 'admin'
        row, d = _closest(gnPop_by_cc.get(iso), lat, lon, radius)
        if row is not None:
            return row, d, 'populous'
        return None, None, None

    out_rows = []
    for _, p in places.iterrows():
        full = p.ActionGeo_FullName
        cc = p.ActionGeo_CountryCode
        iso = FIPS2ISO.get(cc, cc)
        lat, lon = p.lat, p.lon
        key = _norm(full)

        sub_cc = gn_by_cc.get(iso)
        matched = None
        match_dist = None
        if sub_cc is not None and not sub_cc.empty and key:
            hit_mask = sub_cc['_keys'].apply(lambda ks: key in ks)
            cand = sub_cc[hit_mask]
            if not cand.empty:
                cand = cand.copy()
                cand['dist'] = haversine(lat, lon, cand.lat.values, cand.lon.values)
                cand = cand[cand.dist <= NAME_MATCH_RADIUS_KM]
                if not cand.empty:
                    # fclass 여럿이면 P 우선; 그다음 인구; 그다음 거리
                    cand['_is_p'] = (cand.fclass == 'P').astype(int)
                    cand = cand.sort_values(['_is_p', 'population', 'dist'],
                                            ascending=[False, False, True])
                    matched = cand.iloc[0]
                    match_dist = float(matched['dist'])

        # P-도시는 country suffix 붙여 동음이의 충돌 방지 (Najaf|IR vs Najaf|IQ).
        # 수역(H)·지형(T)·지역(L)·기타는 그대로, 다국가에 걸친 entity 통합 유지.
        def _city(name):
            return f'{name}|{cc}'

        matched_adm1 = None  # 충돌 감지용, 같은 (name, country)에 여러 adm1이면 분리 suffix 추가

        if matched is not None:
            fclass = matched.fclass
            matched_name = matched['name']
            match_method = 'name_match'
            if fclass == 'P':
                standard_name = _city(matched['name'])
                matched_adm1 = matched.get('adm1') if hasattr(matched, 'get') else (matched['adm1'] if 'adm1' in matched else None)
                reason = f'P ({match_dist:.1f}km): {full!r} → {standard_name!r}'
            elif fclass == 'S':
                near, nd, tier = nearest_admin_city(iso, lat, lon)
                if near is not None:
                    standard_name = _city(near['name'])
                    matched_adm1 = near.get('adm1') if hasattr(near, 'get') else (near['adm1'] if 'adm1' in near.index else None)
                    reason = f'S→{tier} ({nd:.1f}km, {near.fcode}): {full!r} → {standard_name!r}'
                else:
                    standard_name = full
                    reason = f'S, no admin/populous city within {NEAR_P_RADIUS_KM:.0f}km → self'
            elif fclass in {'H', 'T', 'L'}:
                # 지명 안에 P-도시가 들어 있으면 그 P-도시로 매핑
                # (Bay Of Haifa → Haifa). Gulf Of Oman은 self: "oman"은
                # 국가지 P-도시가 아니라 부분 문자열 매칭이 안 걸린다.
                key = _norm(full)
                sub_p = gnP_by_cc.get(iso)
                remapped = None
                if sub_p is not None and not sub_p.empty and key:
                    # Bbox 사전 필터 (~0.4° ≈ 44km)로 후보 줄이기
                    bb = sub_p[(sub_p.lat.between(lat - 0.4, lat + 0.4))
                               & (sub_p.lon.between(lon - 0.4, lon + 0.4))]
                    for _, pc in bb.iterrows():
                        pkey = _norm(pc['name'])
                        if len(pkey) < 4 or pkey not in key or pkey == key:
                            continue
                        d = float(haversine(lat, lon, pc.lat, pc.lon))
                        if d <= NEAR_P_RADIUS_KM and (
                            remapped is None
                            or pc.population > remapped['pop']
                            or (pc.population == remapped['pop'] and d < remapped['d'])):
                            remapped = {'name': pc['name'], 'd': d, 'pop': pc.population,
                                       'adm1': pc.get('adm1') if hasattr(pc, 'get') else (pc['adm1'] if 'adm1' in pc.index else None)}
                if remapped is not None:
                    standard_name = _city(remapped['name'])
                    matched_adm1 = remapped.get('adm1')
                    reason = f'{fclass}→substring P ({remapped["d"]:.1f}km): {full!r} → {standard_name!r}'
                else:
                    standard_name = full
                    reason = f'{fclass} (feature, no substring P) → self'
            else:
                standard_name = full
                reason = f'{fclass} (feature) → self'
        else:
            fclass = None
            matched_name = None
            match_method = 'no_match'
            match_dist = None
            standard_name = full
            reason = 'no GeoNames match → self'

        out_rows.append({
            'ActionGeo_FullName': full,
            'ActionGeo_CountryCode': cc,
            'lat': lat, 'lon': lon,
            'n_events': p.n_events,
            'n_mentions': p.n_mentions,
            'fclass': fclass,
            'match_method': match_method,
            'match_dist_km': round(match_dist, 2) if match_dist is not None else None,
            'matched_geonames_name': matched_name,
            'matched_adm1': matched_adm1,
            'standard_name': standard_name,
            'remap_reason': reason,
        })

    out = pd.DataFrame(out_rows)

    # 같은 (standard_name) 안에 다른 adm1이 섞이면 충돌 분리 (Mahabad 케이스)
    # 같은 country 안 동음이의 도시 → matched_adm1 기준으로 suffix 추가.
    # 예: 'Mahābād|IR' (adm1=01) + 'Mahābād|IR' (adm1=42) → 'Mahābād|IR|01' / 'Mahābād|IR|42'
    sub = out[out['matched_adm1'].notna() & out['standard_name'].str.contains(r'\|', na=False)].copy()
    nadm1 = sub.groupby('standard_name')['matched_adm1'].nunique()
    collision_names = set(nadm1[nadm1 > 1].index)
    if collision_names:
        mask = out['standard_name'].isin(collision_names) & out['matched_adm1'].notna()
        for idx in out[mask].index:
            base = out.at[idx, 'standard_name']
            adm1 = out.at[idx, 'matched_adm1']
            out.at[idx, 'standard_name'] = f"{base}|{adm1}"
            out.at[idx, 'remap_reason']  = (out.at[idx, 'remap_reason'] or '') + f"  [collision adm1={adm1}]"
        print(f'  동음이의 분리 (adm1 추가): {len(collision_names)}개 standard_name → {int(mask.sum())} 행 변경')

    # manual overrides, 자동 매핑 후 강제 통합 (도시는 country suffix 붙여 일관성 유지)
    for full, std in MANUAL_OVERRIDES.items():
        mask = out.ActionGeo_FullName == full
        n = int(mask.sum())
        if n:
            std_with_cc = out.loc[mask, 'ActionGeo_CountryCode'].apply(lambda c: f'{std}|{c}')
            out.loc[mask, 'standard_name'] = std_with_cc.values
            out.loc[mask, 'remap_reason']  = "manual override → " + std_with_cc.values
            print(f'  manual override: {full} → {std}|<cc>  ({n} row)')

    out.to_csv(OUT, index=False)
    print(f'\nSaved: {OUT}')
    print(f'\nfclass distribution:')
    print(out.fclass.fillna('(none)').value_counts().to_string())
    print(f'\nmatch_method:')
    print(out.match_method.value_counts().to_string())
    changed = (out.ActionGeo_FullName != out.standard_name).sum()
    print(f'\nRemapped: {changed} / {len(out)}')
    print(f'Names reduced: {out.ActionGeo_FullName.nunique()} → {out.standard_name.nunique()}')


if __name__ == '__main__':
    main()