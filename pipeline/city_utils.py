"""
도시명 정규화 유틸리티
- GDELT / GT / 수작업 문서 사이의 흔한 표기 차이를 흡수한다.
- 실제 다른 도시를 과도하게 합치지 않도록 "철자/동의어" 위주만 다룬다.
"""
from __future__ import annotations

import re


_CITY_ALIAS_MAP = {
    "teheran": "Tehran",
    "tehran": "Tehran",
    "isfahan": "Isfahan",
    "esfahan": "Isfahan",
    "mecca": "Makkah",
    "makkah": "Makkah",
    "riyad": "Riyadh",
    "riyadh": "Riyadh",
    "tebriz": "Tabriz",
    "tabriz": "Tabriz",
    "hamedan": "Hamedan",
    "hamadan": "Hamedan",
    "ahwaz": "Ahvaz",
    "ahvaz": "Ahvaz",
    "bagdad": "Baghdad",
    "baghdad": "Baghdad",
    "arbil": "Erbil",
    "erbil": "Erbil",
    "duhok": "Duhok",
    "dohuk": "Duhok",
    "siraz": "Shiraz",
    "shiraz": "Shiraz",
    "mashad": "Mashhad",
    "mashhad": "Mashhad",
    "bushire": "Bushehr",
    "bushehr": "Bushehr",
    "beer sheba": "Beersheba",
    "beer sheva": "Beersheba",
    "be'er sheva": "Beersheba",
    "tel aviv yafo": "Tel Aviv",
    "tel aviv-yafo": "Tel Aviv",
    "kharg": "Kharg Island",
    "jurf al nasr": "Jurf al Sakhr",
    "jurf al-nasr": "Jurf al Sakhr",
    "jurf al sakhr": "Jurf al Sakhr",
    "kuwait": "Kuwait City",
}


def _normalize_key(value: str) -> str:
    key = str(value).strip().lower()
    key = key.split(",")[0].strip()
    key = re.sub(r"[\u2018\u2019\u201c\u201d']", "", key)
    key = re.sub(r"[-_/]+", " ", key)
    key = re.sub(r"\s+", " ", key).strip()
    return key


def normalize_city_name(value: str) -> str:
    """도시명을 alias map을 통해 대표 이름으로 정규화한다."""
    raw = str(value).strip()
    if not raw:
        return raw

    key = _normalize_key(raw)
    alias = _CITY_ALIAS_MAP.get(key)
    if alias is not None:
        return alias

    # alias가 없으면 원본 표기를 최대한 보존하되, 콤마 이후 세부 표기만 제거한다.
    return raw.split(",")[0].strip()


def normalize_city_key(value: str) -> str:
    """비교/조인용 lowercase key."""
    return normalize_city_name(value).strip().lower()


def _is_reliable_feature_id(feature_id: str) -> bool:
    fid = str(feature_id).strip()
    if not fid or fid.lower() == "nan":
        return False
    # 국가 코드처럼 짧은 alpha-only 값은 위치 식별자보다는 국가 약어일 가능성이 높다.
    if fid.isalpha() and len(fid) <= 3:
        return False
    return True


def _orthographic_signature(value: str) -> str:
    text = normalize_city_name(value).lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def unify_city_by_feature_id(city_series, feature_id_series):
    """FeatureID가 같은 철자 변형은 같은 대표 이름으로 맞춘다.

    주의:
    - FeatureID 하나에 city와 region이 함께 묶이는 사례가 있어 무조건 병합하지 않는다.
    - orthographic signature가 동일한 경우만 병합한다.
    """
    cities = city_series.astype(str).copy()
    features = feature_id_series.astype(str).copy()

    replacements: dict[str, str] = {}
    tmp = {}
    for idx, (city, fid) in enumerate(zip(cities.tolist(), features.tolist())):
        if not _is_reliable_feature_id(fid):
            continue
        tmp.setdefault(fid, []).append(city)

    for fid, names in tmp.items():
        signatures = {_orthographic_signature(name) for name in names}
        if len(signatures) != 1:
            continue
        normalized_names = [normalize_city_name(name) for name in names]
        counts = {}
        for name in normalized_names:
            counts[name] = counts.get(name, 0) + 1
        primary = max(counts, key=lambda name: (counts[name], len(name)))
        replacements[fid] = primary

    if not replacements:
        return cities

    out = cities.copy()
    for idx, fid in enumerate(features.tolist()):
        primary = replacements.get(fid)
        if primary is not None:
            out.iloc[idx] = primary
    return out
