"""
Level 2a, 위성 카탈로그 로더
정적 기본 위성 목록과 eo-predictor 기반 별도 시나리오를 공통 형식으로 제공한다.
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EO_PREDICTOR_SAT_DIR = PROJECT_ROOT / "data" / "satellite_data"

# 운영 기본 위성 5대 (단독 또는 다른 시나리오 조합 시 사용).
# 선정 기준: 센서 비율 / 해상도 / 군집 여부 / 데이터 접근성 / 활성 상태.
DEFAULT_SATELLITES: list[dict] = [
    {
        "name": "SpaceEye-T",  "norad_id": 63229,
        "type": "optical",     "swath_km": 12,    "resolution_m": 0.25,
        "off_nadir_deg": 45,   "orbit": "SSO",    "altitude_km": 510,
        "priority": 1,
    },
    {
        "name": "KOMPSAT-7",   "norad_id": 66820,
        "type": "optical",     "swath_km": 15,    "resolution_m": 0.30,
        "off_nadir_deg": 30,   "orbit": "SSO",    "altitude_km": 570,
        "priority": 2,
    },
    {
        "name": "SkySat-C12",  "norad_id": 43797,
        "type": "optical",     "swath_km": 5.9,   "resolution_m": 0.50,
        "off_nadir_deg": 30,   "orbit": "SSO",    "altitude_km": 500,
        "priority": 3,
    },
    {
        "name": "Sentinel-2A", "norad_id": 40697,
        "type": "optical",     "swath_km": 290,   "resolution_m": 10,
        "off_nadir_deg": None, "orbit": "SSO",    "altitude_km": 786,
        "priority": 4,
    },
    {
        "name": "ICEYE-X2",    "norad_id": 43800,
        "type": "sar",         "swath_km": 30,    "resolution_m": 1,
        "off_nadir_deg": 35,   "orbit": "SSO",    "altitude_km": 570,
        "priority": 5,
    },
]


def _default_spaceeye_entry() -> dict:
    for sat in DEFAULT_SATELLITES:
        if sat["name"] == "SpaceEye-T":
            return sat.copy()
    raise ValueError("기본 SATELLITES에서 SpaceEye-T를 찾을 수 없습니다.")


def _default_satellite_entry(name: str) -> dict:
    for sat in DEFAULT_SATELLITES:
        if sat["name"] == name:
            return sat.copy()
    raise ValueError(f"기본 SATELLITES에서 {name}를 찾을 수 없습니다.")


def _load_eo_constellation(filename: str) -> list[dict]:
    file_path = EO_PREDICTOR_SAT_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"EO Predictor 위성 정의 파일이 없습니다: {file_path}")

    constellation = json.loads(file_path.read_text(encoding="utf-8"))
    satellites = []
    for norad_id in constellation.get("norad_ids", []):
        satellites.append({
            "name": f"{constellation['constellation']}-{norad_id}",
            "norad_id": int(norad_id),
            "type": str(constellation["sensor_type"]).lower(),
            "swath_km": float(constellation["swath_km"]),
            "resolution_m": float(constellation["spatial_res_cm"]) / 100.0,
            "off_nadir_deg": constellation.get("off_nadir_deg"),
            "orbit": "SSO",
            "altitude_km": float(constellation["altitude_km"]),
            "priority": 10,
            "constellation": constellation["constellation"],
            "operator": constellation.get("operator", ""),
            "data_access": constellation.get("data_access", ""),
            "tasking": constellation.get("tasking"),
            "source": "eo-predictor",
        })
    return satellites


def load_satellite_catalog(scenario: str = "default") -> list[dict]:
    """실행 시나리오에 맞는 위성 목록을 반환한다."""
    if scenario == "default":
        return [sat.copy() for sat in DEFAULT_SATELLITES]

    if scenario == "coverage":
        # 모든 위성군 사용해 감시 공백 최소화.
        # SpaceEye-T(1), KOMPSAT-7(2): config의 default 5대 중 운영 가치 높은 둘.
        # PlanetScope(2)/ICEYE(5)/Sentinel-1·2·3(6): JSON 카탈로그에서 군집 로드.
        satellites = [
            _default_satellite_entry("SpaceEye-T"),
            _default_satellite_entry("KOMPSAT-7"),
        ]
        priority_by_family = {
            "PlanetScope": 2,
            "ICEYE": 5,
            "Sentinel-1": 6,
            "Sentinel-2": 6,
            "Sentinel-3": 6,
        }
        for filename in (
            "planetscope.json",
            "iceye.json",
            "sentinel-1.json",
            "sentinel-2.json",
            "sentinel-3.json",
        ):
            for satellite in _load_eo_constellation(filename):
                satellite["priority"] = priority_by_family.get(
                    satellite.get("constellation"),
                    satellite["priority"],
                )
                satellites.append(satellite)
        return satellites

    if scenario == "tri-mix":
        satellites = []
        satellites.extend(_load_eo_constellation("iceye.json"))
        satellites.extend(_load_eo_constellation("planetscope.json"))
        spaceeye = _default_spaceeye_entry()
        spaceeye["priority"] = 1
        spaceeye["source"] = "custom-default"
        satellites.append(spaceeye)
        return satellites

    if scenario == "iceye-spaceeye":
        satellites = _load_eo_constellation("iceye.json")
        spaceeye = _default_spaceeye_entry()
        spaceeye["priority"] = 1
        spaceeye["source"] = "custom-default"
        satellites.append(spaceeye)
        return satellites

    if scenario == "planetscope-spaceeye":
        satellites = _load_eo_constellation("planetscope.json")
        spaceeye = _default_spaceeye_entry()
        spaceeye["priority"] = 1
        spaceeye["source"] = "custom-default"
        satellites.append(spaceeye)
        return satellites

    raise ValueError(f"지원하지 않는 위성 시나리오입니다: {scenario}")
