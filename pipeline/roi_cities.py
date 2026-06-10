"""ROI 도시 좌표, Level 1 탐지 대상 + 핵심 인프라.

용도:
1. 탐지된 도시명이 ROI에 등재된 경우 ROI 좌표를 우선 사용 (좌표 정확도 보강)
2. pass_predictor / schedule_builder 단독 실행 시 fallback (cities=None)

실제 운영 스케줄링은 매일 LLM이 검증한 분쟁 도시들로 동적으로 정해짐.
이 리스트는 강제 대상이 아님.
"""

ROI_CITIES = {
    "Isfahan":      {"lat": 32.6546, "lon": 51.6680},
    "Natanz":       {"lat": 33.5130, "lon": 51.9220},
    "Bushehr":      {"lat": 28.9684, "lon": 50.8385},
    "Tehran":       {"lat": 35.6892, "lon": 51.3890},
    "Tabriz":       {"lat": 38.0800, "lon": 46.2919},
    "Kharg Island": {"lat": 29.2333, "lon": 50.3167},
    "Dimona":       {"lat": 31.0700, "lon": 35.2100},
    "Beirut":       {"lat": 33.8938, "lon": 35.5018},
    "Baghdad":      {"lat": 33.3152, "lon": 44.3661},
    "Gaza":         {"lat": 31.5000, "lon": 34.4667},
    "Tel Aviv":     {"lat": 32.0853, "lon": 34.7818},
    "Minab":        {"lat": 27.1064, "lon": 57.0850},
    "Ras Laffan":   {"lat": 25.9300, "lon": 51.5300},
    "Fujairah":     {"lat": 25.1288, "lon": 56.3265},
    "Dubai":        {"lat": 25.2048, "lon": 55.2708},
}
