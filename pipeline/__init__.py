"""SIA conflict-monitoring + satellite-tasking pipeline.

실행 순서 (run_pipeline.py 진입점 기준):

  공통/설정
  config.py              · 모든 상수 (파일 경로, COEFS, RISK_THRESHOLDS 등)
  cache.py               · SQLite 캐시 (스크랩 본문 + LLM 응답)
  text_norm.py           · 도시명 정규화 헬퍼
  city_utils.py          · GeoNames 표준명 매핑

  데이터 수집
  gdelt_fetcher.py       · GDELT 일일 zip 다운로드 → parquet

  전처리
  preprocess.py          · clean_gdelt_data + apply_standard_scaling

  Kalman 필터
  kalman_filter.py       · compute_conflict_index → detect_anomalies (top-K)

  LLM 검증
  llm_verification.py    · verify_anomalies_with_llm (Gemini, 캐시 hit)

  중복 제거
  event_dedup.py         · 최근 동일 이벤트 archive 기반 dedup

  위성 스케줄링
  tle_fetcher.py         · TLE 다운로드/캐시
  satellite_catalog.py   · 위성 카탈로그 + ROI
  pass_predictor.py      · 통과 예측 (skyfield)
  weather_checker.py     · 구름량 + 일출/일몰 (현지 좌표 기준)
  schedule_builder.py    · build_schedule → schedule_<date>.json

  대시보드 출력
  daily_json_writer.py   · dashboard/daily_<date>.json + records/daily_<date>.json

  orchestrator
  run_pipeline.py        · 1~7 통합 실행

서브패키지:
  training/   · 일회성 학습 스크립트 (fit_coefs, build_name_map, normalize_gt, update_scaler)
"""
