// JSON 데이터 가져오기

export interface GDELTData {
  date: string;
  generated_at: string;
  scenario_name: string;
  alert_count: number;
  alerts: Array<{
    city: string;
    risk_level: number;
    risk_label: string;
    conflict_index: number;
    innovation_z: number;
    guide: string;
    events: number;
    lat: number;
    lon: number;
    country_code: string;
    llm_confidence: number;
    llm_reason: string;
  }>;
}

export interface ScheduleData {
  generated_utc: string;
  mode: string;
  satellite_scenario: string;
  prediction_hours: number;
  satellites_tracked: number;
  total_passes: number;
  shootable_passes: number;
  recommendations: Array<{
    satellite: string;
    norad_id: number;
    sensor_type: string;
    resolution_m: number;
    priority: number;
    city: string;
    lat: number;
    lon: number;
    pass_time_utc: string;
    max_elevation_deg: number;
    cloud_cover_pct: number;
    cloud_status: string;
    daylight: boolean;
    shootable: boolean;
    quality_score: number;
    action_priority_label: string;
    recommendation_reason: string;
  }>;
}

// GDELT 데이터 가져오기
export async function fetchGDELTData(date: string): Promise<GDELTData | null> {
  try {
    const response = await fetch(`/data/${date}.json`);
    if (!response.ok) return null;
    return await response.json();
  } catch (error) {
    console.error('GDELT 데이터 로드 실패:', error);
    return null;
  }
}

// 위성 스케줄 데이터 가져오기
export async function fetchScheduleData(date: string): Promise<ScheduleData | null> {
  try {
    const response = await fetch(`/data/schedule_${date}.json`);
    if (!response.ok) return null;
    return await response.json();
  } catch (error) {
    console.error('스케줄 데이터 로드 실패:', error);
    return null;
  }
}