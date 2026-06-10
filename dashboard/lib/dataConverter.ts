// GDELT 데이터를 ROI 형식으로 변환

import { GDELTData, ScheduleData } from './api';

export interface ROI {
  id: string;
  name: string;
  location: string;
  lat: number;
  lng: number;
  priority: 'high' | 'medium' | 'low';
  riskLevel: number;
  riskLabel: string;
  conflictIndex: number;
  events: number;
  guide: string;
  llmReason: string;
  schedules?: Array<{
    satellite: string;
    passTime: string;
    qualityScore: number;
    actionLabel: string;
    reason: string;
  }>;
}

// GDELT 데이터를 ROI로 변환
export function convertGDELTToROIs(gdeltData: GDELTData, scheduleData?: ScheduleData): ROI[] {
  return gdeltData.alerts.map((alert, index) => {
    // 해당 도시의 위성 스케줄 찾기
    const citySchedules = scheduleData?.recommendations
      .filter(rec => rec.city === alert.city)
      .slice(0, 3) // 상위 3개만
      .map(rec => ({
        satellite: rec.satellite,
        passTime: rec.pass_time_utc,
        qualityScore: rec.quality_score,
        actionLabel: rec.action_priority_label,
        reason: rec.recommendation_reason,
      }));

    // Priority 결정
    let priority: 'high' | 'medium' | 'low' = 'low';
    if (alert.risk_level === 3) priority = 'high';
    else if (alert.risk_level === 2) priority = 'medium';

    return {
      id: `roi-${index + 1}`,
      name: alert.city,
      location: alert.country_code,
      lat: alert.lat,
      lng: alert.lon,
      priority,
      riskLevel: alert.risk_level,
      riskLabel: alert.risk_label,
      conflictIndex: alert.conflict_index,
      events: alert.events,
      guide: alert.guide,
      llmReason: alert.llm_reason,
      schedules: citySchedules,
    };
  });
}

// 날짜 포맷 변환 (YYYYMMDD → YYYY-MM-DD)
export function formatDateForDisplay(dateStr: string): string {
  if (dateStr.length !== 8) return dateStr;
  return `${dateStr.slice(0, 4)}-${dateStr.slice(4, 6)}-${dateStr.slice(6, 8)}`;
}