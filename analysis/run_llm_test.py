import argparse
import pandas as pd
import json

from pipeline.config import MAIN_PATH, URL_PATH
from pipeline.kalman_filter import compute_conflict_index, detect_anomalies
from pipeline.llm_verification import verify_anomalies_with_llm
def test_llm_pipeline(target_date: str):
    print("\n" + "="*60)
    print(f" STARTING SIA LLM TEST PIPELINE: {target_date}")
    print("="*60)
    
    print("\n[SYSTEM] Loading Datasets...")
    try:
        raw_df = pd.read_parquet(MAIN_PATH)
        url_df = pd.read_parquet(URL_PATH)
        raw_df['date'] = raw_df['SQLDATE'].astype(str).str[:8]
    except Exception as e:
        print(f"[ERROR] Data load failed: {e}")
        return

    print("\n[SYSTEM] Running Kalman Filter (Track 1)...")
    city_daily, raw_filtered = compute_conflict_index(raw_df) 
    anomalies_df = detect_anomalies(city_daily, target_date)
    
    if anomalies_df.empty:
        print(f"\n[INFO] No anomalies detected for {target_date}.")
        return

    print("\n[SYSTEM] Initiating LLM Verification (Track 2)...")
    final_results = verify_anomalies_with_llm(anomalies_df, raw_filtered, url_df, target_date)

    print_test_report(final_results, target_date)

def print_test_report(df: pd.DataFrame, date: str):
    print("\n" + "="*60)
    print(f" FINAL VERIFICATION REPORT ({date})")
    print("="*60)
    
    # 시스템 설계에 맞춘 모든 유효 상태값 리스트화
    valid_statuses = ['SUCCESS', 'AMBIGUOUS', 'DATE_MISMATCH', 'DROPPED',
                      'ARTICLE_UNREACHABLE', 'NO_MENTION', 'ERROR']
    test_alerts = df[df['llm_status'].isin(valid_statuses)].sort_values('std_innov', ascending=False)
    
    for _, row in test_alerts.iterrows():
        status = row['llm_status']
        print(f"\nCITY: {row['city']} | Z-Score {row['std_innov']:.1f}")
        print(f"STATUS: {status}")
        
        report_data = json.loads(row['llm_report']) if row['llm_report'] else {}
        
        # 상태에 따른 출력 형식 분기
        if status in ['SUCCESS', 'AMBIGUOUS']:
            # 유효한 이상 징후일 경우 JSON 전체 출력
            print(json.dumps(report_data, ensure_ascii=False, indent=2))
        else:
            # 기각 및 실패 상태일 경우 LLM의 거절 사유(혹은 파이썬의 에러 메시지)만 1줄 출력
            print(f"메시지: {report_data.get('Summary', '내용 없음')}")

        if row.get('source_urls'):
            print("[기사 URL]")
            for u in row['source_urls']:
                print(f"  - {u}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, required=True, help="YYYYMMDD")
    args = parser.parse_args()
    
    test_llm_pipeline(args.date)