import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

import pandas as pd
import trafilatura
from trafilatura.metadata import extract_metadata
from google import genai
from google.genai import types

from pipeline.config import (
    API_KEY,
    LLM_MODELS
)
from pipeline import cache

# 병렬 처리 상한 (rate-limit 걸리면 조정)
CITY_CONCURRENCY   = 5    # 동시 처리 도시 수 (≤ Gemini QPS 예산)
SCRAPE_CONCURRENCY = 6    # 도시당 동시 URL 가져오기 수
# 전역 최대 동시 HTTP = CITY_CONCURRENCY × SCRAPE_CONCURRENCY ≈ 18

import os
client = genai.Client(api_key=API_KEY)
MODEL_ID          = os.getenv("GEMINI_MODEL_OVERRIDE") or LLM_MODELS[1]   # 기본: gemini-3.1-flash-lite-preview
FALLBACK_MODEL_ID = os.getenv("GEMINI_FALLBACK_OVERRIDE") or LLM_MODELS[0]   # 기본: gemini-2.5-flash-lite


def scrape_article(url: str) -> tuple:
    """(status, text) 반환. status ∈ {'ok', 'unreachable', 'no_text'}; text는 None일 수 있음."""
    hit = cache.get_scrape(url)
    if hit is not None:
        return hit["status"], hit["text"]

    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            print(f"  [DEBUG] 사이트 접근 차단/응답 없음: {url}")
            cache.set_scrape(url, "unreachable", None)
            return "unreachable", None
        text = trafilatura.extract(downloaded, include_links=False, include_images=False)
        if text:
            cache.set_scrape(url, "ok", text)
            return "ok", text
        cache.set_scrape(url, "no_text", None)
        return "no_text", None
    except Exception as e:
        print(f"  [DEBUG] 스크래핑 오류 발생: {url} ({e})")
        cache.set_scrape(url, "unreachable", None)
        return "unreachable", None
    
def extract_relevant_context(text: str, target_names) -> str:
    """target_names: str or list[str]. 기사 내 어떤 이름이든 매칭되면 문맥 추출."""
    if isinstance(target_names, str):
        target_names = [target_names]
    name_lowers = [n.lower() for n in target_names if n]
    if not name_lowers:
        return None
    text_lower = text.lower()
    if not any(n in text_lower for n in name_lowers):
        return None

    # 문장 단위 분할 (문단 구조에 의존하지 않음)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if not sentences:
        return None

    # 리드: 기사 첫 3문장 (날짜 앵커 확보용)
    lead_indices = set(range(min(3, len(sentences))))

    # 지명 언급 ± 앞뒤 2문장 (상위 2개 언급, 어떤 alias든 매칭)
    found_indices = [i for i, s in enumerate(sentences)
                     if any(n in s.lower() for n in name_lowers)]
    context_indices = set()
    for idx in found_indices[:2]:
        context_indices.update(range(max(0, idx - 2), min(len(sentences), idx + 3)))

    if not context_indices:
        return None

    # 리드와 본문 언급 병합, 원래 순서 유지. 사이 간격이 있으면 [...] 마커 삽입
    all_indices = sorted(lead_indices | context_indices)
    chunks = []
    prev = None
    for i in all_indices:
        if prev is not None and i > prev + 1:
            chunks.append("[...]")
        chunks.append(sentences[i])
        prev = i
    return " ".join(chunks)


PROMPT_VERSION = "v8"


def _window_bounds(target_date: str) -> tuple[str, str]:
    """target_date(YYYYMMDD)로부터 7일 창 [start, end] ISO 날짜 반환."""
    from datetime import datetime, timedelta
    end = datetime.strptime(target_date, "%Y%m%d").date()
    start = end - timedelta(days=6)
    return start.isoformat(), end.isoformat()


def build_gemini_prompt(target_city: str, articles_data: list, target_date: str, aliases: list = None) -> str:
    articles_context = ""
    for idx, data in enumerate(articles_data, 1):
        articles_context += f"\n[Article {idx}]\nContent: {data['text']}\n"

    win_start, _ = _window_bounds(target_date)

    alias_line = ""
    if aliases:
        alias_str = ", ".join(f'"{a}"' for a in aliases)
        alias_line = (f"\nMentions of any of these related place names in the articles refer to "
                      f"{target_city} (same administrative area): {alias_str}.\n")

    prompt = f"""You are verifying whether a physical conflict event happened in {target_city}
on or after {win_start} (no upper bound — events reported in the most recent days are
all in-window, including events the article dates as "today", "yesterday", or the next
day from the report's perspective).{alias_line}

How to read dates in the articles:
- "yesterday", "last night", "this morning", "hours ago", or a day-of-week like
  "Monday"/"Saturday" → in-window.
- "last week/month/year", "anniversary of", specific old years (1979, 2024) → out-of-window.
- If you genuinely cannot tell when the event happened, return AMBIGUOUS.

Choose ONE label for {target_city}:
- SUCCESS       : direct attack, strike, bombing, or active fighting in-window.
- AMBIGUOUS     : only indirect military indicators (sirens, evacuation, troop movement,
                  staging) in-window, OR the event date is unclear.
- DATE_MISMATCH : a conflict event is described but clearly falls outside the window
                  (i.e. before {win_start}).
- DROPPED       : {target_city} appears only as a dateline, or there is no conflict
                  content about it (civilian mourning, protests, political statements,
                  historical mentions all count as DROPPED).
- NO_MENTION    : {target_city} is not meaningfully mentioned in any article.

If articles disagree, the strongest in-window signal wins:
SUCCESS > AMBIGUOUS > DATE_MISMATCH > DROPPED > NO_MENTION.

Write "message" in Korean. Keep {target_city} in English as given.
Output JSON only:
{{"status": "SUCCESS" | "AMBIGUOUS" | "DATE_MISMATCH" | "DROPPED" | "NO_MENTION",
  "message": "<one Korean line: attacker / type / target+impact, or reason>"}}

Articles:
{articles_context}
"""
    return prompt

GEMINI_CALL_DELAY_SEC = 1.5   # 각 호출 전 기본 페이싱
GEMINI_RETRY_WAIT_SEC = 2.5   # 일시 실패 시 초기 대기 (backoff가 이 값에 곱해짐)
GEMINI_MAX_RETRIES    = 1     # 503은 단시간 재시도 효과 없음 → 빠르게 fallback 전환
GEMINI_CALL_TIMEOUT   = int(os.getenv("GEMINI_CALL_TIMEOUT", "30"))    # 단일 API 호출당 초 (pro-preview는 90초 이상 권장)
GEMINI_FINAL_WAIT_SEC = 0     # cooldown 비활성: 지속 503에서 60s 더 기다려도 회복 안 됨

# 재시도할 가치가 있는 일시 실패를 식별하는 부분 문자열.
# 포함: rate limit, 서비스 장애, 관측된 모든 네트워크/프로토콜 오류
# (RemoteProtocolError "Server disconnected", ConnectError DNS 실패, timeout).
_TRANSIENT_MARKERS = (
    '503', 'UNAVAILABLE', '429', 'RESOURCE_EXHAUSTED', '500', 'INTERNAL',
    '504', 'DEADLINE_EXCEEDED', 'timeout', 'Timeout',
    'Server disconnected', 'RemoteProtocolError', 'ConnectError',
    'Connection', 'nodename nor servname', 'Temporary failure',
    'EOF occurred', 'SSL',
)


def _is_transient(exc: Exception) -> bool:
    name = type(exc).__name__
    msg = str(exc)
    return any(m in msg or m in name for m in _TRANSIENT_MARKERS)


def _invoke_model(model_id: str, prompt: str):
    return client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            safety_settings=[
                types.SafetySetting(category=c, threshold='BLOCK_NONE')
                for c in [
                    'HARM_CATEGORY_HARASSMENT', 'HARM_CATEGORY_HATE_SPEECH',
                    'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'HARM_CATEGORY_DANGEROUS_CONTENT'
                ]
            ]
        )
    )


def _try_model_with_retries(model_id: str, prompt: str, label: str):
    """(response, exhausted_transient) 반환. exhausted_transient=True면 모든 재시도가 일시 오류로
    실패한 것 — 다른 모델을 시도하거나 대기할 가치가 있음. None이면 non-transient 포기."""
    for attempt in range(1 + GEMINI_MAX_RETRIES):
        try:
            with ThreadPoolExecutor(max_workers=1) as _tp:
                fut = _tp.submit(_invoke_model, model_id, prompt)
                return fut.result(timeout=GEMINI_CALL_TIMEOUT), False
        except FuturesTimeoutError:
            if attempt == GEMINI_MAX_RETRIES:
                print(f"  [GEMINI {label} EXHAUSTED] timeout after {GEMINI_MAX_RETRIES} retries")
                return None, True
            wait = GEMINI_RETRY_WAIT_SEC + attempt * 0.5
            print(f"  [GEMINI {label} RETRY {attempt+1}/{GEMINI_MAX_RETRIES}] TimeoutError ({GEMINI_CALL_TIMEOUT}s); sleeping {wait:.1f}s")
            time.sleep(wait)
        except Exception as e:
            if not _is_transient(e):
                print(f"  [GEMINI {label} NON-TRANSIENT, giving up] {type(e).__name__}: {e}")
                return None, False
            if attempt == GEMINI_MAX_RETRIES:
                print(f"  [GEMINI {label} EXHAUSTED] {type(e).__name__}: {str(e)[:150]}")
                return None, True
            wait = GEMINI_RETRY_WAIT_SEC + attempt * 0.5
            print(f"  [GEMINI {label} RETRY {attempt+1}/{GEMINI_MAX_RETRIES}] {type(e).__name__}; sleeping {wait:.1f}s")
            time.sleep(wait)
    return None, True


def call_gemini_verification(prompt: str) -> dict:
    """
    견고한 호출: 1차 → fallback 모델 → (60초 대기) → 1차 한 번 더.
    세 단계가 모두 실패하거나 non-transient 오류일 때만 None 반환.
    """
    time.sleep(GEMINI_CALL_DELAY_SEC)

    # 1단계: 1차 모델
    response, exhausted = _try_model_with_retries(MODEL_ID, prompt, "PRIMARY")
    if response is not None:
        pass
    elif not exhausted:
        return None  # non-transient이면 포기
    else:
        # 2단계: fallback 모델 (lite는 보통 덜 혼잡)
        print(f"  [GEMINI FALLBACK] switching to {FALLBACK_MODEL_ID}")
        response, exhausted2 = _try_model_with_retries(FALLBACK_MODEL_ID, prompt, "FALLBACK")
        if response is None and exhausted2 and GEMINI_FINAL_WAIT_SEC > 0:
            # 3단계: 마지막으로 길게 대기 후 1차 한 번 더 (현재 비활성)
            print(f"  [GEMINI COOLDOWN] sleeping {GEMINI_FINAL_WAIT_SEC}s then retrying PRIMARY")
            time.sleep(GEMINI_FINAL_WAIT_SEC)
            response, _ = _try_model_with_retries(MODEL_ID, prompt, "FINAL")
            if response is None:
                print("  [GEMINI GIVE UP] all strategies exhausted, row will be ERROR")
                return None
        elif response is None:
            return None  # fallback에서 non-transient

    # safety / finish-reason / 빈 응답 문제 확인
    raw_text = getattr(response, "text", None) or ""
    if not raw_text:
        try:
            fin = response.candidates[0].finish_reason if response.candidates else None
            sr  = response.candidates[0].safety_ratings if response.candidates else None
            print(f"  [GEMINI ERROR: empty response] finish_reason={fin} safety={sr}")
        except Exception as e:
            print(f"  [GEMINI ERROR: empty response, inspection failed] {e}")
        return None

    json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if not json_match:
        print(f"  [GEMINI ERROR: no JSON in response] preview={raw_text[:200]!r}")
        return None

    try:
        return json.loads(json_match.group())
    except json.JSONDecodeError as e:
        print(f"  [GEMINI ERROR: JSON parse] {e}, preview={json_match.group()[:200]!r}")
        return None

#
# llm_status 값 (이상치 행마다 설정):
#   UNVERIFIED        , 미평가 (top-K 밖)
#   ARTICLE_UNREACHABLE, 모든 URL 접근 실패 또는 본문 없음 (precision 분모에서 제외)
#   NO_MENTION        , URL은 받았으나 어느 기사도 대상 도시를 언급 안 함 (FP: 필터 lead 부정확)
#   SUCCESS           , LLM이 직접 분쟁 확인 (TP)
#   AMBIGUOUS         , LLM이 간접 긴장 확인 (TP)
#   DATE_MISMATCH     , LLM이 다른 날짜 사건으로 판정 (FP)
#   DROPPED           , LLM이 기각 (FP)
#   ERROR             , LLM API 실패 (precision 분모에서 제외)
#
def _scrape_and_extract(url: str, target_names: list) -> tuple:
    """스레드 워커: fetch + 언급 문맥 추출. (url, status, extracted_or_None, mention_count) 반환.
    mention_count = 모든 target_names 합산 (alias 함께 카운트)."""
    status, text = scrape_article(url)
    if status != 'ok' or not text:
        return (url, status, None, 0)
    text_lower = text.lower()
    mention_count = sum(text_lower.count(n.lower()) for n in target_names if n)
    extracted = extract_relevant_context(text, target_names)
    return (url, status, extracted, mention_count)


def _process_city(row: pd.Series, filtered_df: pd.DataFrame, url_df: pd.DataFrame, target_date: str) -> dict:
    """한 도시의 scrape+LLM 파이프라인 실행. 메인 스레드 기록용 result dict 반환.
    row['city']는 standard_name. (standard_name, date)에 기여한 원본 FullName들을
    filtered_df에서 가져와, LLM이 그중 어느 것이든 기사에서 찾는다."""
    target_city = row['city']  # standard_name (예: 'Tehran|IR')
    # standard_name에 country suffix가 붙어있어 사람·LLM 출력엔 떼고 사용
    display_city = target_city.split('|', 1)[0]
    logs = [f"\nTarget: {target_city} | Z-Score: {row['std_innov']:.1f}"]

    city_events = filtered_df[(filtered_df['standard_name'] == target_city) & (filtered_df['date'] == target_date)]
    # 분쟁 격렬도 ↑ 우선, 동률 시 매체수 ↑, LLM 전달 3개가 가장 폭력적인 이벤트의 기사가 되도록
    potential_events = city_events.sort_values(
        ['GoldsteinScale', 'weighted_index'], ascending=[True, False]
    ).head(30)
    merged_events = potential_events.merge(url_df[['GLOBALEVENTID', 'SOURCEURL']], on='GLOBALEVENTID', how='inner')
    unique_urls = merged_events['SOURCEURL'].unique().tolist()

    # 표시명: NumMentions 합이 가장 큰 FullName (대표 원본명).
    # 예: standard_name "Al Farwānīyah" → 표시 "Kuwait International Airport".
    if not city_events.empty and 'NumMentions' in city_events.columns:
        name_weights = city_events.groupby('ActionGeo_FullName')['NumMentions'].sum().sort_values(ascending=False)
        display_name = name_weights.index[0] if not name_weights.empty else display_city
    else:
        display_name = display_city

    # 기사 매칭은 raw 도시명으로, standard_name suffix('|IR')는 본문에 안 나옴
    orig_names = city_events['ActionGeo_FullName'].dropna().astype(str).unique().tolist()
    target_names = list(dict.fromkeys([display_city] + orig_names))
    aliases = [n for n in target_names if n != display_city]

    scrape_stats = {'ok': 0, 'unreachable': 0, 'no_text': 0, 'no_mention': 0, 'matched': 0}
    alias_note = f" (aliases: {aliases})" if aliases else ""
    logs.append(f"  -> Scanning {len(unique_urls)} sources in parallel for '{target_city}'{alias_note}...")

    # 병렬 scrape + extract. 결정성: 전부 submit 후 원래 순서로 순회.
    results_by_url = {}
    if unique_urls:
        with ThreadPoolExecutor(max_workers=SCRAPE_CONCURRENCY) as ex:
            futs = {ex.submit(_scrape_and_extract, u, target_names): u for u in unique_urls}
            for fut in as_completed(futs):
                try:
                    url, status, extracted, mc = fut.result()
                    results_by_url[url] = (status, extracted, mc)
                except Exception as e:
                    results_by_url[futs[fut]] = ('unreachable', None, 0)
                    logs.append(f"     [ERROR] scrape future failed: {e}")

    # 원래 URL 순서로 처음 3개 매칭 수집 (결정적).
    # 동일/유사 본문 중복 제거 (wire 기사는 그대로 재배포됨).
    import hashlib
    def _norm_hash(s: str) -> str:
        # 공백/개행 정규화 → 동일 문장 재배포 기사(동일 AP/Reuters wire)를 동일 해시로
        return hashlib.md5(' '.join(s.split()).encode('utf-8')).hexdigest()

    # 추출된 모든 매칭 기사를 보관 (text + mention_count + url), 처음 3개는 LLM 입력,
    # 나머지는 DATE_MISMATCH/DROPPED fallback 시 4번째 후보로 사용.
    matched_articles, articles_data, valid_urls = [], [], []
    seen_hashes = set()
    scrape_stats['duplicate'] = 0
    for url in unique_urls:
        status, extracted, mc = results_by_url.get(url, ('unreachable', None, 0))
        if status == 'ok':
            scrape_stats['ok'] += 1
            if extracted:
                scrape_stats['matched'] += 1
                h = _norm_hash(extracted)
                if h in seen_hashes:
                    scrape_stats['duplicate'] += 1
                    logs.append(f"     [DUP] skip repost: {url[:60]}...")
                    continue
                seen_hashes.add(h)
                matched_articles.append({'text': extracted, 'mc': mc, 'url': url})
                if len(articles_data) < 3:
                    articles_data.append({'text': extracted, 'mc': mc, 'url': url})
                    valid_urls.append(url)
                    logs.append(f"     [MATCH] ({mc} mentions): {url[:60]}...")
            else:
                scrape_stats['no_mention'] += 1
        else:
            scrape_stats[status] = scrape_stats.get(status, 0) + 1

    result = {
        'source_urls':  valid_urls,
        'scrape_stats': json.dumps(scrape_stats),
        'llm_status':   None,
        'llm_report':   None,
        'is_anomaly':   True,
        'logs':         logs,
    }

    # 유효 기사 없음 → 실패 원인 구분
    if not articles_data:
        if scrape_stats['ok'] == 0 and (scrape_stats['unreachable'] + scrape_stats['no_text']) > 0:
            final_status = 'ARTICLE_UNREACHABLE'
            msg = f"모든 URL 접근 불가/본문 없음 (unreachable={scrape_stats['unreachable']}, no_text={scrape_stats['no_text']})"
        else:
            final_status = 'NO_MENTION'
            msg = f"기사는 받았으나 '{display_city}' 언급 없음 (ok={scrape_stats['ok']}, no_mention={scrape_stats['no_mention']})"
        result['llm_status'] = final_status
        result['llm_report'] = json.dumps({"Summary": msg}, ensure_ascii=False)
        result['is_anomaly'] = False
        return result

    # 캐시와 함께 LLM 호출
    article_texts = [d['text'] for d in articles_data]
    # 캐시 키는 display_city (suffix 제거) 사용, standard_name 변경(country suffix 추가)
    # 이전에도 LLM은 'Tehran' 텍스트만 봤으므로 캐시 키도 동일하게 유지하면 hit률 보존.
    cache_key = cache.make_llm_key(PROMPT_VERSION, display_city, target_date, article_texts, MODEL_ID)
    llm_result = cache.get_llm(cache_key)
    # 하위호환: 모델명 없던 레거시 캐시 폴백 (주 모델이 2.5-flash였을 때 생성)
    if llm_result is None and MODEL_ID == "gemini-2.5-flash":
        legacy_key = cache.make_llm_key(PROMPT_VERSION, display_city, target_date, article_texts, "")
        llm_result = cache.get_llm(legacy_key)
    if llm_result is not None:
        logs.append(f"  -> [CACHE HIT] Gemini response reused for '{target_city}'")
    else:
        logs.append(f"  -> Sending {len(articles_data)} articles to Gemini for '{target_city}'...")
        try:
            llm_result = call_gemini_verification(build_gemini_prompt(display_city, articles_data, target_date, aliases))
        except Exception as e:
            logs.append(f"  -> [LLM ERROR] {e}")
            llm_result = None
        if llm_result is not None:
            cache.set_llm(cache_key, llm_result)

    if not llm_result:
        result['llm_status'] = 'ERROR'
        return result

    # Fallback: status가 DATE_MISMATCH/DROPPED/AMBIGUOUS이고 4번째 매칭 기사 있으면 1개 더 보고 재호출
    from pipeline.config import BASELINE_NO_PERF as _NO_PERF
    fallback_status = llm_result.get('status')
    if not _NO_PERF and fallback_status in ('DATE_MISMATCH', 'DROPPED', 'AMBIGUOUS') and len(matched_articles) > len(articles_data):
        next_article = matched_articles[len(articles_data)]
        articles_data.append(next_article)
        valid_urls.append(next_article['url'])
        logs.append(f"  -> [FALLBACK] {fallback_status} → 4번째 기사 추가 ({next_article['mc']} mentions): {next_article['url'][:60]}...")
        article_texts2 = [d['text'] for d in articles_data]
        cache_key2 = cache.make_llm_key(PROMPT_VERSION, display_city, target_date, article_texts2, MODEL_ID)
        llm_result2 = cache.get_llm(cache_key2)
        if llm_result2 is None:
            try:
                llm_result2 = call_gemini_verification(build_gemini_prompt(display_city, articles_data, target_date, aliases))
            except Exception as e:
                logs.append(f"  -> [FALLBACK LLM ERROR] {e}")
                llm_result2 = None
            if llm_result2 is not None:
                cache.set_llm(cache_key2, llm_result2)
        if llm_result2 is not None:
            llm_result = llm_result2
            logs.append(f"  -> [FALLBACK RESULT] {llm_result.get('status')}")
            # 4번째 기사(fallback으로 추가)를 맨 앞으로, 결과를 흔든 후속 입력
            if valid_urls:
                valid_urls = [valid_urls[-1]] + valid_urls[:-1]

    result['source_urls'] = valid_urls

    geo_lat  = potential_events['ActionGeo_Lat'].iloc[0]  if not potential_events.empty else None
    geo_long = potential_events['ActionGeo_Long'].iloc[0] if not potential_events.empty else None
    status   = llm_result.get('status', 'ERROR')

    final_report = {
        "SQLDATE":      target_date,
        "City":         display_name,   # 원본 FullName (대표), 예: "Kuwait International Airport"
        "StandardName": target_city,    # Kalman 집계 단위, 예: "Al Farwānīyah"
        "Aliases":      aliases,
        "Latitude":     float(geo_lat)  if geo_lat  is not None else None,
        "Longitude":    float(geo_long) if geo_long is not None else None,
        "Summary":      llm_result.get('message', '메시지 없음'),
    }
    result['llm_status'] = status
    result['llm_report'] = json.dumps(final_report, ensure_ascii=False)
    result['is_anomaly'] = status in ['SUCCESS', 'AMBIGUOUS']
    return result


def verify_anomalies_with_llm(anomalies: pd.DataFrame, filtered_df: pd.DataFrame, url_df: pd.DataFrame, target_date: str, top_k: int = 20) -> pd.DataFrame:
    print(f"\n[TRACK 2] LLM Verification Started: {target_date} (top_k={top_k})")

    today_anomalies = anomalies[(anomalies['date'] == target_date) & (anomalies['is_anomaly'] == True)]
    top_targets = today_anomalies.sort_values('std_innov', ascending=False).head(top_k)

    anomalies['llm_status']   = 'UNVERIFIED'
    anomalies['llm_report']   = None
    anomalies['source_urls']  = None
    anomalies['scrape_stats'] = None

    if top_targets.empty:
        return anomalies

    rows = [(idx, row) for idx, row in top_targets.iterrows()]

    with ThreadPoolExecutor(max_workers=CITY_CONCURRENCY) as ex:
        fut_to_idx = {ex.submit(_process_city, row, filtered_df, url_df, target_date): idx for idx, row in rows}
        for fut in as_completed(fut_to_idx):
            idx = fut_to_idx[fut]
            try:
                result = fut.result()
            except Exception as e:
                print(f"[ERROR] city idx={idx} processing failed: {e}")
                continue

            # 이 도시의 버퍼된 로그 출력 (스레드 간 섞임 방지)
            for line in result['logs']:
                print(line)

            anomalies.at[idx, 'source_urls']  = result['source_urls']
            anomalies.at[idx, 'scrape_stats'] = result['scrape_stats']
            if result['llm_status'] is not None:
                anomalies.at[idx, 'llm_status'] = result['llm_status']
            if result['llm_report'] is not None:
                anomalies.at[idx, 'llm_report'] = result['llm_report']
            if not result['is_anomaly']:
                anomalies.at[idx, 'is_anomaly'] = False

    return anomalies