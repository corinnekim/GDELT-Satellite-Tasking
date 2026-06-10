"""
공통 이름 정규화.

GDELT, GT, LLM 기사 매칭 등 모든 이름 비교에서 동일한 정규화를 사용한다.
소문자 + 알파벳/숫자 외 모든 문자 제거 (공백, 하이픈, 아포스트로피, 특수 apostrophe,
물음표, 느낌표, 마침표, 쉼표, 세미콜론 등).

예:
  "Ma'alot-Tarshiha"   → "maalottarshiha"
  "Al-Qaim"            → "alqaim"
  "Bay Of Haifa"       → "bayofhaifa"
  "H̱anni’el"          → "hannıel"  (결합문자 자체는 유지됨 - 필요시 NFKD 추가)
"""
import re
import unicodedata


def _norm(s: str) -> str:
    if not isinstance(s, str):
        return ""
    # NFKD 분해 후 결합기호 제거 (ā → a, ‘ → 제거)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s