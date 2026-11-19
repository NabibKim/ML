"""리뷰 텍스트 전처리 함수 모음.

학습(02_preprocess.ipynb)과 서비스(app.py)에서 **같은 함수**를 사용해
학습 시점과 추론 시점의 입력 분포를 일치시킨다.
"""
import re
import unicodedata

# Steam BBCode: [h1], [/b], [spoiler], [url=...], [*] 등
RE_BBCODE = re.compile(r"\[/?[a-zA-Z0-9*]{1,10}(?:=[^\]]{0,200})?\]")
RE_HTML = re.compile(r"<[^>]{1,100}>")
RE_URL = re.compile(r"https?://\S+|www\.\S+")
# 박스·블록·도형 문자(아스키아트), 이모지 — 체크박스(☐☑☒, U+2610~2612)는 속성 템플릿에 쓰이므로 유지
RE_SYMBOL = re.compile(
    "[─-▟■-◿☀-☏☓-⛿✀-➿"
    "\U0001F000-\U0001FAFF️​-‍]"
)
# 같은 문자가 3번 이상 반복 → 2번으로 (ㅋㅋㅋㅋ→ㅋㅋ, !!!!→!!). 숫자는 제외(1000시간 보존)
RE_REPEAT = re.compile(r"([^\d\s])\1{2,}")
RE_SPACE = re.compile(r"\s+")

RE_HANGUL = re.compile(r"[가-힣ㄱ-ㅎㅏ-ㅣ]")
RE_LATIN = re.compile(r"[A-Za-z]")


def remove_markup(text: str) -> str:
    """① 마크업 제거: BBCode·HTML 태그·URL."""
    text = RE_BBCODE.sub(" ", text)
    text = RE_HTML.sub(" ", text)
    text = RE_URL.sub(" ", text)
    return text


def normalize_chars(text: str) -> str:
    """② 문자 정규화: 유니코드 NFC, 아스키아트·이모지 제거, 반복 문자 축약, 공백 정리."""
    text = unicodedata.normalize("NFC", text)
    text = RE_SYMBOL.sub(" ", text)
    text = RE_REPEAT.sub(r"\1\1", text)
    text = RE_SPACE.sub(" ", text)
    return text.strip()


def clean_text(text) -> str:
    """전체 정제 파이프라인(①+②). 서비스 입력에도 그대로 적용한다."""
    if not isinstance(text, str):
        return ""
    return normalize_chars(remove_markup(text))


def hangul_ratio(text: str) -> float:
    """문자(한글+영문) 중 한글 비율. 숫자·기호는 분모에서 제외."""
    h = len(RE_HANGUL.findall(text))
    l = len(RE_LATIN.findall(text))
    return h / (h + l) if (h + l) else 0.0


def dedup_key(text: str) -> str:
    """③ 중복 판정용 키: 공백·문장부호를 제거한 소문자 문자열."""
    return re.sub(r"[\s\W_]+", "", text.lower())
