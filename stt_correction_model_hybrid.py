import os
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# -------------------------------
# 0. OpenAI 설정
# -------------------------------
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
api_key = os.getenv("OPENAI_API_KEY") # 실제 API 키로 교체
client = OpenAI(api_key=api_key) if api_key else None


def _generate_text(prompt, model_name=None):
    if client is None:
        return None

    try:
        res = client.responses.create(
            model=model_name or OPENAI_MODEL,
            input=prompt,
        )
        text = (getattr(res, "output_text", "") or "").strip()
        return text or None
    except Exception:
        return None


# -------------------------------
# 1. 데이터 로드
# -------------------------------
df = pd.read_csv("menu.csv", encoding="cp949")
menu_list = df["상품명"].tolist()
base_menu_list = sorted(
    {
        menu[4:].strip() if menu.startswith("ICE ") else menu.strip()
        for menu in menu_list
    }
)
menu_terms = sorted(
    {
        token
        for menu in menu_list
        for token in menu.split()
        if token
    }
)


# -------------------------------
# 2. 임베딩 모델
# -------------------------------
model = SentenceTransformer(
    "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
)

menu_embeddings = model.encode(menu_list)
term_embeddings = model.encode(menu_terms)

PROTECTED_WORDS = {"ICE", "HOT"}
GENERIC_WORDS = {
    "커피", "카피", "차", "티", "디저트", "메뉴", "종류", "전체", "목록",
    "추천", "보여줘", "보여", "뭐", "있어", "없는", "있는", "들어간",
    "들어가는", "안", "빼고", "제외", "없이", "카페인", "디카페인",
    "디카페", "우유", "대두", "달걀", "밀", "복숭아", "아황산류",
    "돼지고기", "음료", "알레르기",
}
CONDITION_HINTS = {
    "메뉴", "종류", "전체", "목록", "추천", "보여", "뭐", "있어", "알려줘",
    "없는", "있는", "들어간", "안", "빼고", "제외", "없이", "알레르기",
}
NOISE_WORDS = {
    "한잔", "한", "잔", "주세요", "주세용", "부탁해", "부탁", "로", "으로", "좀",
    "먹고", "싶어", "싶어요", "원해", "원해요", "줘", "주세요요", "음", "어", "저기"
}
DESSERT_HINTS = {"케익", "케이크", "케잌", "쿠키", "브레드", "빵"}
TEA_HINTS = {"차", "아이스티", "캐모마일", "페퍼민트", "레몬차", "유자차", "복숭아"}
COFFEE_HINTS = {"아메리카노", "라떼", "모카", "마끼아또", "에스프레소", "커피"}


# -------------------------------
# 3. 텍스트 정규화
# -------------------------------
def normalize_text(query):
    query = query.replace("아이스티", "__ICED_TEA__")
    query = query.replace("핫초코", "__HOT_CHOCO__")
    query = query.replace("아이스", "ICE")
    query = query.replace("아잇", "ICE")
    query = query.replace("아이수", "ICE")
    query = query.replace("차가운", "ICE")
    query = query.replace("시원한", "ICE")
    query = query.replace("뜨거운", "HOT")
    query = query.replace("따뜻한", "HOT")
    query = query.replace("핫", "HOT")

    query = query.replace("__ICED_TEA__", "아이스티")
    query = query.replace("__HOT_CHOCO__", "핫초코")
    return query.strip()


def strip_temp_tokens(text):
    return " ".join(
        token for token in text.split()
        if token not in {"ICE", "HOT"}
    )


def compact_text(text):
    return "".join(text.split())


def is_condition_query(query):
    return any(token in query for token in CONDITION_HINTS)


def is_generic_category_query(query):
    compact = compact_text(strip_temp_tokens(query))
    return compact in {"커피", "차", "디저트"}


def mentions_decaf(text):
    return any(token in text for token in ["디카페인", "디카페", "디카폐", "디카패"])


def has_hint(text, hints):
    return any(token in text for token in hints)


def has_pastry_signal(text):
    compact = compact_text(text)
    return any(token in compact for token in DESSERT_HINTS)


def has_tea_signal(text):
    compact = compact_text(text)
    if any(token in text for token in TEA_HINTS):
        return True
    if "아이스티" in compact:
        return True
    if compact.endswith("티") and not compact.endswith("라티"):
        return True
    if compact.endswith("차"):
        return True
    return False


def has_standalone_token(text, token):
    compact = compact_text(text)
    return token in text.split() or compact == token


def collapse_repeated_tokens(text):
    tokens = text.split()
    collapsed = []
    for token in tokens:
        if collapsed and collapsed[-1] == token:
            continue
        collapsed.append(token)
    return " ".join(collapsed)


def find_exact_menu_match(query):
    requested_temp = extract_requested_temp(query)
    compact_query = compact_text(strip_temp_tokens(query))
    if not compact_query:
        return None

    matched = []
    for menu in menu_list:
        base_name = menu[4:].strip() if menu.startswith("ICE ") else menu.strip()
        base_compact = compact_text(base_name)
        full_compact = compact_text(menu)
        if compact_query == full_compact or compact_query == base_compact:
            matched.append(menu)
        elif compact_query and base_compact in compact_query:
            if len(base_compact) / max(len(compact_query), 1) >= 0.72:
                matched.append(menu)
        elif compact_query and compact_query in base_compact:
            if len(compact_query) / max(len(base_compact), 1) >= 0.72:
                matched.append(menu)

    if not matched:
        return None

    ranked = []
    wants_decaf = mentions_decaf(query)
    for menu in matched:
        base_name = menu[4:].strip() if menu.startswith("ICE ") else menu.strip()
        base_compact = compact_text(base_name)
        full_compact = compact_text(menu)
        score = len(base_name)
        if compact_query == full_compact or compact_query == base_compact:
            score += 100
        elif compact_query and base_compact in compact_query:
            score += 55
        elif compact_query and compact_query in base_compact:
            score += 35
        if requested_temp == "ICE":
            score += 20 if menu.startswith("ICE ") else -10
        elif requested_temp == "HOT":
            score += 20 if not menu.startswith("ICE ") else -10
        else:
            score += -6 if menu.startswith("ICE ") else 0
        if wants_decaf:
            score += 20 if "디카페인" in menu else -10
        elif "디카페인" in menu:
            score -= 8
        ranked.append((score, menu))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked[0][1]


def extract_requested_temp(query):
    upper_query = query.upper()
    compact = compact_text(query)
    if "아이스티" in compact or "아이스뜨" in compact or "아이스 뜨" in query:
        return None
    if "ICE" in upper_query or any(token in query for token in ["아이스", "아잇", "아이쓰", "아이수", "차가운", "시원한"]):
        return "ICE"
    if has_standalone_token(query, "아아"):
        return "ICE"
    if any(token in query for token in ["뜨거운", "따뜻한", "핫"]) or has_standalone_token(query, "뜨아"):
        return "HOT"
    return None


def preserve_requested_temp(original_query, corrected_query):
    requested_temp = extract_requested_temp(original_query)
    corrected = corrected_query

    if requested_temp is None:
        corrected = corrected.replace("ICE ", "")
        corrected = corrected.replace("HOT ", "")
        return " ".join(corrected.split())

    if requested_temp == "ICE" and not corrected.startswith("ICE "):
        corrected = f"ICE {corrected}"
    return " ".join(corrected.split())


def should_use_llm(query):
    if client is None:
        return False
    if is_condition_query(query) or is_generic_category_query(query):
        return False
    if find_exact_menu_match(query):
        return False
    return True


def apply_query_level_correction(query):
    if not query:
        return query

    if is_condition_query(query) or is_generic_category_query(query):
        return query

    temp_prefix = "ICE " if "ICE" in query.split() else ""
    wants_decaf = mentions_decaf(query)
    decaf_prefix = "디카페인 " if wants_decaf else ""
    base_query = (
        strip_temp_tokens(query)
        .replace("디카페인", " ")
        .replace("디카페", " ")
        .replace("디카폐", " ")
        .strip()
    )
    if not base_query:
        return query

    exact_lookup = {
        compact_text(name): name
        for name in base_menu_list
    }
    compact_query = compact_text(base_query)
    query_has_dessert = has_pastry_signal(base_query)
    query_has_tea = has_tea_signal(base_query)
    query_has_coffee = has_hint(base_query, COFFEE_HINTS)

    if len(compact_query) <= 4 and not (query_has_dessert or query_has_tea or query_has_coffee or wants_decaf):
        return query

    scored = []
    for base_name in base_menu_list:
        compact_name = compact_text(base_name)
        name_has_dessert = has_pastry_signal(base_name)
        name_has_tea = has_tea_signal(base_name)
        name_has_coffee = has_hint(base_name, COFFEE_HINTS)
        score = max(
            fuzz.WRatio(base_query, base_name),
            fuzz.WRatio(compact_query, compact_name),
            fuzz.token_set_ratio(base_query, base_name),
        )

        if compact_query == compact_name:
            score += 35
        elif compact_query and compact_name in compact_query:
            score += 24 + min(len(compact_name), 12)
        elif compact_query and compact_query in compact_name:
            score += 16

        token_overlap = 0
        for token in base_name.split():
            compact_token = compact_text(token)
            if compact_token and compact_token in compact_query:
                token_overlap += 1
            elif compact_query and compact_query in compact_token:
                token_overlap += 1
        score += token_overlap * 9

        if query_has_dessert:
            if name_has_dessert:
                score += 32
            else:
                score -= 14

        if query_has_tea:
            if name_has_tea:
                score += 12
            elif name_has_coffee:
                score -= 8

        if query_has_coffee:
            if name_has_coffee:
                score += 10

        if wants_decaf:
            score += 18 if "디카페인" in base_name else -10

        if len(compact_name) >= max(len(compact_query) - 1, 1):
            score += 4
        elif len(compact_name) + 2 < len(compact_query):
            score -= 6

        scored.append((float(score), base_name))

    scored.sort(key=lambda x: x[0], reverse=True)
    threshold = 68 if query_has_dessert else 74
    if scored and scored[0][0] >= threshold:
        return f"{temp_prefix}{decaf_prefix}{scored[0][1]}".strip()

    return query


# -------------------------------
# 4. LLM 1차 보정
# -------------------------------
def llm_correct(query):
    if not should_use_llm(query):
        return query

    menu_preview = ", ".join(menu_list[:40])

    prompt = f"""
다음 문장은 음성인식 오인식 결과입니다.
카페 메뉴 주문 문장으로 정확하게 교정하세요.
아래 메뉴 목록에 맞게 오타를 교정하고, 가장 가까운 메뉴명으로 바꿔주세요.
규칙:
- HOT/ICE 토큰은 절대 삭제하지 마세요.
- 약어를 자연스럽게 풀어주세요. (예: 아아->ICE 아메리카노, 뜨아->아메리카노)
- 메뉴 카테고리/메뉴 키워드 오타도 교정하세요. (예: 디져트->디저트, 라때->라떼)
- 질문에 대한 답변, 추천, 설명, 메뉴 나열을 하지 말고 입력 문장만 교정하세요.
- 메뉴명이나 주문 문장 자체를 교정하는 경우에만 수정하세요.
- 알레르기/카페인/카테고리 질문은 표현만 다듬고 의미를 바꾸지 마세요.
- 디저트, 커피, 차 같은 카테고리 단어를 임의의 개별 메뉴로 바꾸지 마세요.
- 설명 없이 교정된 문장 한 줄만 출력하세요.

메뉴 목록(일부):
{menu_preview}

입력: {query}
출력:
"""

    text = _generate_text(prompt)
    return text if text else query


# -------------------------------
# 5. 최종 보정 (LLM + fuzzy + SBERT)
# -------------------------------
def correct_text(query):
    original_query = query
    query = normalize_text(query)
    exact_menu = find_exact_menu_match(query)
    if exact_menu:
        return preserve_requested_temp(original_query, exact_menu)

    direct_corrected = apply_query_level_correction(query)
    if direct_corrected != query:
        query = direct_corrected
    else:
        query = llm_correct(query)
        query = apply_query_level_correction(query)

    if has_pastry_signal(original_query) and not has_pastry_signal(query):
        pastry_candidate = apply_query_level_correction(normalize_text(original_query))
        if pastry_candidate != normalize_text(original_query) and has_pastry_signal(pastry_candidate):
            query = pastry_candidate

    query = collapse_repeated_tokens(query)
    exact_menu = find_exact_menu_match(query)
    if exact_menu:
        return preserve_requested_temp(original_query, exact_menu)

    if has_pastry_signal(query):
        corrected = preserve_requested_temp(original_query, query)
        corrected = collapse_repeated_tokens(corrected)
        return " ".join(corrected.split())

    query = preserve_requested_temp(original_query, query)

    words = query.split()
    corrected_words = []

    for word in words:
        if word in NOISE_WORDS:
            continue

        if word in PROTECTED_WORDS:
            corrected_words.append(word)
            continue

        if word in menu_terms:
            corrected_words.append(word)
            continue

        if word in GENERIC_WORDS:
            corrected_words.append(word)
            continue

        match = process.extractOne(word, menu_terms, scorer=fuzz.WRatio)
        if not match:
            corrected_words.append(word)
            continue

        fuzz_match, fuzz_score, _ = match

        word_emb = model.encode([word])[0]
        scores = [
            np.dot(word_emb, term_emb)
            / (np.linalg.norm(word_emb) * np.linalg.norm(term_emb))
            for term_emb in term_embeddings
        ]

        embed_match = menu_terms[int(np.argmax(scores))]
        embed_score = float(np.max(scores))

        if fuzz_score >= 82:
            corrected_words.append(fuzz_match)
        elif fuzz_score >= 58 and embed_score >= 0.45 and fuzz_match == embed_match:
            corrected_words.append(fuzz_match)
        elif embed_score >= 0.72:
            corrected_words.append(embed_match)
        else:
            corrected_words.append(word)

    corrected = " ".join(corrected_words)
    corrected = corrected.replace(" HOT ", " ")
    if corrected.startswith("HOT "):
        corrected = corrected[4:]
    corrected = collapse_repeated_tokens(corrected)
    if "핫초코" in compact_text(original_query) or ("핫" in original_query and "초코" in original_query):
        corrected = corrected.replace("ICE 초코", "초코")
    corrected = preserve_requested_temp(original_query, corrected)
    corrected = collapse_repeated_tokens(corrected)
    corrected = " ".join(corrected.split())
    return corrected


# -------------------------------
# 6. 후보 랭킹
# -------------------------------
def rank_menu_candidates(query, top_k=5):
    query_emb = model.encode([query])[0]
    scores = np.dot(menu_embeddings, query_emb) / (
        np.linalg.norm(menu_embeddings, axis=1) * np.linalg.norm(query_emb)
    )

    ranked = []
    for idx, menu in enumerate(menu_list):
        fuzzy_score = fuzz.token_set_ratio(query, menu) / 100.0
        semantic_score = float(scores[idx])
        hybrid_score = semantic_score * 0.5 + fuzzy_score * 0.5
        ranked.append((hybrid_score, menu))

    ranked.sort(reverse=True)
    return [menu for _, menu in ranked[:top_k]]
