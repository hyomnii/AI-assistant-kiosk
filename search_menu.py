import pandas as pd
import os
import json
import re
import numpy as np
import sounddevice as sd
import whisper
from chromadb import PersistentClient
from openai import OpenAI
from rapidfuzz import fuzz, process

from stt_correction_model_hybrid import correct_text, model, rank_menu_candidates


# -------------------------------
# 0. OpenAI 설정
# -------------------------------
api_key = "API_KEY"  # 실제 API 키로 교체
client_llm = OpenAI(api_key=api_key) if api_key else None
INTENT_MODEL = os.getenv("OPENAI_INTENT_MODEL", "gpt-4o-mini")
RERANK_MODEL = os.getenv("OPENAI_RERANK_MODEL", "gpt-4o-mini")


def _generate_openai_text(prompt, model_name):
    if client_llm is None:
        return None

    try:
        res = client_llm.responses.create(
            model=model_name,
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

client_db = PersistentClient(
    path=r"C:\Users\joh82\Documents\GitHub\AI-assistant-kiosk\menu_DB"
)
collection = client_db.get_collection(name="menu_db")


# -------------------------------
# 2. 후보 생성
# -------------------------------
def get_candidates():
    return [
        {
            "name": row["상품명"],
            "category": row["카테고리"],
            "is_ice": "ICE" in row["상품명"],
            "allergy": str(row["알레르기"]) if pd.notna(row["알레르기"]) else "",
            "caffeine": float(row["카페인"]) if pd.notna(row["카페인"]) else 0.0,
        }
        for _, row in df.iterrows()
    ]


# -------------------------------
# 2-1. 문자열 정리
# -------------------------------
NOISE_WORDS = {"한잔", "한", "하나", "잔", "주세요", "부탁", "좀", "먹고", "싶어", "싶어요", "줘"}
GROUP_HINTS = {"종류", "메뉴", "전체", "목록", "추천", "뭐", "있어", "보여"}
DESIRE_HINTS = {"먹고", "싶어", "싶어요", "땡겨", "원해", "원해요"}
ALLERGENS = ["우유", "대두", "달걀", "밀", "복숭아", "아황산류", "돼지고기"]
CATEGORY_TOKENS = {"커피", "차", "디저트"}
FILTER_HINTS = {"메뉴", "종류", "전체", "목록", "추천", "보여", "뭐", "있어", "알려줘", "없는", "있는", "들어간", "안", "빼고", "제외", "없이", "알레르기"}
DESSERT_HINTS = {"케익", "케이크", "케잌", "쿠키", "브레드", "빵"}
TEA_HINTS = {"차", "아이스티", "캐모마일", "페퍼민트", "레몬차", "유자차", "복숭아"}


def compact_text(text):
    return "".join(text.split())


def sanitize_query(text):
    compact = text.replace(",", " ").replace("/", " ").strip()
    parts = [w for w in compact.split() if w and w not in NOISE_WORDS]
    return " ".join(parts)


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


def choose_match_query(raw_query, corrected_query):
    corrected = sanitize_query(strip_temp_tokens(corrected_query))
    raw = sanitize_query(strip_temp_tokens(raw_query))

    if not corrected:
        return raw
    if not raw:
        return corrected

    if has_pastry_signal(raw) and not has_pastry_signal(corrected):
        return raw
    if has_tea_signal(raw) and not has_tea_signal(corrected):
        return raw

    corrected_compact = compact_text(corrected)
    raw_compact = compact_text(raw)
    if len(corrected_compact) + 1 < len(raw_compact):
        return raw
    return corrected


def strip_temp_tokens(text):
    text = text.replace("ICE", " ")
    text = text.replace("아이스", " ")
    text = text.replace("차가운", " ")
    text = text.replace("시원한", " ")
    text = text.replace("따뜻한", " ")
    text = text.replace("뜨거운", " ")
    text = text.replace("핫", " ")
    return " ".join(text.split())


def base_menu_name(name):
    n = name
    if n.startswith("ICE "):
        n = n[4:]
    return n.strip()


def is_group_intent(query):
    return any(token in query for token in GROUP_HINTS | DESIRE_HINTS)


def mentions_decaf(text):
    return any(x in text for x in ["디카페인", "디카페", "디카폐", "디카패"])


def best_single_match(match_query, temp, candidates):
    if not match_query:
        return None, 0.0

    scored = []
    wants_decaf = mentions_decaf(match_query)

    for c in candidates:
        name = c["name"]
        is_ice = c["is_ice"]
        base = base_menu_name(name)
        score = fuzz.token_set_ratio(match_query, base)

        if temp == "ICE":
            score += 12 if is_ice else -8
        elif temp == "HOT":
            score += 12 if not is_ice else -12

        if wants_decaf:
            score += 22 if "디카페인" in name else -18
        else:
            score -= 18 if "디카페인" in name else 0

        # 디저트 계열 토큰 보너스
        if has_pastry_signal(match_query) and has_pastry_signal(name):
            score += 24
        if "쿠키" in match_query and "쿠키" in name:
            score += 20
        if any(x in match_query for x in ["빵", "브레드"]) and any(x in name for x in ["빵", "브레드"]):
            score += 20

        # 완전 부분일치 보너스
        if match_query in base:
            score += 10

        scored.append((float(score), name))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0]


def has_negative_constraint(query):
    negatives = ["없는", "안들어간", "안 들어간", "제외", "빼고", "없이"]
    return any(n in query for n in negatives)


def has_filter_context(query):
    return any(token in query for token in FILTER_HINTS)


def extract_conditions(raw_query, corrected_query=None):
    query = raw_query.replace(" ", "")
    corrected_compact = compact_text(strip_temp_tokens(corrected_query or raw_query))
    raw_compact = compact_text(strip_temp_tokens(raw_query))

    category = None
    if raw_compact in CATEGORY_TOKENS:
        category = raw_compact
    elif corrected_compact in CATEGORY_TOKENS:
        category = corrected_compact
    elif is_group_intent(raw_query) or has_filter_context(raw_query) or query in CATEGORY_TOKENS:
        for token in CATEGORY_TOKENS:
            if token in query:
                category = token
                break

    mentioned_allergens = [a for a in ALLERGENS if a in raw_query] if has_filter_context(raw_query) else []
    exclude_allergen = has_negative_constraint(raw_query)

    caffeine_mode = None
    if "디카페인" in raw_query or "디카페" in raw_query or "디카폐" in raw_query:
        caffeine_mode = "DECAF"
    elif "카페인" in raw_query:
        if "적" in raw_query or "낮" in raw_query:
            caffeine_mode = "NO"
        elif has_negative_constraint(raw_query):
            caffeine_mode = "NO"
        else:
            caffeine_mode = "YES"

    return {
        "category": category,
        "allergens": mentioned_allergens,
        "exclude_allergen": exclude_allergen,
        "caffeine_mode": caffeine_mode,
    }


def should_expand_category(raw_query, match_query, category):
    if not category:
        return False
    if is_group_intent(raw_query):
        return True
    compact = compact_text(match_query)
    return compact in CATEGORY_TOKENS or compact == f"{category}하나"


def is_condition_list_query(raw_query, conditions):
    return bool(
        conditions["allergens"]
        or has_negative_constraint(raw_query)
        or ("카페인" in raw_query and has_filter_context(raw_query))
        or (mentions_decaf(raw_query) and has_filter_context(raw_query))
    )


# -------------------------------
# 3. 조건 추출
# -------------------------------
def extract_temp(query):
    q = query.lower()
    compact = query.replace(" ", "")

    # '아이스티'는 음료명 일부이므로 온도 의도로 쓰지 않음
    if "아이스티" in compact or "아이스뜨" in compact:
        return None

    if "ice" in q or any(x in query for x in ["아이스", "아잇", "아이쓰", "아이수", "차가운", "시원한"]):
        return "ICE"

    if has_standalone_token(query, "아아"):
        return "ICE"

    if any(x in query for x in ["따뜻한", "뜨거운", "핫"]) or has_standalone_token(query, "뜨아"):
        return "HOT"

    return None


# -------------------------------
# 4. 카테고리 자동 인식 (핵심)
# -------------------------------
def detect_category(query, allow_single=False):
    compact = query.replace(" ", "")
    compact = compact.replace("메뉴", "").replace("종류", "").replace("추천", "")
    category_aliases = {
        "커피": "커피",
        "디저트": "디저트",
        "차": "차",
    }
    wants_group = is_group_intent(query)

    if not wants_group and not allow_single:
        return None

    for key, value in category_aliases.items():
        if compact == key or (wants_group and key in compact):
            return value

    # 카테고리 단어 자체가 오타일 때 fuzzy로 보정
    token = compact
    if token and wants_group:
        match = process.extractOne(token, list(category_aliases.keys()), scorer=fuzz.WRatio)
        if match and match[1] >= 60:
            return category_aliases[match[0]]

    return None


def fuzzy_single_menu_match(match_query, candidates, temp):
    if not match_query:
        return None

    # 후보가 충분히 좁혀진 뒤에 메뉴명 자체를 직접 fuzzy 매칭
    names = [c["name"] for c in candidates]
    if not names:
        return None

    base_lookup = {}
    base_names = []
    for name in names:
        base = base_menu_name(name)
        base_lookup.setdefault(base, []).append(name)
        base_names.append(base)

    exact_name = base_lookup.get(match_query)
    if exact_name:
        possible = exact_name
        ranked = []
        for n in possible:
            s = 0.0
            if temp == "ICE":
                s += 10 if "ICE" in n else -10
            elif temp == "HOT":
                s += 10 if "ICE" not in n else -10
            ranked.append((s, n))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return ranked[0][1]

    match = process.extractOne(match_query, base_names, scorer=fuzz.WRatio)
    if not match:
        return None

    base_name, score, _ = match
    if score < 64:
        return None

    possible = base_lookup.get(base_name, [])
    if not possible:
        return None

    wants_decaf = mentions_decaf(match_query)
    wants_dessert = has_pastry_signal(match_query)
    wants_tea = has_tea_signal(match_query)
    ranked = []
    for n in possible:
        s = 0.0
        if temp == "ICE":
            s += 10 if "ICE" in n else -10
        elif temp == "HOT":
            s += 10 if "ICE" not in n else -10

        if wants_decaf:
            s += 20 if "디카페인" in n else -15
        else:
            s += -15 if "디카페인" in n else 0

        if has_pastry_signal(match_query) and has_pastry_signal(n):
            s += 18
        if wants_dessert:
            s += 14 if has_pastry_signal(n) else -8
        if wants_tea:
            s += 8 if has_tea_signal(n) else 0

        ranked.append((s, n))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked[0][1]


def exact_menu_match(match_source_query, candidates, temp):
    compact_query = compact_text(strip_temp_tokens(match_source_query))
    if not compact_query:
        return None

    matched = []
    for c in candidates:
        name = c["name"]
        base = base_menu_name(name)
        if compact_text(base) in compact_query or compact_text(name) in compact_query:
            matched.append(name)

    if not matched:
        return None

    ranked = []
    for name in matched:
        score = len(base_menu_name(name))
        if temp == "ICE":
            score += 20 if "ICE" in name else -10
        elif temp == "HOT":
            score += 20 if "ICE" not in name else -10
        ranked.append((score, name))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked[0][1]


# -------------------------------
# 5. 그룹 판단
# -------------------------------
def is_group_query(query):

    matches = [
        name for name in df["상품명"]
        if query in name
    ]

    return len(matches) >= 2


# -------------------------------
# 6. 온도 필터
# -------------------------------
def apply_temp_filter(candidates, temp):

    if temp == "ICE":
        return [c for c in candidates if c["is_ice"]]

    if temp == "HOT":
        return [c for c in candidates if not c["is_ice"]]

    return candidates


def apply_condition_filters(candidates, conditions):
    result = list(candidates)

    if conditions["category"]:
        result = [c for c in result if conditions["category"] in c["category"]]

    if conditions["allergens"]:
        if conditions["exclude_allergen"]:
            result = [
                c for c in result
                if not any(a in c["allergy"] for a in conditions["allergens"])
            ]
        else:
            result = [
                c for c in result
                if any(a in c["allergy"] for a in conditions["allergens"])
            ]

    if conditions["caffeine_mode"] == "DECAF":
        result = [c for c in result if "디카페인" in c["name"]]
    elif conditions["caffeine_mode"] == "NO":
        result = [
            c for c in result
            if ("디카페인" in c["name"]) or (c["caffeine"] <= 1.0)
        ]
    elif conditions["caffeine_mode"] == "YES":
        result = [c for c in result if c["caffeine"] > 1.0]

    return result


# -------------------------------
# 7. LLM 의도 분석
# -------------------------------
def llm_intent(query):
    if client_llm is None:
        return None

    prompt = f"""
다음 문장의 메뉴 주문 의도를 추출하세요. 오타/띄어쓰기 오류를 보정해서 판단하세요.

- category (커피/디저트/차/null)
- keyword (메뉴 핵심 단어 1개, 예: 아메리카노/카페라떼/레몬차/복숭아 아이스티)

입력: {query}
반드시 JSON만 출력:
{{
  "category": "커피|디저트|차|null",
  "keyword": "문자열 또는 null"
}}
"""

    try:
        text = _generate_openai_text(prompt, INTENT_MODEL) or ""
        if text.startswith("{") and text.endswith("}"):
            return json.loads(text)
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        return None
    except Exception:
        return None


# -------------------------------
# 8. LLM 재정렬
# -------------------------------
def llm_rerank(query, candidates):
    if client_llm is None:
        return candidates

    text = "\n".join(candidates)

    prompt = f"""
사용자 요청에 가장 적합한 순서로 정렬하세요.

입력: {query}

후보:
{text}

메뉴 이름만 줄바꿈으로 출력
"""

    try:
        text_out = _generate_openai_text(prompt, RERANK_MODEL) or ""
        reranked = [x.strip() for x in text_out.split("\n") if x.strip()]
        valid = [x for x in reranked if x in candidates]
        if valid:
            remain = [x for x in candidates if x not in valid]
            return valid + remain
        return candidates
    except Exception:
        return candidates


# -------------------------------
# 9. 벡터 검색
# -------------------------------
def vector_search(query, top_k=5):

    emb = model.encode([query])

    results = collection.query(
        query_embeddings=emb.tolist(),
        n_results=top_k
    )

    return [r["name"] for r in results["metadatas"][0]]


# -------------------------------
# 10. 메인
# -------------------------------
def search_menu(query):
    raw_query = query
    temp = extract_temp(raw_query)
    corrected_query = correct_text(raw_query)
    match_query = choose_match_query(raw_query, corrected_query)
    raw_sanitized = sanitize_query(strip_temp_tokens(raw_query))
    if not match_query:
        match_query = raw_sanitized

    candidates = get_candidates()
    conditions = extract_conditions(raw_query, corrected_query=corrected_query)
    category = (
        conditions["category"]
        or detect_category(corrected_query, allow_single=True)
        or detect_category(raw_query, allow_single=True)
    )
    if category and not conditions["category"]:
        conditions["category"] = category

    filtered_candidates = apply_temp_filter(candidates, temp)
    filtered_candidates = apply_condition_filters(filtered_candidates, conditions)

    if not filtered_candidates:
        filtered_candidates = apply_temp_filter(candidates, temp)

    exact_match = exact_menu_match(corrected_query, filtered_candidates, temp) or exact_menu_match(raw_query, filtered_candidates, temp)
    if exact_match and not is_condition_list_query(raw_query, conditions):
        return [exact_match]

    # -------------------------------
    # 0. 카테고리/조건 목록 응답
    # -------------------------------
    if is_condition_list_query(raw_query, conditions):
        names = [c["name"] for c in filtered_candidates]
        return names[:10]

    if should_expand_category(raw_query, match_query, conditions["category"]) or has_negative_constraint(raw_query):
        names = [c["name"] for c in filtered_candidates]
        return names[:10]

    direct_fuzzy = fuzzy_single_menu_match(match_query, filtered_candidates, temp)
    if direct_fuzzy:
        return [direct_fuzzy]

    # -------------------------------
    # 1. 단일 메뉴 우선 매칭
    # -------------------------------
    top_score, top_name = best_single_match(match_query, temp, filtered_candidates)
    if top_name and top_score >= 78 and not is_group_intent(raw_query):
        return [top_name]

    # -------------------------------
    # 1-1. LLM 의도 보정 (저신뢰 단일 질의 우선)
    # -------------------------------
    if not is_group_intent(raw_query):
        intent = llm_intent(raw_query)
        if intent:
            key = str(intent.get("keyword") or "").strip()
            llm_category = str(intent.get("category") or "").strip()

            if llm_category and llm_category != "null":
                cat_results = [
                    c["name"] for c in filtered_candidates
                    if llm_category in c["category"]
                ]
                if cat_results:
                    return cat_results[:10]

            if key:
                # LLM 키워드를 후보 메뉴에 fuzzy 매칭해 단일 결과 보강
                key_match = fuzzy_single_menu_match(key, filtered_candidates, temp)
                if key_match:
                    return [key_match]

    # -------------------------------
    # 2. 카테고리 자동 인식
    # -------------------------------
    category = detect_category(raw_query)

    if category:
        results = [
            c for c in filtered_candidates
            if category in c["category"]
        ]

        return [c["name"] for c in results]

    # -------------------------------
    # 3. 패턴 기반 그룹 (명시적 그룹 의도일 때만)
    # -------------------------------
    if is_group_intent(raw_query) and is_group_query(match_query):

        results = [
            c for c in filtered_candidates
            if match_query in c["name"]
        ]

        return [c["name"] for c in results]

    # -------------------------------
    # 4. 부분 매칭
    # -------------------------------
    partial = [c for c in filtered_candidates if match_query in c["name"]]

    if partial:
        return [partial[0]["name"]]

    # -------------------------------
    # 5. vector fallback
    # -------------------------------
    vector_results = vector_search(match_query)

    allowed_names = {c["name"] for c in filtered_candidates}
    filtered = [v for v in vector_results if v in allowed_names]

    if filtered:
        return [filtered[0]]

    # -------------------------------
    # 6. LLM fallback
    # -------------------------------
    intent = llm_intent(raw_query)

    if intent and intent.get("keyword"):
        key = intent["keyword"]

        results = [
            c["name"] for c in filtered_candidates
            if key in c["name"]
        ]

        if results:
            return results

    # -------------------------------
    # 7. 최종
    # -------------------------------
    ranked = rank_menu_candidates(match_query, top_k=5)
    allowed_names = {c["name"] for c in filtered_candidates}
    ranked = [r for r in ranked if r in allowed_names]

    ranked = llm_rerank(raw_query, ranked)

    return ranked[:1]


# -------------------------------
# 11. Whisper STT 입력
# -------------------------------
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
STT_SAMPLE_RATE = 16000
STT_CHANNELS = 1
STT_RECORD_SECONDS = float(os.getenv("STT_RECORD_SECONDS", "4"))


def listen_with_whisper(stt_model):
    print("\n엔터를 누르면 음성 입력을 시작합니다. 종료하려면 Ctrl+C 또는 'exit'를 말씀하세요.")
    input()
    print(f"{STT_RECORD_SECONDS:.0f}초 동안 듣는 중...")

    audio = sd.rec(
        int(STT_RECORD_SECONDS * STT_SAMPLE_RATE),
        samplerate=STT_SAMPLE_RATE,
        channels=STT_CHANNELS,
        dtype="float32",
    )
    sd.wait()

    audio = np.squeeze(audio).astype(np.float32)
    if audio.size == 0:
        return ""

    result = stt_model.transcribe(audio, language="ko", fp16=False)
    return (result.get("text") or "").strip()


# -------------------------------
# 실행
# -------------------------------
if __name__ == "__main__":
    print("Whisper STT 모델을 로딩합니다...")
    stt_model = whisper.load_model(WHISPER_MODEL_SIZE)

    while True:
        try:
            q = listen_with_whisper(stt_model)
        except KeyboardInterrupt:
            print("\n종료합니다.")
            break
        except Exception as e:
            print(f"음성 입력 오류: {e}")
            continue

        print(f"인식된 문장: {q}")

        if q.strip() in ["종료", "취소", "exit", "quit"]:
            break

        result = search_menu(q)

        if not result:
            print("조건에 맞는 메뉴가 없습니다.")
        else:
            for r in result:
                print(r)
