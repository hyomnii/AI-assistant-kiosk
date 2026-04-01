# STT 오인식 보정 모델 - 문자 기반 + 의미 기반
# 문자 기반 보정 : RapidFuzz 활용해 철자 오류 처리
# 의미 기반 보정 : KR-SBERT 모델 사용해 문장 임베딩 후 의미 유사도 계산

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer


# 데이터 로드 및 메뉴 구성
df = pd.read_csv("menu.csv", encoding="cp949")
menu_list = df["상품명"].tolist()

# 메뉴명에서 단어 단위 추출
menu_terms = sorted({
    token
    for menu in menu_list
    for token in menu.split()
    if token
})


# SBERT 모델 로드 및 임베딩 생성
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
menu_embeddings = model.encode(menu_list)
term_embeddings = model.encode(menu_terms)


PROTECTED_WORDS = {"ICE"}

ORDER_STOPWORDS = {
    "하나", "둘", "셋", "넷",
    "한", "두", "세", "네",
    "한잔", "두잔", "세잔", "네잔",
    "잔", "개", "컵",
    "주세요", "주세용", "줘", "줘봐", "줘봐요",
    "부탁해", "부탁해요", "좀", "하나만"
}

COLD_WORDS = ("아이스", "차가운", "시원한")
HOT_WORDS = ("뜨거운", "따뜻한", "핫")


# 텍스트 정규화 함수 (불필요 단어 제거 및 온도 통일)
def normalize_text(query):
    words = []

    for raw_word in query.split():
        if not raw_word:
            continue

        cold_score = max(fuzz.WRatio(raw_word, cold) for cold in COLD_WORDS)
        hot_score = max(fuzz.WRatio(raw_word, hot) for hot in HOT_WORDS)

        # ICE 표현 통일
        if cold_score >= 70:
            words.append("ICE")
            continue

        # HOT 표현 제거
        if hot_score >= 80:
            continue

        # 주문 관련 불필요 단어 제거
        if raw_word in ORDER_STOPWORDS:
            continue

        words.append(raw_word)

    return " ".join(words).strip()


# 붙어있는 단어를 메뉴 단어 기준으로 분해
def split_menu_terms(query):
    for term in menu_terms:
        if term in query and term != query:
            query = query.replace(term, f" {term} ")
    return query


# 단어 유사도 계산
def _menu_term_similarity(word, term):
    wratio = fuzz.WRatio(word, term)
    partial = fuzz.partial_ratio(word, term)
    token_sort = fuzz.token_sort_ratio(word, term)
    return wratio, partial, token_sort


# 보정 여부 판단
def _should_correct(word, candidate, fuzz_score, embed_match, embed_score):
    _, partial_score, token_sort_score = _menu_term_similarity(word, candidate)

    if fuzz_score >= 82:
        return True

    if fuzz_score >= 64 and partial_score >= 85:
        return True

    if fuzz_score >= 58 and embed_score >= 0.45 and candidate == embed_match:
        return True

    if partial_score >= 90 and token_sort_score >= 70 and candidate == embed_match:
        return True

    return False


# STT 오인식 보정 함수
def correct_text(query):
    query = normalize_text(query)
    query = split_menu_terms(query)

    words = query.split()
    corrected_words = []

    for word in words:
        if word in PROTECTED_WORDS:
            corrected_words.append(word)
            continue

        # fuzzy 기반 후보 탐색
        match = process.extractOne(word, menu_terms, scorer=fuzz.WRatio)

        if not match:
            corrected_words.append(word)
            continue

        fuzz_match, fuzz_score, _ = match

        # embedding 유사도 계산
        word_emb = model.encode([word])[0]
        scores = [
            np.dot(word_emb, term_emb) /
            (np.linalg.norm(word_emb) * np.linalg.norm(term_emb))
            for term_emb in term_embeddings
        ]

        embed_match = menu_terms[np.argmax(scores)]
        embed_score = max(scores)

        # 보정 여부 결정
        if _should_correct(word, fuzz_match, fuzz_score, embed_match, embed_score):
            corrected_words.append(fuzz_match)
        else:
            corrected_words.append(word)

    return " ".join(corrected_words)


# fallback 메뉴 추천 (semantic + fuzzy 혼합)
def rank_menu_candidates(query, top_k=5):
    query = normalize_text(query)
    query_emb = model.encode([query])[0]

    semantic_scores = np.dot(menu_embeddings, query_emb) / (
        np.linalg.norm(menu_embeddings, axis=1) * np.linalg.norm(query_emb)
    )

    ranked = []
    for idx, menu in enumerate(menu_list):
        fuzzy_score = fuzz.token_set_ratio(query, menu) / 100.0
        semantic_score = float(semantic_scores[idx])

        # semantic + fuzzy 혼합 점수
        hybrid_score = semantic_score * 0.7 + fuzzy_score * 0.3
        ranked.append((hybrid_score, menu))

    ranked.sort(reverse=True)

    return [menu for _, menu in ranked[:top_k]]