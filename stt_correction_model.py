# STT 오인식 보정 모델 - 문자 기반 + 의미 기반
# 문자 기반 보정 : RapidFuzz 활용해 철자 오류 처리
# 의미 기반 보정 : KR-SBERT 모델 사용해 문장 임베딩 후 의미 유사도 계산

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from rapidfuzz import process, fuzz

# -------------------------------
# 1. 메뉴 데이터 로드
# -------------------------------
df = pd.read_csv("menu.csv", encoding="cp949")
menu_list = df["상품명"].tolist()
menu_terms = sorted({
    token
    for menu in menu_list
    for token in menu.split()
    if token
})

# -------------------------------
# 2. 모델 로드
# -------------------------------
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

# -------------------------------
# 3. 메뉴 임베딩 생성
# -------------------------------
menu_embeddings = model.encode(menu_list)
term_embeddings = model.encode(menu_terms)

PROTECTED_WORDS = {"ICE"}

# -------------------------------
# 4. 텍스트 정규화
# -------------------------------
def normalize_text(query):
    query = query.replace("아이스", "ICE")
    query = query.replace("차가운", "ICE")
    query = query.replace("시원한", "ICE")

    query = query.replace("뜨거운", "")
    query = query.replace("따뜻한", "")

    return query.strip()

# -------------------------------
# 5. 오인식 보정
# -------------------------------
def correct_text(query):

    query = normalize_text(query)

    words = query.split()
    corrected_words = []

    for word in words:
        if word in PROTECTED_WORDS:
            corrected_words.append(word)
            continue

        # 1. fuzzy (철자 기반): 전체 메뉴명이 아니라 메뉴 토큰 단위로 보정
        fuzz_match, fuzz_score, _ = process.extractOne(
            word, menu_terms, scorer=fuzz.WRatio
        )

        # 2. SBERT (의미 기반)
        word_emb = model.encode([word])[0]

        scores = [
            np.dot(word_emb, term_emb) / (
                np.linalg.norm(word_emb) * np.linalg.norm(term_emb)
            )
            for term_emb in term_embeddings
        ]

        embed_match = menu_terms[np.argmax(scores)]
        embed_score = max(scores)

        # 1. 철자 유사도가 매우 높으면 사용
        if fuzz_score >= 82:
            corrected_words.append(fuzz_match)

        # 2. fuzzy와 의미 기반이 같은 후보를 가리키면 완만한 오인식도 보정
        elif fuzz_score >= 58 and embed_score >= 0.45 and fuzz_match == embed_match:
            corrected_words.append(fuzz_match)

        # 3. 의미 기반 점수가 충분히 높을 때만 사용
        elif embed_score >= 0.72:
            corrected_words.append(embed_match)

        # 4. 둘 다 애매하면 원래 단어 유지
        else:
            corrected_words.append(word)

    return " ".join(corrected_words)
# -------------------------------
# 6. 후보 반환 (옵션)
# -------------------------------
def correct_with_candidates(query, top_k=3):

    query = normalize_text(query)

    query_emb = model.encode([query])[0]

    scores = np.dot(menu_embeddings, query_emb) / (
        np.linalg.norm(menu_embeddings, axis=1) * np.linalg.norm(query_emb)
    )

    top_idx = np.argsort(scores)[::-1][:top_k]

    candidates = [menu_list[i] for i in top_idx]

    return candidates[0], candidates


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
        hybrid_score = semantic_score * 0.7 + fuzzy_score * 0.3
        ranked.append((hybrid_score, menu))

    ranked.sort(reverse=True)
    return [menu for _, menu in ranked[:top_k]]
