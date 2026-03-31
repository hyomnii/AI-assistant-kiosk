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

# -------------------------------
# 2. 모델 로드
# -------------------------------
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

# -------------------------------
# 3. 메뉴 임베딩 생성
# -------------------------------
menu_embeddings = model.encode(menu_list)

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

        # 1. fuzzy (철자 기반)
        fuzz_match, fuzz_score, _ = process.extractOne(
            word, menu_list, scorer=fuzz.partial_ratio
        )

        # 2. SBERT (의미 기반)
        word_emb = model.encode([word])[0]

        scores = [
            np.dot(word_emb, menu_emb) / (
                np.linalg.norm(word_emb) * np.linalg.norm(menu_emb)
            )
            for menu_emb in menu_embeddings
        ]

        embed_match = menu_list[np.argmax(scores)]
        embed_score = max(scores)

        # 1. 철자 유사도가 충분히 높으면 사용
        if fuzz_score > 55:
            corrected_words.append(fuzz_match)

        # 2. 아니면 의미 기반 사용
        elif embed_score > 0.5:
            corrected_words.append(embed_match)

        # 3. 둘 다 애매하면 원래 단어 유지
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