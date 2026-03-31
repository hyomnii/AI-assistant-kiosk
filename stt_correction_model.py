# STT 오인식 보정 모델 - 문자 기반 + 의미 기반
# 문자 기반 보정 : RapidFuzz 활용해 철자 오류 처리
# 의미 기반 보정 : KR-SBERT 모델 사용해 문장 임베딩
# 이후 코사인 유사도 통해 가장 유사한 메뉴명 탐색
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
# 2. 문장 임베딩 모델 로드
# -------------------------------
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")


# -------------------------------
# 3. 메뉴명 임베딩 미리 생성
# -------------------------------
menu_embeddings = model.encode(menu_list)


# -------------------------------
# 4. 입력 문장 정규화
# -------------------------------
# 온도 표현 정규화
def normalize_text(query):
    query = query.replace("아이스", "ICE")
    query = query.replace("차가운", "ICE")
    query = query.replace("시원한", "ICE")

    query = query.replace("뜨거운", "")
    query = query.replace("따뜻한", "")

    return query.strip()


# -------------------------------
# 5. 코사인 유사도 계산 함수
# -------------------------------
# 두 벡터가 얼마나 비슷한 방향을 가지는지 계산
# 값이 1에 가까울수록 의미적으로 더 유사
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# -------------------------------
# 6. STT 오인식 보정 함수
# -------------------------------
# 문장 안의 단어 중 메뉴명으로 보이는 부분만 보정
def correct_text(query):

    # 1. 입력 문장 정규화
    query = normalize_text(query)

    # 2. 띄어쓰기 기준으로 단어 분리
    words = query.split()

    # 3. 보정된 단어를 담을 리스트
    corrected_words = []

    # 4. 각 단어별로 오인식 여부 검사
    for word in words:

        # -------------------------------
        # 4-1. fuzzy matching
        # -------------------------------
        # 철자 기반으로 가장 비슷한 메뉴명 탐색
        fuzz_match, fuzz_score, _ = process.extractOne(
            word, menu_list, scorer=fuzz.partial_ratio
        )

        # -------------------------------
        # 4-2. 임베딩 기반 유사도 계산
        # -------------------------------
        # 전체 메뉴명 벡터와 비교하여 가장 의미적으로 가까운 메뉴 탐색
        word_emb = model.encode([word])[0]

        scores = [
            cosine_similarity(word_emb, menu_emb)
            for menu_emb in menu_embeddings
        ]

        best_idx = int(np.argmax(scores))
        embed_match = menu_list[best_idx]

        # -------------------------------
        # 4-3. 최종 교체 여부 결정
        # -------------------------------
        # 철자 유사도가 충분히 높으면 해당 메뉴명으로 교체
        if fuzz_score > 70:
            corrected_words.append(fuzz_match)
        else:
            corrected_words.append(word)

    # 5. 보정된 단어들을 다시 하나의 문장으로 합쳐 반환
    return " ".join(corrected_words)


# -------------------------------
# 7. 후보 메뉴 리스트 반환 함수 (옵션)
# -------------------------------
# 입력과 가장 가까운 메뉴 후보를 몇 개 보여주기 위한 용도입니다.
def correct_with_candidates(query, top_k=3):
    # 1. 문장 정규화
    query = normalize_text(query)

    # 2. 입력 문장을 벡터로 변환
    query_emb = model.encode([query])[0]

    # 3. 전체 메뉴와 유사도 계산
    scores = [
        cosine_similarity(query_emb, menu_emb)
        for menu_emb in menu_embeddings
    ]

    # 4. 유사도가 높은 상위 top_k개 메뉴 인덱스 추출
    top_idx = np.argsort(scores)[::-1][:top_k]

    # 5. 인덱스를 실제 메뉴명으로 변환
    candidates = [menu_list[i] for i in top_idx]

    # 6. 1등 후보와 전체 후보 리스트 반환
    return candidates[0], candidates