import pandas as pd
from chromadb import PersistentClient
from rapidfuzz import process, fuzz

# 오인식 보정 + 모델 공유
from stt_correction_model import correct_text, model, rank_menu_candidates


# -------------------------------
# 1. 데이터 및 DB 로드
# -------------------------------
df = pd.read_csv("menu.csv", encoding="cp949")

client = PersistentClient(
    path=r"C:\Users\joh82\Documents\GitHub\AI-assistant-kiosk\menu_DB"
)
collection = client.get_collection(name="menu_db")
BASE_KEYWORDS = ["아메리카노", "라떼", "에이드", "차", "케익", "쿠키", "빵"]


# -------------------------------
# 2. 정확도 매칭
# -------------------------------
def exact_match(query):
    for name in df["상품명"]:
        if query.strip() == name:
            return [name]
    return None


# -------------------------------
# 3. 부분 정확 매칭
# -------------------------------
def partial_match(query):
    results = []
    for name in df["상품명"]:
        if query in name:
            results.append(name)
    return results if results else None


def fuzzy_keyword_match(query, score_cutoff=80):
    words = [word for word in query.split() if word]
    keywords = []

    for word in words or [query]:
        match = process.extractOne(
            word,
            BASE_KEYWORDS,
            scorer=fuzz.WRatio,
            score_cutoff=score_cutoff
        )
        if match:
            keyword = match[0]
            if keyword not in keywords:
                keywords.append(keyword)

    return keywords


# -------------------------------
# 4. 의도 추출
# -------------------------------
def extract_conditions(query):
    # 온도
    temp = None
    if "ICE" in query:
        temp = "ICE"
    elif any(x in query for x in ["따뜻한", "뜨거운", "핫"]):
        temp = "HOT"

    # 알레르기
    allergens = ["우유", "대두", "달걀", "밀", "복숭아", "아황산류", "돼지고기"]
    found_allergens = [a for a in allergens if a in query]

    negative_allergy = any(x in query for x in ["없는", "제외", "빼고", "안들어간"])
    negative_category = "아닌" in query

    # 카페인
    caffeine = None
    if "카페인" in query:
        caffeine = True

    # -------------------------------
    # 키워드 추출 
    # -------------------------------
    # 1️. 기본 포함
    keywords = [k for k in BASE_KEYWORDS if k in query]

    # 2️. 철자 보완
    for keyword in fuzzy_keyword_match(query):
        if keyword not in keywords:
            keywords.append(keyword)

    return temp, found_allergens, negative_allergy, negative_category, keywords, caffeine


# -------------------------------
# 5. 카테고리 추출
# -------------------------------
def extract_category(query):
    if "커피" in query:
        return "커피"
    if "빵" in query or "디저트" in query:
        return "디저트"
    if "차" in query:
        return "차"
    if "에이드" in query:
        return "에이드"
    return None


# -------------------------------
# 6. 벡터 검색 (fallback)
# -------------------------------
def vector_search(query, top_k=10):
    emb = model.encode([query])

    results = collection.query(
        query_embeddings=emb.tolist(),
        n_results=top_k
    )

    return results["metadatas"][0]


# -------------------------------
# 7. 전체 후보 생성
# -------------------------------
def get_all_candidates():
    return [
        {
            "name": row["상품명"],
            "category": row["카테고리"],
            "allergy": str(row["알레르기"]),
            "caffeine": float(row["카페인"]),
            "is_ice": "ICE" in row["상품명"]
        }
        for _, row in df.iterrows()
    ]


# -------------------------------
# 8. 필터
# -------------------------------
def apply_filters(candidates, temp, allergens, neg_allergy, neg_category, keywords, category, caffeine):

    # 온도
    if temp == "ICE":
        candidates = [c for c in candidates if c["is_ice"]]
    elif temp == "HOT":
        candidates = [c for c in candidates if not c["is_ice"]]

    # 카테고리
    if category:
        if neg_category:
            candidates = [c for c in candidates if category not in c["category"]]
        else:
            candidates = [c for c in candidates if category in c["category"]]

    # 키워드
    if keywords:
        candidates = [
            c for c in candidates
            if any(k in c["name"] for k in keywords)
        ]

    # 알레르기
    if allergens:
        if neg_allergy:
            candidates = [
                c for c in candidates
                if not any(a in c["allergy"] for a in allergens)
            ]
        else:
            candidates = [
                c for c in candidates
                if any(a in c["allergy"] for a in allergens)
            ]

    # 카페인
    if caffeine is not None:
        if neg_allergy:
            candidates = [c for c in candidates if c["caffeine"] == 0]
        else:
            candidates = [c for c in candidates if c["caffeine"] > 0]

    return candidates


# -------------------------------
# 9. 메인
# -------------------------------
def search_menu(query):

    # 1️. 오인식 보정
    query = correct_text(query)

    # 2️. 정확 매칭
    exact = exact_match(query)
    if exact:
        return exact

    # 3️. 부분 매칭
    partial = partial_match(query)
    if partial:
        return partial[:5]

    # 4️. 짧은 메뉴명/오인식 질의는 키워드 기준으로 먼저 좁힘
    direct_keywords = fuzzy_keyword_match(query)
    if direct_keywords and len(query.split()) <= 2:
        direct_matches = [
            name for name in df["상품명"]
            if any(keyword in name for keyword in direct_keywords)
        ]
        if direct_matches:
            return direct_matches[:10]

    # 5️. 조건 추출
    temp, allergens, neg_allergy, neg_category, keywords, caffeine = extract_conditions(query)
    category = extract_category(query)

    # 6️. 필터 먼저
    candidates = get_all_candidates()

    filtered = apply_filters(
        candidates,
        temp,
        allergens,
        neg_allergy,
        neg_category,
        keywords,
        category,
        caffeine
    )

    if filtered:
        return [c["name"] for c in filtered[:10]]
    
    # 7. fallback: vector search
    candidates = vector_search(query)

    filtered = apply_filters(
        candidates,
        temp,
        allergens,
        neg_allergy,
        neg_category,
        keywords,
        category,
        caffeine
    )

    if filtered:
        return [c["name"] for c in filtered[:10]]

    return rank_menu_candidates(query, top_k=10)


# -------------------------------
# 10. 실행
# -------------------------------
while True:
    query = input("어떤 메뉴를 드릴까요? : ")

    if query.strip() in ["취소", "종료", "exit", "quit"]:
        print("프로그램을 종료합니다.")
        break

    result = search_menu(query)

    if not result:
        print("조건에 맞는 메뉴가 없습니다.")
    else:
        for r in result:
            print(r)
