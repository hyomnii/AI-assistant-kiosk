import os
import hashlib
import numpy as np
import pandas as pd
from openai import OpenAI
from rapidfuzz import fuzz


# -------------------------------
# 0. OpenAI 설정
# -------------------------------
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
api_key = os.getenv("OPENAI_API_KEY") # 실제 API 키로 교체하세요
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
# 1. 데이터 로드 (프롬프트 가드용)
# -------------------------------
df = pd.read_csv("menu.csv", encoding="cp949")
menu_list = df["상품명"].tolist()


# -------------------------------
# 1-1. search_menu 호환용 경량 벡터 모델
# -------------------------------
class _HashVectorModel:
    def __init__(self, dim=768):
        self.dim = dim

    def _encode_one(self, text):
        vec = np.zeros(self.dim, dtype=np.float32)
        if not text:
            return vec

        # 문자열을 안정적으로 해시해 고정 차원 벡터로 투영
        for token in text.split():
            h = hashlib.sha256(token.encode("utf-8")).digest()
            for i in range(0, len(h), 4):
                chunk = h[i:i + 4]
                idx = int.from_bytes(chunk, "little") % self.dim
                vec[idx] += 1.0

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def encode(self, texts):
        return np.vstack([self._encode_one(t) for t in texts])


model = _HashVectorModel(dim=768)


# -------------------------------
# 2. 텍스트 정규화
# -------------------------------
def normalize_text(query):
    query = query.replace("아이스", "ICE")
    query = query.replace("차가운", "ICE")
    query = query.replace("시원한", "ICE")
    query = query.replace("뜨거운", "HOT")
    query = query.replace("따뜻한", "HOT")
    query = query.replace("핫", "HOT")
    return query.strip()


# -------------------------------
# 3. LLM only 보정
# -------------------------------
def correct_text(query):
    query = normalize_text(query)

    if client is None:
        corrected = query
    else:
        prompt = f"""
다음 문장은 카페 키오스크 STT 오인식 문장입니다.
카페 주문 의도를 보존하며 정확히 교정하세요.
규칙:
- 출력은 교정 문장 1줄만.
- HOT/ICE 토큰은 유지하세요.
- 메뉴명은 아래 목록 기준으로 우선 교정하세요.

메뉴 목록:
{menu_list}

입력: {query}
출력:
"""
        text = _generate_text(prompt)
        corrected = text if text else query

    corrected = corrected.replace("HOT", " ")
    corrected = " ".join(corrected.split())
    return corrected


# -------------------------------
# 4. 후보 랭킹 (호환용)
# -------------------------------
def rank_menu_candidates(query, top_k=5):
    query = normalize_text(query)
    scored = []

    for menu in menu_list:
        score = fuzz.token_set_ratio(query, menu)
        scored.append((score, menu))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [menu for _, menu in scored[:top_k]]
