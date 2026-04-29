import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

import json
import pandas as pd
from tqdm import tqdm
import whisper
from stt_correction_model_hybrid import correct_text
from search_menu import search_menu
import random

# --- [1. 경로 설정] ---
DATASET_DIR = os.path.join(current_dir, "cafe_dataset_foreign") 
AUDIO_DIR = os.path.join(DATASET_DIR, "audio")
LABEL_DIR = os.path.join(DATASET_DIR, "label")
RESULT_SAVE_PATH = "카페도메인_최종결과_카테고리검증.csv"

TEST_SAMPLE_COUNT = 225

# 🌟 1. 키워드를 카테고리별로 묶어줍니다 (진짜 범주화 평가를 위해!)
CATEGORY_DICT = {
    "커피류": ["아메리카노", "에스프레소", "라떼", "카페라떼", "바닐라라떼", "연유라떼", "카페모카", "카라멜마끼아또", "디카페인", "콜드브루"],
    "차/음료류": ["아이스티", "복숭아", "유자차", "레몬차", "캐모마일", "페퍼민트", "녹차라떼", "곡물라떼", "초코", "핫초코"],
    "디저트류": ["허니브레드", "소금빵", "케익", "치즈 케익", "마카다미아", "초콜릿", "쿠키"]
    # "커피", "디저트", "우유"
}

# 모든 키워드를 하나로 합친 리스트 (기존 필터링용)
CAFE_KEYWORDS = sum(CATEGORY_DICT.values(), [])

def get_category(word):
    """단어가 어느 카테고리에 속하는지 찾아주는 함수"""
    for cat, words in CATEGORY_DICT.items():
        if any(w in word for w in words) or any(word in w for w in words):
            return cat
    return "기타"

def get_ground_truth(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            text = data.get('transcription') or data.get('text') or ""
            if isinstance(text, dict):
                text = text.get('Reading') or str(text)
            return text
    except:
        return ""

def run_cafe_domain_test():
    print("Whisper STT 모델 로딩 중...")
    stt_model = whisper.load_model("base")
    
    label_map = {os.path.splitext(f)[0]: os.path.join(LABEL_DIR, f) for f in os.listdir(LABEL_DIR) if f.endswith('.json')}
    audio_list = [os.path.join(AUDIO_DIR, f) for f in os.listdir(AUDIO_DIR) if f.endswith(('.wav', '.mp3'))]

    if not audio_list:
        print("❌ 오디오 파일을 찾지 못했습니다.")
        return

    actual_sample_size = min(TEST_SAMPLE_COUNT, len(audio_list))
    audio_list = random.sample(audio_list, actual_sample_size)
    print(f"🎯 총 {actual_sample_size}개의 파일을 테스트합니다!\n")

    test_results = []
    keyword_success_count = 0 
    rag_success_count = 0

    for audio_path in tqdm(audio_list):
        filename = os.path.basename(audio_path)
        file_id = os.path.splitext(filename)[0]

        if file_id not in label_map: continue
            
        true_text = get_ground_truth(label_map[file_id])

        matched_keywords = [kw for kw in CAFE_KEYWORDS if kw in true_text]
        if not matched_keywords:
            raw_stt_temp = stt_model.transcribe(audio_path, language="ko", fp16=False)['text'].strip()
            matched_keywords = [kw for kw in CAFE_KEYWORDS if kw in raw_stt_temp]
            
        trigger_keyword_str = ", ".join(matched_keywords) if matched_keywords else "알수없음"

        # STT -> 교정 -> RAG 파이프라인
        raw_stt = stt_model.transcribe(audio_path, language="ko", fp16=False)['text'].strip()
        corrected_text = correct_text(raw_stt)
        matched_menu = search_menu(corrected_text)

        # 🌟 지표 1: 완벽한 단어 매칭 (키워드 보존 성공률)
        is_keyword_success = "X"
        if matched_menu and matched_keywords:
            for kw in matched_keywords:
                if any(kw in menu_item for menu_item in matched_menu):
                    is_keyword_success = "O"
                    break 
        if is_keyword_success == "O": keyword_success_count += 1

        # 🌟 지표 2 (업그레이드!): 진짜 RAG 카테고리 범주화 성공률
        is_rag_success = "X"
        if matched_menu and matched_keywords:
            # 1. 사용자가 원했던 카테고리들 파악 (예: 라떼 -> 커피류)
            target_categories = [get_category(kw) for kw in matched_keywords]
            
            # 2. 추천받은 메뉴들 중에 해당 카테고리가 단 하나라도 있는지 확인!
            for menu_item in matched_menu:
                menu_cat = get_category(menu_item)
                if menu_cat in target_categories and menu_cat != "기타":
                    is_rag_success = "O"
                    break
                    
        if is_rag_success == "O": rag_success_count += 1

        test_results.append({
            "File": filename,
            "Ground_Truth": true_text,
            "Raw_STT": raw_stt,
            "Corrected_STT": corrected_text,
            "Trigger_Keyword": trigger_keyword_str,
            "Matched_Menu": matched_menu,
            "교정성공(키워드보존)": is_keyword_success,
            "RAG성공(카테고리일치)": is_rag_success  # <-- 엑셀 제목도 변경!
        })

    df = pd.DataFrame(test_results)
    df.to_csv(RESULT_SAVE_PATH, index=False, encoding='utf-8-sig')
    
    if actual_sample_size > 0:
        keyword_rate = (keyword_success_count / actual_sample_size) * 100
        rag_rate = (rag_success_count / actual_sample_size) * 100

        print("\n" + "="*50)
        print("📊 [캡스톤 발표용 핵심 성과 지표]")
        print(f"1️⃣ 핵심 단어 방어 성공률: {keyword_rate:.1f}% ({keyword_success_count}/{actual_sample_size})")
        print(f"   -> 완전히 동일한 메뉴 키워드가 살아남은 비율")
        print(f"2️⃣ 진짜 카테고리 범주화 성공률: {rag_rate:.1f}% ({rag_success_count}/{actual_sample_size})")
        print(f"   -> (NEW) 단어는 틀렸어도, RAG가 '커피류/디저트류' 등 같은 범주의 메뉴를 똑똑하게 추천해 낸 비율")
        print("="*50)

if __name__ == "__main__":
    run_cafe_domain_test()