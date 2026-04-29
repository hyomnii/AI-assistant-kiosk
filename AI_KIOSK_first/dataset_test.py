import os
import json
import pandas as pd
from tqdm import tqdm
import whisper
from stt_correction_model_hybrid import correct_text
from search_menu import search_menu
import random

# --- [1. 경로 설정] ---
current_dir = os.path.dirname(os.path.abspath(__file__)) 
DATASET_DIR = os.path.join(current_dir, "cafe_dataset_foreign") 

AUDIO_DIR = os.path.join(DATASET_DIR, "audio")
LABEL_DIR = os.path.join(DATASET_DIR, "label")
RESULT_SAVE_PATH = "카페도메인_실험결과_외국인_50개.csv"

TEST_SAMPLE_COUNT = 50 

CAFE_KEYWORDS = [
    "아메리카노", "에스프레소", "라떼", "카페라떼", "바닐라라떼", "연유라떼", "카페모카", "카라멜마끼아또", 
    "디카페인", "콜드브루", "아이스티", "복숭아", "유자차", "레몬차", "캐모마일", "페퍼민트", 
    "녹차라떼", "곡물라떼", "초코", "핫초코",
    "허니브레드", "소금빵", "케익", "치즈 케익", "마카다미아", "초콜릿", "쿠키",
    # "커피", "디저트", "우유"
]

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
    
    print("데이터셋 파일 목록을 수집 중입니다...")
    label_map = {}
    for f in os.listdir(LABEL_DIR):
        if f.endswith('.json'):
            file_id = os.path.splitext(f)[0]
            label_map[file_id] = os.path.join(LABEL_DIR, f)

    audio_list = []
    for f in os.listdir(AUDIO_DIR):
        if f.endswith(('.wav', '.mp3')):
            audio_list.append(os.path.join(AUDIO_DIR, f))

    if len(audio_list) == 0:
        print("❌ 오디오 파일을 찾지 못했습니다.")
        return

    actual_sample_size = min(TEST_SAMPLE_COUNT, len(audio_list))
    audio_list = random.sample(audio_list, actual_sample_size)
    print(f"🎯 총 {actual_sample_size}개의 파일을 테스트합니다!\n")

    test_results = []
    keyword_success_count = 0  # 키워드 방어(교정) 성공 카운트
    rag_success_count = 0

    for audio_path in tqdm(audio_list):
        filename = os.path.basename(audio_path)
        file_id = os.path.splitext(filename)[0]

        if file_id not in label_map:
            continue
            
        label_path = label_map[file_id]
        true_text = get_ground_truth(label_path)

        # Trigger Keyword 찾기
        matched_keywords = [kw for kw in CAFE_KEYWORDS if kw in true_text]
        if not matched_keywords:
            raw_stt_temp = stt_model.transcribe(audio_path, language="ko", fp16=False)['text'].strip()
            matched_keywords = [kw for kw in CAFE_KEYWORDS if kw in raw_stt_temp]
            
        trigger_keyword_str = ", ".join(matched_keywords) if matched_keywords else "알수없음"

        # STT 및 교정 파이프라인
        raw_stt = stt_model.transcribe(audio_path, language="ko", fp16=False)['text'].strip()
        corrected_text = correct_text(raw_stt)
        matched_menu = search_menu(corrected_text)

        # 지표 1 : 키워드가 추천 메뉴(matched_menu) 안에 살아남았는가?
        is_keyword_success = "X"
        if matched_menu and matched_keywords:
            for kw in matched_keywords:
                # 예: kw="복숭아", menu_item="ICE 복숭아 아이스티" -> 포함되어 있으면 성공!
                if any(kw in menu_item for menu_item in matched_menu):
                    is_keyword_success = "O"
                    break # 하나라도 살아남았으면 성공 처리
                    
        if is_keyword_success == "O":
            keyword_success_count += 1

        # 지표 2: RAG 매칭 성공 여부
        is_rag_success = "O" if matched_menu else "X"
        if is_rag_success == "O":
            rag_success_count += 1

        test_results.append({
            "File": filename,
            "Ground_Truth": true_text,
            "Raw_STT": raw_stt,
            "Corrected_STT": corrected_text,
            "Trigger_Keyword": trigger_keyword_str,
            "Matched_Menu": matched_menu,
            "교정성공(키워드보존)": is_keyword_success, # <-- 엑셀에 명확하게 표기
            "RAG범주화성공": is_rag_success
        })

    # 결과 저장
    df = pd.DataFrame(test_results)
    df.to_csv(RESULT_SAVE_PATH, index=False, encoding='utf-8-sig')
    
    # 통계 지표 계산
    keyword_rate = (keyword_success_count / actual_sample_size) * 100
    rag_rate = (rag_success_count / actual_sample_size) * 100

    print("\n" + "="*50)
    print("📊 [캡스톤 발표용 핵심 성과 지표]")
    print(f"1️⃣ 핵심 의도(키워드) 교정 성공률: {keyword_rate:.1f}% ({keyword_success_count}/{actual_sample_size})")
    print(f"   -> 사용자의 핵심 발화 단어가 시스템 파이프라인을 뚫고 최종 메뉴에 정확히 안착한 비율입니다.")
    print(f"2️⃣ RAG 기반 메뉴 범주화 성공률: {rag_rate:.1f}% ({rag_success_count}/{actual_sample_size})")
    print(f"   -> 1번에서 정확한 메뉴를 못 찾았더라도, 연관된 메뉴 범주를 방어적으로 추천해낸 비율입니다.")
    print("="*50)
    print(f"🎉 엑셀 파일이 '{RESULT_SAVE_PATH}'에 저장되었습니다.")

if __name__ == "__main__":
    run_cafe_domain_test()