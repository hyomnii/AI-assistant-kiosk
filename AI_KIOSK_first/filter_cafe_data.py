import os
import json
import shutil
from tqdm import tqdm

# --- [1. 경로 설정] ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
DATASET_DIR = os.path.join(parent_dir, "dataset_foreign")

AUDIO_DIR = os.path.join(DATASET_DIR, "audio")
LABEL_DIR = os.path.join(DATASET_DIR, "label")

# 새로 만들어질 "카페 전용" 데이터 폴더
CAFE_DATASET_DIR = os.path.join(current_dir, "cafe_dataset_foreign")
CAFE_AUDIO_DIR = os.path.join(CAFE_DATASET_DIR, "audio")
CAFE_LABEL_DIR = os.path.join(CAFE_DATASET_DIR, "label")

# --- [2. 카페 관련 키워드 사전] ---
# 이 단어들이 포함된 문장만 쏙쏙 뽑아냅니다!
# --- [2. 카페 관련 키워드 사전] ---
CAFE_KEYWORDS = [
    "아메리카노", "에스프레소", "라떼", "카페라떼", "바닐라라떼", "연유라떼", "카페모카", "카라멜마끼아또", 
    "디카페인", "콜드브루", "아이스티", "복숭아", "유자차", "레몬차", "캐모마일", "페퍼민트", 
    "녹차라떼", "곡물라떼", "초코", "핫초코",
    "허니브레드", "소금빵", "케익", "치즈 케익", "마카다미아", "초콜릿", "쿠키",
    # "커피", "디저트", "우유"
]

def create_folders():
    os.makedirs(CAFE_AUDIO_DIR, exist_ok=True)
    os.makedirs(CAFE_LABEL_DIR, exist_ok=True)

def extract_text_from_json(json_path):
    """JSON에서 정답 텍스트를 안전하게 뽑아내는 함수"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # AI HUB JSON 구조에 맞춰 텍스트 추출 (딕셔너리 형태도 방어)
            text = data.get('transcription') or data.get('text') or ""
            if isinstance(text, dict): 
                text = text.get('Reading') or str(text)
            return text
    except:
        return ""

def run_filter():
    print("🧹 18만 개 데이터 중 카페 관련 데이터만 필터링합니다...")
    create_folders()

    # 1. 라벨 파일(.json) 전체 수집
    label_list = []
    for root, _, files in os.walk(LABEL_DIR):
        for f in files:
            if f.endswith('.json'):
                label_list.append(os.path.join(root, f))

    # 2. 오디오 파일 맵핑 생성 (빠른 찾기)
    audio_map = {}
    for root, _, files in os.walk(AUDIO_DIR):
        for f in files:
            if f.endswith(('.wav', '.mp3')):
                file_id = os.path.splitext(f)[0]
                audio_map[file_id] = os.path.join(root, f)

    matched_count = 0

    # 3. 키워드 필터링 및 파일 복사
    for label_path in tqdm(label_list, desc="데이터 필터링 중"):
        text = extract_text_from_json(label_path)
        
        # 키워드가 하나라도 포함되어 있는지 검사
        if any(keyword in text for keyword in CAFE_KEYWORDS):
            filename = os.path.basename(label_path)
            file_id = os.path.splitext(filename)[0]

            # 짝꿍 오디오 파일이 존재하는지 확인
            if file_id in audio_map:
                audio_path = audio_map[file_id]
                
                # 새로운 cafe_dataset 폴더로 짝꿍 복사 (원본은 안전하게 유지)
                new_label_path = os.path.join(CAFE_LABEL_DIR, filename)
                new_audio_path = os.path.join(CAFE_AUDIO_DIR, os.path.basename(audio_path))
                
                shutil.copy2(label_path, new_label_path)
                shutil.copy2(audio_path, new_audio_path)
                matched_count += 1

    print("\n" + "="*50)
    print(f"🎉 필터링 완료! 총 {matched_count}개의 '카페 도메인' 데이터를 찾았습니다.")
    print(f"📂 저장 위치: {CAFE_DATASET_DIR}")
    print("="*50)

if __name__ == "__main__":
    run_filter()