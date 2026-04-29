import numpy as np
import sounddevice as sd
import whisper
import pandas as pd
from datetime import datetime

# beam_final.py에 정의된 함수와 변수들을 가져옵니다.
from beam_final import apply_ultimate_beamforming, STT_RECORD_SECONDS, RATE, STT_CHANNELS, WHISPER_MODEL_SIZE
from stt_correction_model_hybrid import correct_text
from search_menu import search_menu
# main 파이프라인에서 구현했던 LLM 응답 함수 (동일한 파일에 있거나 여기에 정의)
from main import generate_kiosk_response

def run_angle_experiment():
    print("🚀 각도별 지향성 실험을 시작합니다.")
    stt_model = whisper.load_model(WHISPER_MODEL_SIZE)
    
    # 실험할 각도 리스트
    test_angles = [0, 30, 60, 90]
    experiment_data = []

    print("\n" + "="*80)
    print(f"{'각도':^6} | {'RAW 결과':^20} | {'BEAM 결과':^20} | {'교정 결과':^20}")
    print("-"*80)

    for angle in test_angles:
        input(f"\n📍 [{angle}도] 위치에 스피커/발화 세팅 후 [Enter]를 누르세요...")
        print(f"🎤 {STT_RECORD_SECONDS}초간 녹음 중...", end="", flush=True)
        
        # 1. 음성 녹음
        recording = sd.rec(int(STT_RECORD_SECONDS * RATE), samplerate=RATE, channels=STT_CHANNELS)
        sd.wait()
        print(" [완료]")

        # 2. RAW STT (전처리 없음 - 1번 채널 기준)
        raw_audio = recording[:, 0].astype(np.float32)
        raw_stt = stt_model.transcribe(raw_audio, language="ko", fp16=False)['text'].strip()

        # 3. BEAM STT (빔포밍 전처리 적용)
        beam_audio = apply_ultimate_beamforming(recording)
        beam_stt = stt_model.transcribe(beam_audio, language="ko", fp16=False)['text'].strip()

        # 4. 교정 후 결과 (하이브리드 후처리)
        corrected_text = correct_text(beam_stt) if beam_stt else ""

        # 5. LLM 응답 생성 (RAG 결과 포함)
        menu_result = search_menu(corrected_text)
        llm_response = generate_kiosk_response(menu_result, corrected_text)

        # 결과 저장
        row = {
            "각도": f"{angle}도",
            "RAW 결과": raw_stt,
            "BEAM 결과": beam_stt,
            "교정 후 결과": corrected_text,
            "LLM 응답": llm_response
        }
        experiment_data.append(row)

        # 화면 실시간 출력
        print(f"{row['각도']:^8} | {row['RAW 결과'][:15]:^20} | {row['BEAM 결과'][:15]:^20} | {row['교정 후 결과'][:15]:^20}")
        print(f"🗨️ LLM: {row['LLM 응답']}")

    # CSV 파일로 저장 (PPT용 데이터)
    df = pd.DataFrame(experiment_data)
    filename = f"각도별_실험결과_{datetime.now().strftime('%m%d_%H%M')}.csv"
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print("\n" + "="*80)
    print(f"🎉 모든 실험 완료! 결과가 '{filename}'에 저장되었습니다.")

if __name__ == "__main__":
    run_angle_experiment()