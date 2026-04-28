import numpy as np
import sounddevice as sd
import whisper
from gtts import gTTS  # 단순 구현용 TTS (Pyttsx3나 OpenAI TTS로 대체 가능)
import os
import time

# 기존 모듈 임포트
from beam_final_noisetest import apply_hybrid_beamforming
from stt_correction_model_hybrid import correct_text
from search_menu import search_menu, client_llm  # RAG 및 OpenAI 클라이언트

# --- [설정 및 모델 로드] ---
RATE = 16000
stt_model = whisper.load_model("base")

def speak(text):
    """TTS 기능을 통해 사용자에게 음성으로 응답합니다."""
    print(f"🤖 AI 응답: {text}")
    tts = gTTS(text=text, lang='ko')
    tts.save("response.mp3")
    # os.system("start response.mp3") # 윈도우 환경 실행

def run_kiosk_pipeline(audio_data):
    """전체 파이프라인 통합 실행 함수"""
    
    # 1. 전처리 (Beamforming)
    # 입력 audio_data는 (N, 2) 형태의 스테레오 데이터라고 가정
    cleaned_audio = apply_hybrid_beamforming(audio_data, limit_angle=15)
    
    # 2. STT (Whisper)
    result = stt_model.transcribe(cleaned_audio, language="ko")
    raw_text = result['text'].strip()
    print(f"🔍 인식된 텍스트: {raw_text}")
    
    if not raw_text:
        return "죄송합니다. 목소리를 듣지 못했어요. 다시 말씀해 주시겠어요?"

    # 3. 후처리 (Hybrid Correction)
    corrected_text = correct_text(raw_text)
    print(f"✨ 교정된 텍스트: {corrected_text}")

    # 4. RAG (Menu Search)
    menu_result = search_menu(corrected_text)
    
    # 5. LLM (Final Response Generation)
    # 검색된 메뉴 정보를 바탕으로 자연스러운 대화 생성
    if menu_result:
        prompt = f"사용자가 '{corrected_text}'라고 주문했습니다. 검색된 메뉴는 '{menu_result[0]}'입니다. 이 메뉴에 대해 친절하게 확인 답변을 해주고 추가 주문 여부를 물어보세요."
    else:
        prompt = f"사용자가 '{corrected_text}'라고 말했지만 메뉴판에서 찾지 못했습니다. 메뉴에 있는지 정중히 확인을 요청하는 답변을 하세요."

    response = client_llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "당신은 친절한 키오스크 AI 도우미입니다."},
                  {"role": "user", "content": prompt}]
    )
    final_msg = response.choices[0].message.content
    
    return final_msg

# --- [메인 실행부] ---
def main():
    speak("안녕하세요! 메가커피입니다. 어떤 메뉴를 도와드릴까요?")
    
    while True:
        input("\n[엔터]를 누르고 주문을 말씀하세요...")
        print("🎤 녹음 중...")
        recording = sd.rec(int(5 * RATE), samplerate=RATE, channels=2)
        sd.wait()
        
        # 통합 파이프라인 가동
        final_response = run_kiosk_pipeline(recording)
        
        # 6. TTS 출력
        speak(final_response)

if __name__ == "__main__":
    main()