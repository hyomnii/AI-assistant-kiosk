import numpy as np
import sounddevice as sd
import scipy.signal
import whisper
from gtts import gTTS
import os

from stt_correction_model_hybrid import correct_text
from search_menu import search_menu, client_llm, WHISPER_MODEL_SIZE

# --- [beamforming 파라미터] ---
RATE = 16000
MIC_DISTANCE = 0.077
SOUND_SPEED = 343.0
STT_RECORD_SECONDS = 4.0
STT_CHANNELS = 2

# --- [최종 반영된 beamforming 함수] ---
def apply_ultimate_beamforming(stereo_data, max_angle=15, lambda_val=3.0):
    left = stereo_data[:, 0]
    right = stereo_data[:, 1]

    # [단계 1] GCC-PHAT (15도 눈가리개 적용: 제한된 구간 내에서만 탐색)
    n = len(left) + len(right) - 1
    n_fft = 1 << (n - 1).bit_length()
    x1 = np.fft.rfft(left, n=n_fft)
    x2 = np.fft.rfft(right, n=n_fft)
    s_phat = (x1 * np.conj(x2)) / (np.abs(x1 * np.conj(x2)) + 1e-10)
    cc = np.fft.irfft(s_phat, n=n_fft)
    cc = np.concatenate((cc[-n_fft // 2 :], cc[: n_fft // 2]))
    center = n_fft // 2

    # 처음부터 15도 한계(limit) 안에서만 찾기
    limit_tau = (MIC_DISTANCE * np.sin(np.deg2rad(max_angle))) / SOUND_SPEED
    limit_shift = int(np.ceil(limit_tau * RATE))
    search_range = cc[center - limit_shift : center + limit_shift + 1]
    
    # 15도 안에서 가장 큰 시간차(진원지) 찾기
    true_shift = np.argmax(np.abs(search_range)) - limit_shift

    # [단계 2] 대칭적 위상 정렬
    shift_l = -(true_shift // 2)
    shift_r = true_shift + shift_l
    aligned_left = np.roll(left, shift_l)
    aligned_right = np.roll(right, shift_r)

    y_sum = (aligned_left + aligned_right) / 2.0
    y_diff = (aligned_left - aligned_right) / 2.0

    # [단계 3] STFT 기반 주파수 영역 핀셋 감쇠
    f_bins, t_frames, Z_sum = scipy.signal.stft(y_sum, fs=RATE, nperseg=512)
    _, _, Z_diff = scipy.signal.stft(y_diff, fs=RATE, nperseg=512)

    mag_sum = np.abs(Z_sum)
    mag_diff = np.abs(Z_diff)

    # 주파수 대역별 가중치 계산 및 적용
    W_freq = np.clip(1.0 - lambda_val * (mag_diff / (mag_sum + 1e-10)), 0.05, 1.0)
    Z_final = Z_sum * W_freq

    # 최종 시간 영역 신호로 복원
    _, combined = scipy.signal.istft(Z_final, fs=RATE)
    
    # 원본 길이와 맞추기
    combined = combined[:len(y_sum)]
    
    return combined.astype(np.float32)


def listen_with_beamforming(stt_model):
    print("\n엔터를 누르면 음성 입력을 시작합니다. 종료하려면 Ctrl+C 또는 '종료'를 말하세요.")
    input()
    print(f"{STT_RECORD_SECONDS:.0f}초 동안 듣는 중...")

    stereo_audio = sd.rec(
        int(STT_RECORD_SECONDS * RATE),
        samplerate=RATE,
        channels=STT_CHANNELS,
        dtype="float32",
    )
    sd.wait()

    if stereo_audio.size == 0:
        return ""

    beam_audio = apply_ultimate_beamforming(stereo_audio)

    result = stt_model.transcribe(beam_audio, language="ko", fp16=False)
    return (result.get("text") or "").strip()

def generate_kiosk_response(menu_results, user_query):
    """
    RAG 결과(menu_results)를 바탕으로 LLM이 답변 생성
    """
    if menu_results:
        # 결과가 1개인 경우와 여러 개인 경우를 LLM이 판단하도록 함
        menu_names = ", ".join(menu_results)
        prompt = f"사용자의 주문: '{user_query}', 검색된 메뉴: [{menu_names}]. \
                   이 정보를 바탕으로 손님에게 메뉴가 맞는지 확인하는 질문을 한 문장으로 친절하게 하세요."
    else:
        prompt = f"사용자의 주문: '{user_query}'. 메뉴판에 없는 메뉴입니다. 정중하게 메뉴가 없음을 알리고 다른 메뉴를 권유하세요."

    # OpenAI LLM 호출 (이미 search_menu.py에 정의된 client_llm 활용)
    response = client_llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "너는 메가커피의 친절한 AI 점원이야."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def speak_tts(text):
    """텍스트를 음성으로 변환하여 재생"""
    print(f"🤖 AI 점원: {text}")
    tts = gTTS(text=text, lang='ko')
    tts.save("kiosk_voice.mp3")
    os.system("start kiosk_voice.mp3")

# --- 실제 루프 적용 예시 ---
# 1. RAG 결과가 나왔다면
# search_results = search_menu(corrected_text)

# 2. LLM이 답변 생성
# final_answer = generate_kiosk_response(search_results, corrected_text)

# 3. TTS로 출력
# speak_tts(final_answer)

if __name__ == "__main__":
    print("Whisper STT 모델을 로딩합니다...")
    stt_model = whisper.load_model(WHISPER_MODEL_SIZE)

    while True:
        try:
            # 1. 음성 녹음 및 빔포밍 STT
            raw_text = listen_with_beamforming(stt_model)
        except KeyboardInterrupt:
            print("\n종료합니다.")
            break
        except Exception as e:
            print(f"음성 입력 오류: {e}")
            continue

        if not raw_text.strip():
            continue

        print(f"인식된 문장: {raw_text}")

        # 종료 키워드 체크
        if raw_text.strip() in ["종료", "취소", "exit", "quit"]:
            speak_tts("프로그램을 종료합니다. 감사합니다.")
            break

        # 2. 하이브리드 후교정
        corrected_text = correct_text(raw_text)
        print(f"교정된 문장: {corrected_text}")

        # 3. RAG 메뉴 검색
        result = search_menu(corrected_text)

        # 4. LLM 응답 생성 및 TTS 출력
        if not result:
            # 검색 결과가 없을 때 (빈 리스트 [] 전달)
            final_answer = generate_kiosk_response([], corrected_text)
            speak_tts(final_answer)
        else:
            # 검색 결과가 있을 때 (메뉴 리스트 전달)
            # result가 객체 리스트라면 필요한 정보(메뉴명 등)만 뽑아서 전달하는 것이 좋습니다.
            final_answer = generate_kiosk_response(result, corrected_text)
            speak_tts(final_answer)

        print("-" * 50)
        

