import time
import numpy as np
import sounddevice as sd
import whisper
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
from beam_final import apply_ultimate_beamforming, RATE

RECORD_SECONDS = 5.0
CHANNELS = 2
WHISPER_MODEL = "base"
TRIAL_COUNT = 10

def get_rms_db(audio_data):
    """오디오 신호의 에너지를 데시벨(dB)로 환산"""
    rms = np.sqrt(np.mean(np.square(audio_data)))
    return 20 * np.log10(rms + 1e-12)

def measure_db_phase(phase_name, duration=3.0):
    """특정 환경의 데시벨을 측정하는 함수"""
    print(f"\n[{phase_name} 데시벨 측정]")
    input("준비가 되면 엔터를 눌러주세요...")
    print("측정 중... (3초)")
    
    audio = sd.rec(int(duration * RATE), samplerate=RATE, channels=CHANNELS, dtype='float32')
    sd.wait()
    
    db = get_rms_db(audio)
    print(f"✅ 측정 완료: 평균 {db:.2f} dB")
    return db

def run_experiment():
    print("==================================================")
    print("🎙️ 빔포밍 성능 검증: 소음 혼합 환경 10회 연속 테스트 🎙️")
    print("==================================================\n")
    
    # 1. 모델 로드
    print("Whisper 모델 로딩 중 (base)...")
    stt_model = whisper.load_model(WHISPER_MODEL)
    print("로딩 완료!\n")

    # 2. 데시벨 측정
    print("--- 1단계: 환경 데시벨 확인 ---")
    print("안내: 측면에서 유튜브 소음만 튼 상태로 측정합니다.")
    noise_db = measure_db_phase("측면 방해 소음 (유튜브)")

    print("\n안내: 유튜브를 끄고, 정면에서 주문 목소리만 냅니다.")
    speech_db = measure_db_phase("정면 타겟 음성 (주문자)")

    print(f"\n[환경 기록] 측면 소음: {noise_db:.1f}dB | 정면 음성: {speech_db:.1f}dB")
    print("이 수치는 최종 결과 리포트에 함께 기록됩니다.\n")

    # 3. 본 실험 (10회 연속 반복)
    print("--- 2단계: 10회 연속 실전 테스트 ---")
    print("안내: 이제 유튜브 소음을 계속 틀어둔 상태에서 테스트를 진행합니다.")
    
    # 결과를 저장할 리스트
    results = []

    for i in range(1, TRIAL_COUNT + 1):
        input(f"\n▶ [ {i} / {TRIAL_COUNT} 회차 ] 준비되면 엔터를 누르고 주문을 말하세요...")
        print(f"🔴 {RECORD_SECONDS}초간 녹음 중...")
        
        mixed_audio = sd.rec(int(RECORD_SECONDS * RATE), samplerate=RATE, channels=CHANNELS, dtype='float32')
        sd.wait()
        print("녹음 완료! 분석 중...")

        # 3-1. 분석 1: Raw Whisper (비교군)
        raw_mono_audio = np.squeeze(mixed_audio[:, 0]).astype(np.float32)
        result_raw = stt_model.transcribe(raw_mono_audio, language="ko", fp16=False)
        text_raw = result_raw.get("text", "").strip()

        # 3-2. 분석 2: 빔포밍 + Whisper (실험군)
        beamformed_audio = apply_ultimate_beamforming(mixed_audio, max_angle=15, lambda_val=2.0)
        result_beam = stt_model.transcribe(beamformed_audio, language="ko", fp16=False)
        text_beam = result_beam.get("text", "").strip()

        # 결과 저장
        results.append((text_raw, text_beam))
        print(f"✅ {i}회차 분석 완료!")

    # 4. 최종 통합 결과 출력
    print("\n\n==================================================")
    print("📊 10회 연속 실험 최종 결과 리포트 📊")
    print("==================================================")
    print(f"[실험 환경] 측면 유튜브 소음: {noise_db:.1f}dB | 정면 주문 음성: {speech_db:.1f}dB")
    print("-" * 50)
    
    for idx, (raw_res, beam_res) in enumerate(results, 1):
        print(f"[{idx}회차]")
        print(f" ❌ 적용 전: {raw_res}")
        print(f" ✅ 적용 후: {beam_res}")
        print("-" * 50)
    
    print("==================================================\n")
    print("수고하셨습니다. 위 결과를 복사해서 PPT나 엑셀에 활용하세요!")

if __name__ == "__main__":
    run_experiment()