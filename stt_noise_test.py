import numpy as np
import sounddevice as sd
import whisper
import time

# --- [공학적 파라미터] ---
RATE = 16000
CHANNELS = 2
RECORD_SEC_CALIB = 5.0  # 캘리브레이션 측정 시간 (5초)
RECORD_SEC_EXP = 6.0    # 🌟 본 실험 녹음 시간 (4초 -> 6초로 연장 완료!)
MIC_DISTANCE = 0.077    # 마이크 간격 (7.7cm)
SOUND_SPEED = 343.0
THRESHOLD_DB = -45.0    # 환각 방지 임계값
MODEL_SIZE = "base"
TRIALS = 3

# --- [1. 기본 신호 처리 및 공간 게이팅 함수] ---
def get_rms_db(audio_data):
    rms = np.sqrt(np.mean(np.square(audio_data)))
    return 20 * np.log10(rms + 1e-12)

def get_true_angle(sig1, sig2, fs):
    """GCC-PHAT으로 진짜 도달각(DOA) 추적"""
    n = len(sig1) + len(sig2) - 1
    N_fft = 1 << (n-1).bit_length()
    
    X1, X2 = np.fft.rfft(sig1, n=N_fft), np.fft.rfft(sig2, n=N_fft)
    S_phat = (X1 * np.conj(X2)) / (np.abs(X1 * np.conj(X2)) + 1e-10)
    
    cc = np.fft.irfft(S_phat, n=N_fft)
    cc = np.concatenate((cc[-N_fft//2:], cc[:N_fft//2]))
    center = N_fft // 2
    
    shift = np.argmax(np.abs(cc)) - center
    tau = shift / fs
    
    val = (tau * SOUND_SPEED) / MIC_DISTANCE
    val = np.clip(val, -1.0, 1.0)
    true_angle = np.rad2deg(np.arcsin(val))
    return true_angle, tau

def apply_smart_beamforming(stereo_data, target_limit=15):
    """실험 1의 오답노트가 반영된 공간 게이팅 빔포밍"""
    left, right = stereo_data[:, 0], stereo_data[:, 1]
    true_angle, tau = get_true_angle(left, right, RATE)
    
    delay_samples = int(np.round(tau * RATE))
    half_delay = delay_samples // 2
    aligned_left = np.roll(left, -half_delay)
    aligned_right = np.roll(right, half_delay)
    combined = (aligned_left + aligned_right) / 2.0
    
    # 목표 각도(15도)를 벗어나면 에너지를 강제로 10%(-20dB) 수준으로 감쇠
    if abs(true_angle) > target_limit:
        combined = combined * 0.1 
        
    return combined

# --- [2. 캘리브레이션 (dB 측정) 모드] ---
def run_calibration():
    print("\n" + "="*50)
    print(" 🎚️ [캘리브레이션 모드] 볼륨(dB) 세팅 ")
    print("="*50)
    print("목소리(20cm)와 소음(50cm)의 dB를 비교하여 볼륨을 조절하세요.")
    print("종료하려면 엔터 대신 'q'를 입력하세요.\n")
    
    target_db = 0
    while True:
        cmd = input("📍 측정할 음원(목소리 or 주파수)을 틀고 [Enter]를 누르세요 ('q'로 종료): ")
        if cmd.lower() == 'q': break
        
        print(f" 🔴 {int(RECORD_SEC_CALIB)}초간 데시벨 측정 중...", end="", flush=True)
        raw = sd.rec(int(RECORD_SEC_CALIB * RATE), samplerate=RATE, channels=CHANNELS)
        sd.wait()
        
        audio_mono = np.mean(raw, axis=1)
        current_db = get_rms_db(audio_mono)
        
        print(f" [완료] 현재 볼륨: {current_db:.2f} dB")
        if target_db == 0:
            target_db = current_db
            print("   ↳ (이 값을 20cm 목소리 기준으로 삼고, 50cm 소음은 이보다 1~2dB 크게 세팅해보세요!)\n")

# --- [3. 본 실험 모드] ---
def run_main_experiment():
    print("\n📦 Whisper 모델 로딩 중...")
    model = whisper.load_model(MODEL_SIZE)
    
    voice_angles = [0, 15, 45, 90]
    noise_angles = [45, 60, 90]
    
    print("\n" + "="*65)
    print(" 🔬 [실험 2] 동시 발화 악조건(Noise > Voice) STT 인식 평가 ")
    print("="*65)
    
    for v_angle in voice_angles:
        for n_angle in noise_angles:
            condition = f"Voice({v_angle}도, 20cm) + Noise({n_angle}도, 50cm)"
            print("\n" + "─"*65)
            print(f"🚀 다음 환경: {condition}")
            
            for i in range(1, TRIALS + 1):
                input(f"  [{i}회차] 두 음원 세팅 완료 후 [Enter]를 누르면 녹음 시작...")
                # 🌟 안내 멘트도 연장된 시간에 맞게 6초로 출력됩니다!
                print(f"  🔴 {int(RECORD_SEC_EXP)}초간 녹음 중...", end="", flush=True)
                
                raw = sd.rec(int(RECORD_SEC_EXP * RATE), samplerate=RATE, channels=CHANNELS)
                sd.wait()
                print(" [완료] 분석 중...")
                
                # 1. 일반 모델 (전처리 없음)
                audio_raw = np.mean(raw, axis=1)
                db_raw = get_rms_db(audio_raw)
                txt_raw = model.transcribe(audio_raw, language="ko", fp16=False)['text'].strip()
                
                # 2. 전처리 모델 (공간 게이팅 빔포밍 + dB Threshold)
                audio_beam = apply_smart_beamforming(raw, target_limit=15)
                db_beam = get_rms_db(audio_beam)
                
                # dB Threshold 적용 (환각 방지)
                if db_beam < THRESHOLD_DB:
                    txt_beam = "(무음 컷오프 - 환각 방지됨)"
                else:
                    txt_beam = model.transcribe(audio_beam, language="ko", fp16=False)['text'].strip()
                
                print(f"   ↳ [일반 모드] 파워: {db_raw:.1f}dB | STT: {txt_raw}")
                print(f"   ↳ [개선 모델] 파워: {db_beam:.1f}dB | STT: {txt_beam}\n")

# --- [메인 메뉴] ---
def main():
    while True:
        print("\n" + "*"*50)
        print(" 1. 볼륨 캘리브레이션 (dB 측정 및 세팅)")
        print(" 2. 본 실험 시작 (Whisper STT 비교)")
        print(" 3. 프로그램 종료")
        print("*"*50)
        
        choice = input("원하는 모드의 번호를 입력하세요: ")
        
        if choice == '1':
            run_calibration()
        elif choice == '2':
            run_main_experiment()
        elif choice == '3':
            print("프로그램을 종료합니다.")
            break
        else:
            print("잘못된 입력입니다.")

if __name__ == "__main__":
    main()