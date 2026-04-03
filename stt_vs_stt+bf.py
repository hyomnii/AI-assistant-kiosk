import queue, threading, time
import numpy as np
import sounddevice as sd
import whisper

# --- 1. 공학적 설정값 (실측 데이터 기반) ---
RATE = 16000
CHANNELS = 2
MODEL_SIZE = "base"

# [핵심] 노트북 마이크 사이의 물리적 거리를 상수로 정의
MIC_DISTANCE = 0.077  # 실측값 7.7cm 반영
SOUND_SPEED = 343.0   # 음속 (m/s)

# 임계치 및 발화 감지 설정
THRESHOLD_DB = -45.0  # 이 값보다 작은 소리는 무시
RECORD_SEC = 5.0      # 5초 단위 녹음

# # --- 2. 빔포밍 및 사운드 분석 함수 ---
# def get_rms_db(audio_data):
#     """오디오 신호의 에너지를 데시벨(dB)로 환산"""
#     rms = np.sqrt(np.mean(np.square(audio_data)))
#     return 20 * np.log10(rms + 1e-12)

# def apply_beamforming(stereo_data, target_angle=0):
#     """
#     [지연 후 합산 빔포밍]
#     실측 거리(MIC_DISTANCE)를 기반으로 정면(0도) 소리 강화
#     """
#     # 1. 수식에 필요한 파라미터
#     d = MIC_DISTANCE  # 마이크 거리 : 0.077m
#     v = SOUND_SPEED   # 음속 : 343.0m/s
#     fs = RATE         # 샘플 rate : 16000Hz
    
#     # 2. 각도(0도)에 따른 시간 지연(tau) 계산
#     # tau = (d * sin(theta)) / v
#     radian = np.deg2rad(target_angle)
#     tau = (d * np.sin(radian)) / v
    
#     # 3. tau(지연 시간)을 샘플 단위로 변환 (0도일 땐 0이 됨)
#     delay_samples = int(tau * fs)
    
#     left = stereo_data[:, 0]
#     right = stereo_data[:, 1]
    
#     # 4. 오른쪽 채널에 지연 적용 후 합산 (Delay-and-Sum)
#     # 정면 0도일 때는 delay_samples가 0이 되어 결국 두 신호를 그대로 더하게 됨
#     shifted_right = np.roll(right, delay_samples)
#     # 보강 간섭 (정면 신호 살리기) / 상쇄 간섭 (측면 소음 죽이기)
#     combined = (left + shifted_right) / 2 
    
#     return combined.astype(np.float32)

# --- 2. 빔포밍 및 사운드 분석 함수 ---
def get_rms_db(audio_data):
    """오디오 신호의 에너지를 데시벨(dB)로 환산"""
    rms = np.sqrt(np.mean(np.square(audio_data)))
    return 20 * np.log10(rms + 1e-12)

# def estimate_delay_gcc_phat(sig1, sig2, fs):
#     """GCC-PHAT를 이용해 두 신호 간의 실제 도달 시간차(Delay)를 동적으로 추정"""
#     n = len(sig1) + len(sig2) - 1
#     N_fft = 1 << (n-1).bit_length() # 2의 거듭제곱으로 패딩
    
#     # 주파수 도메인 변환 및 교차 스펙트럼 계산
#     X1 = np.fft.rfft(sig1, n=N_fft)
#     X2 = np.fft.rfft(sig2, n=N_fft)
#     S = X1 * np.conj(X2)
    
#     # PHAT 가중치 적용 (위상 정보만 추출하여 노이즈에 강함)
#     S_phat = S / (np.abs(S) + 1e-10)
    
#     # 시간 도메인으로 복원 후 최대 상관값을 가지는 딜레이 찾기
#     cc = np.fft.irfft(S_phat, n=N_fft)
#     cc = np.concatenate((cc[-N_fft//2:], cc[:N_fft//2]))
#     shift = np.argmax(np.abs(cc)) - N_fft//2
    
#     # 시간차(초) 반환
#     tau = shift / fs
#     return tau

def estimate_delay_gcc_phat(sig1, sig2, fs, max_angle=15):
    """
    [제한적 GCC-PHAT]
    두 신호 간의 시간차(Delay)를 추정하되, 키오스크 환경에 맞춰
    지정된 각도(max_angle, 기본값 좌우 15도) 내의 소리만 추적함.
    """
    n = len(sig1) + len(sig2) - 1
    N_fft = 1 << (n-1).bit_length() # 2의 거듭제곱으로 패딩
    
    # 1. 주파수 도메인 변환 및 교차 스펙트럼 계산
    X1 = np.fft.rfft(sig1, n=N_fft)
    X2 = np.fft.rfft(sig2, n=N_fft)
    S = X1 * np.conj(X2)
    
    # 2. PHAT 가중치 적용 (위상 정보만 추출)
    S_phat = S / (np.abs(S) + 1e-10)
    
    # 3. 시간 도메인으로 복원
    cc = np.fft.irfft(S_phat, n=N_fft)
    cc = np.concatenate((cc[-N_fft//2:], cc[:N_fft//2]))
    
    # --- [🚨핵심 수정 부분: 탐색 범위 제한] ---
    center = N_fft // 2
    
    # 허용할 최대 각도(max_angle)에 따른 최대 지연 샘플 수 계산
    max_tau = (MIC_DISTANCE * np.sin(np.deg2rad(max_angle))) / SOUND_SPEED
    max_shift = int(np.ceil(max_tau * fs)) # 올림 처리로 안전한 탐색 범위 확보
    
    # 제한된 범위(-max_shift ~ +max_shift) 내에서만 cc 배열 잘라내기
    search_range = cc[center - max_shift : center + max_shift + 1]
    
    # 잘라낸 범위 안에서만 가장 뚜렷한 소리의 위치(local_shift) 찾기
    local_shift = np.argmax(np.abs(search_range))
    
    # 실제 shift 값(-max_shift ~ +max_shift)으로 복원
    shift = local_shift - max_shift
    # ------------------------------------------
    
    # 시간차(초) 반환
    tau = shift / fs
    return tau

def apply_beamforming(stereo_data):
    """
    [적응형 빔포밍 - GCC-PHAT 적용]
    실시간으로 두 마이크 간의 시간차(tau)를 추정하여 위상을 정렬한 뒤 합산
    """
    fs = RATE
    left = stereo_data[:, 0]
    right = stereo_data[:, 1]
    
    # 1. 실시간 도달 시간차(tau) 계산
    tau = estimate_delay_gcc_phat(left, right, fs)
    
    # 2. tau를 샘플 수로 변환
    delay_samples = int(np.round(tau * fs))
    
    # 3. 양쪽 채널을 절반씩 이동하여 중앙(가상 마이크) 기준으로 위상 정렬
    half_delay = delay_samples // 2
    
    aligned_left = np.roll(left, -half_delay)
    aligned_right = np.roll(right, half_delay)
    
    # 4. 합산 (상쇄 간섭 및 보강 간섭 발생)
    combined = (aligned_left + aligned_right) / 2.0 
    
    return combined.astype(np.float32)

# --- 3. STT 프로세서 함수 (함수명 확인 완료) ---
def stt_processor(model, job_q):
    while True:
        audio_a, audio_b, current_db = job_q.get()
        if audio_a is None: break
        
        try:
            # 모드 A는 무조건 전사
            res_a = model.transcribe(audio_a, language="ko", fp16=False)
            
            # 모드 B는 임계값을 넘었을 때만 전사 (오인식 방지)
            if current_db < THRESHOLD_DB:
                res_b_text = "(무시됨 - 정면 음성 크기 미달)"
            else:
                res_b = model.transcribe(audio_b, language="ko", fp16=False)
                res_b_text = res_b['text'].strip()

            print("\n" + "="*60)
            print(f"✅ 분석 결과 (에너지: {current_db:.1f} dB)")
            print(f" [일반 모드 A] : {res_a['text'].strip()}")
            print(f" [빔포밍 모드 B] : {res_b_text}")
            print("="*60)
        except Exception as e:
            print(f"❌ 전사 오류: {e}")
        
        job_q.task_done()

# --- 4. 메인 실행부 ---
def main():
    print(f"🚀 실측 데이터({MIC_DISTANCE*100}cm) 기반 시스템 로딩 중...")
    model = whisper.load_model(MODEL_SIZE)
    
    job_q = queue.Queue()
    threading.Thread(target=stt_processor, args=(model, job_q), daemon=True).start()

    print("\n" + "*"*50)
    print("  AI 키오스크 음성 인식 비교 시스템 (7.7cm 최적화)  ")
    print("*"*50)

    try:
        while True:
            print(f"\n🎤 [준비] 잠시 후 {int(RECORD_SEC)}초 녹음을 시작합니다...")
            time.sleep(1.5)
            
            print(f"🔴 >>> 지금 말씀하세요! (정면 0도 지향)")
            
            # 5초간 녹음
            raw_recording = sd.rec(int(RECORD_SEC * RATE), samplerate=RATE, channels=CHANNELS)
            sd.wait()
            
            print("⏳ 녹음 완료! 분석 중입니다...")

            # 데이터 분리 및 전처리
            audio_a = np.mean(raw_recording, axis=1) # 일반 모드
            audio_b = apply_beamforming(raw_recording) # 빔포밍 모드
            current_db = get_rms_db(audio_b)
            
            # 분석 큐에 전달
            job_q.put((audio_a, audio_b, current_db))
            
            # 분석 결과가 출력될 때까지 잠시 대기 (출력 순서 꼬임 방지)
            job_q.join()

    except KeyboardInterrupt:
        print("\n👋 프로그램을 종료합니다.")

if __name__ == "__main__":
    main()