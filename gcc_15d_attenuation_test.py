import numpy as np
import sounddevice as sd
import time

# --- [실험 및 공학적 파라미터] ---
RATE = 16000
CHANNELS = 2
RECORD_SEC = 0.5       # 한 번 측정할 때 녹음할 시간
MEASURE_COUNT = 20     # 🌟 요청하신 대로 20번 연속 측정으로 세팅!

MIC_DISTANCE = 0.077   # 마이크 간격 7.7cm
SOUND_SPEED = 343.0

# --- [궁극의 빔포밍 함수 (팀원 아이디어 적용)] ---
def apply_ultimate_beamforming(stereo_data, max_angle=15, lambda_val=2.0):
    """
    1. Restricted Search로 15도 내의 타겟만 추적
    2. Diff(차) 신호의 에너지를 계산하여 측면 소음일 경우 Gain을 동적으로 깎아버림
    """
    left = stereo_data[:, 0]
    right = stereo_data[:, 1]

    # [단계 1] GCC-PHAT 계산
    n = len(left) + len(right) - 1
    N_fft = 1 << (n-1).bit_length()

    X1 = np.fft.rfft(left, n=N_fft)
    X2 = np.fft.rfft(right, n=N_fft)
    S = X1 * np.conj(X2)
    S_phat = S / (np.abs(S) + 1e-10)

    cc = np.fft.irfft(S_phat, n=N_fft)
    cc = np.concatenate((cc[-N_fft//2:], cc[:N_fft//2]))
    center = N_fft // 2

    # [단계 2] 15도 블라인더(Restricted Search) 씌우기
    max_tau = (MIC_DISTANCE * np.sin(np.deg2rad(max_angle))) / SOUND_SPEED
    max_shift = int(np.ceil(max_tau * RATE))
    search_range = cc[center - max_shift : center + max_shift + 1]

    shift = np.argmax(np.abs(search_range)) - max_shift
    tau = shift / RATE

    # [단계 3] 위상 정렬 (Delay-and-Sum 준비)
    delay_samples = int(np.round(tau * RATE))
    half_delay = delay_samples // 2

    aligned_left = np.roll(left, -half_delay)
    aligned_right = np.roll(right, half_delay)

    # [단계 4] 합(Sum)과 차(Diff) 신호 생성
    y_sum = (aligned_left + aligned_right) / 2.0
    y_diff = (aligned_left - aligned_right) / 2.0

    # [단계 5] Diff 에너지를 이용한 공간 게이팅 (Spatial Gating)
    rms_sum = np.sqrt(np.mean(np.square(y_sum)) + 1e-10)
    rms_diff = np.sqrt(np.mean(np.square(y_diff)) + 1e-10)

    # 측면 소음일수록 rms_diff가 커짐 -> penalty가 커짐
    penalty = lambda_val * (rms_diff / rms_sum)
    
    # Gain을 최소 0.05(-26dB)부터 최대 1.0(0dB) 사이로 제한
    gain = np.clip(1.0 - penalty, 0.05, 1.0) 

    # 최종 출력
    combined = y_sum * gain
    return combined.astype(np.float32)

# --- [메인 실험 루프] ---
def main():
    # 🌟 요청하신 각도로 세팅 완료!
    angles = [0, 15, 30, 45, 60, 90]
    results_db = []

    print("=" * 65)
    print(" 🎙️ [초정밀 실험 재도전] 15도 제한 + 차(Diff) 감쇠 빔포밍 ")
    print("=" * 65)
    
    for angle in angles:
        # 주의: 1kHz 대신 꼭 백색소음이나 사람 음성(TTS)으로 테스트하세요!
        input(f"\n📍 스마트폰을 [{angle}도]에 놓고 [백색소음/음성]을 튼 후 [Enter]를 누르세요...")
        print(f" 🔴 {angle}도 데이터 수집 중... ", end="", flush=True)
        
        linear_powers = []
        
        # 20번 연속 샘플링
        for i in range(MEASURE_COUNT):
            raw = sd.rec(int(RECORD_SEC * RATE), samplerate=RATE, channels=CHANNELS)
            sd.wait()
            
            # 궁극의 빔포밍 적용 (0도가 너무 깎이면 lambda_val=1.0 으로 낮춰보세요)
            audio_beam = apply_ultimate_beamforming(raw, max_angle=15, lambda_val=2.0)
            
            # 선형 에너지(Linear Power) 추출
            rms = np.sqrt(np.mean(np.square(audio_beam)))
            power_linear = rms ** 2
            linear_powers.append(power_linear)
            
            # 진행 상황을 점(■)으로 표시
            print("■", end="", flush=True)
            
        # 20개 샘플의 진짜 에너지 평균을 구한 뒤 마지막에 dB로 환산!
        avg_power_linear = np.mean(linear_powers)
        avg_db = 10 * np.log10(avg_power_linear + 1e-12)
        
        results_db.append(avg_db)
        print(f" [완료] 평균 파워: {avg_db:.2f} dB")

    print("\n" + "=" * 65)
    print(" 📊 최종 초정밀 실험 결과 (엑셀 복사용) ")
    print("=" * 65)
    
    # 0도를 기준점으로 정규화(Normalization)
    baseline_db = results_db[0]
    
    print("각도(도)\t절대 평균 파워(dB)\t감쇠량(dB)")
    print("-" * 65)
    for i, angle in enumerate(angles):
        abs_db = results_db[i]
        norm_db = abs_db - baseline_db  
        print(f"{angle}\t{abs_db:.2f}\t\t\t{norm_db:.2f}")

if __name__ == "__main__":
    main()