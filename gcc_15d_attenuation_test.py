# 실험 1. 0~15도 제한 적응형 빔포밍

import numpy as np
import sounddevice as sd
import time

RATE = 16000
CHANNELS = 2
RECORD_SEC = 0.5       # 한 번 측정할 때 녹음할 시간 (0.5초)
MEASURE_COUNT = 20     # 각도당 반복 측정 횟수 (20번)

MIC_DISTANCE = 0.077   # 7.7cm
SOUND_SPEED = 343.0

# --- [이전과 동일한 빔포밍 함수들] ---
def estimate_delay_gcc_phat(sig1, sig2, fs, max_angle=15):
    n = len(sig1) + len(sig2) - 1
    N_fft = 1 << (n-1).bit_length()
    
    X1, X2 = np.fft.rfft(sig1, n=N_fft), np.fft.rfft(sig2, n=N_fft)
    S_phat = (X1 * np.conj(X2)) / (np.abs(X1 * np.conj(X2)) + 1e-10)
    
    cc = np.fft.irfft(S_phat, n=N_fft)
    cc = np.concatenate((cc[-N_fft//2:], cc[:N_fft//2]))
    center = N_fft // 2
    
    max_tau = (MIC_DISTANCE * np.sin(np.deg2rad(max_angle))) / SOUND_SPEED
    max_shift = int(np.ceil(max_tau * fs))
    search_range = cc[center - max_shift : center + max_shift + 1]
    
    shift = np.argmax(np.abs(search_range)) - max_shift
    return shift / fs

def apply_beamforming(stereo_data, max_angle=15):
    left, right = stereo_data[:, 0], stereo_data[:, 1]
    tau = estimate_delay_gcc_phat(left, right, RATE, max_angle)
    
    delay_samples = int(np.round(tau * RATE))
    half_delay = delay_samples // 2
    
    aligned_left = np.roll(left, -half_delay)
    aligned_right = np.roll(right, half_delay)
    
    return (aligned_left + aligned_right) / 2.0
# ------------------------------------

def main():
    angles = [0, 15, 30, 45, 60, 90]
    results_db = []

    print("=" * 65)
    print(" 🎙️ [초정밀 실험] 15도 제한 빔포밍 각도별 감쇠량 (20회 평균) ")
    print("=" * 65)
    print(f"👉 측정 방식: 각도별 {RECORD_SEC}초 x {MEASURE_COUNT}회 연속 측정 후 에너지 평균 연산")
    
    for angle in angles:
        input(f"\n📍 스마트폰을 [{angle}도]에 놓고 1kHz를 튼 후 [Enter]를 누르세요...")
        print(f" 🔴 {angle}도 데이터 수집 중... ", end="", flush=True)
        
        linear_powers = []
        
        # 20번 연속 샘플링
        for i in range(MEASURE_COUNT):
            raw = sd.rec(int(RECORD_SEC * RATE), samplerate=RATE, channels=CHANNELS)
            sd.wait()
            
            # 1. 15도 제한 빔포밍 통과
            audio_beam = apply_beamforming(raw, max_angle=15)
            
            # 2. 선형 에너지(Linear Power) 추출 (dB 평균의 오류를 막기 위한 정석)
            rms = np.sqrt(np.mean(np.square(audio_beam)))
            power_linear = rms ** 2
            linear_powers.append(power_linear)
            
            # 진행 상황을 점(.)으로 표시
            print("■", end="", flush=True)
            
        # 3. 20개 샘플의 진짜 에너지 평균을 구한 뒤 마지막에 dB로 환산!
        avg_power_linear = np.mean(linear_powers)
        avg_db = 10 * np.log10(avg_power_linear + 1e-12)
        
        results_db.append(avg_db)
        print(f" [완료] 평균 파워: {avg_db:.2f} dB")

    print("\n" + "=" * 65)
    print(" 📊 최종 초정밀 실험 결과 (엑셀 복사용) ")
    print("=" * 65)
    
    baseline_db = results_db[0]
    
    print("각도(도)\t절대 평균 파워(dB)\t감쇠량(dB)")
    print("-" * 65)
    for i, angle in enumerate(angles):
        abs_db = results_db[i]
        norm_db = abs_db - baseline_db  
        print(f"{angle}\t{abs_db:.2f}\t\t\t{norm_db:.2f}")

if __name__ == "__main__":
    main()
