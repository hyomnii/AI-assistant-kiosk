import numpy as np
import sounddevice as sd
import time

# --- [실험 및 공학적 파라미터] ---
RATE = 16000
CHANNELS = 2
RECORD_SEC = 3.0       # 🌟 3초로 확장 (신호의 안정성 확보)
MEASURE_COUNT = 1      # 🌟 1회 측정으로 단축 (실험 효율화)

MIC_DISTANCE = 0.077   
SOUND_SPEED = 343.0

# --- [수식 1~6번 완벽 매칭 빔포밍 함수] ---
def apply_ultimate_beamforming(stereo_data, max_angle=15, lambda_val=3.0):
    left = stereo_data[:, 0]
    right = stereo_data[:, 1]

    # [식 1] GCC-PHAT (도달 각도 추정)
    n = len(left) + len(right) - 1
    N_fft = 1 << (n-1).bit_length()
    X1, X2 = np.fft.rfft(left, n=N_fft), np.fft.rfft(right, n=N_fft)
    S_phat = (X1 * np.conj(X2)) / (np.abs(X1 * np.conj(X2)) + 1e-10)
    cc = np.fft.irfft(S_phat, n=N_fft)
    cc = np.concatenate((cc[-N_fft//2:], cc[:N_fft//2]))
    center = N_fft // 2

    full_max_tau = MIC_DISTANCE / SOUND_SPEED
    full_max_shift = int(np.ceil(full_max_tau * RATE))
    full_search_range = cc[center - full_max_shift : center + full_max_shift + 1]
    true_shift = np.argmax(np.abs(full_search_range)) - full_max_shift

    # [식 2] Hard Cutoff (15도 제한 필터 g)
    target_max_tau = (MIC_DISTANCE * np.sin(np.deg2rad(max_angle))) / SOUND_SPEED
    target_max_shift = int(np.ceil(target_max_tau * RATE))
    g_theta = 1.0 if abs(true_shift) <= target_max_shift else 0.05

    # [식 3] 대칭적 위상 정렬 (Symmetric Alignment)
    shift_L = -(true_shift // 2)
    shift_R = true_shift + shift_L
    aligned_left = np.roll(left, shift_L)
    aligned_right = np.roll(right, shift_R)

    # [식 4] 합(Sum) 및 차(Diff) 신호 추출
    y_sum = (aligned_left + aligned_right) / 2.0
    y_diff = (aligned_left - aligned_right) / 2.0

    # [식 5] 동적 감쇠 가중치(W) 산출
    rms_sum = np.sqrt(np.mean(np.square(y_sum)) + 1e-10)
    rms_diff = np.sqrt(np.mean(np.square(y_diff)) + 1e-10)
    penalty = lambda_val * (rms_diff / rms_sum)
    W = np.clip(1.0 - penalty, 0.05, 1.0) 

    # [식 6] 최종 출력 신호
    combined = g_theta * W * y_sum
    return combined.astype(np.float32)

# --- [메인 실험 루프] ---
def main():
    # 🌟 요청하신 음수 각도 포함 전체 순서
    angles = [-90, -60, -45, -30, -15, 0, 15, 30, 45, 60, 90]
    results_db = []

    print("=" * 75)
    print(f" 🎙️ [최종 검증] 3초 1회 측정 모드 (람다={3.0}) ")
    print("=" * 75)
    
    for angle in angles:
        input(f"\n📍 [{angle}도] 세팅 후 [Enter]를 누르면 3초간 측정을 시작합니다...")
        print(f" 🔴 측정 중... ", end="", flush=True)
        
        # 3초간 녹음
        raw = sd.rec(int(RECORD_SEC * RATE), samplerate=RATE, channels=CHANNELS)
        sd.wait()
        
        # 빔포밍 적용
        audio_beam = apply_ultimate_beamforming(raw, max_angle=15, lambda_val=3.0)
        
        # 에너지 계산
        rms = np.sqrt(np.mean(np.square(audio_beam)))
        power_db = 20 * np.log10(rms + 1e-12)
        
        results_db.append(power_db)
        print(f" [완료] 측정 파워: {power_db:.2f} dB")

    # --- [정규화 및 결과 출력] ---
    print("\n" + "=" * 75)
    print(" 📊 최종 실험 결과 (자동 0도 기준 정규화) ")
    print("=" * 75)
    
    # 🌟 리스트에서 0도의 인덱스를 찾아 기준점(Baseline)으로 설정
    try:
        zero_idx = angles.index(0)
        baseline_db = results_db[zero_idx]
    except ValueError:
        baseline_db = results_db[0]
        print("⚠️ 주의: 각도 목록에 0도가 없어 첫 번째 데이터를 기준으로 정규화합니다.")
    
    print(f"{'각도(Angle)':^10} | {'절대 파워(dB)':^15} | {'감쇠량(Normalized)':^15}")
    print("-" * 75)
    for i, angle in enumerate(angles):
        abs_db = results_db[i]
        norm_db = abs_db - baseline_db  # 0도 대비 상대적 감쇠량
        print(f"{angle:^12} | {abs_db:^15.2f} | {norm_db:^15.2f} dB")

if __name__ == "__main__":
    main()