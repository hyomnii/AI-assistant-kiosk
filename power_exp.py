import sounddevice as sd
import numpy as np
import time

# --- 실험 환경 설정 ---
RATE = 16000
CHANNELS = 2
RECORD_SEC = 3.0  # 각 포인트당 3초씩 녹음

def get_rms_db(audio_data):
    """오디오 신호의 RMS(에너지)를 데시벨(dB)로 환산"""
    rms = np.sqrt(np.mean(np.square(audio_data)))
    return 20 * np.log10(rms + 1e-12)

def main():
    angles = [0, 15, 30, 45, 60, 90]
    results_db = []

    print("=" * 60)
    print(" 🎙️ 키오스크 빔포밍 파워 감소(Array Factor) 측정 실험 ")
    print("=" * 60)
    print("👉 스마트폰에서 1kHz 주파수(Sine Wave)를 틀어주세요.")
    print("👉 각도 위치마다 50cm 거리를 유지해 주세요.\n")

    # 1. 각도별 순회하며 녹음 및 파워 측정
    for angle in angles:
        input(f"📍 스마트폰을 [{angle}도] 위치에 놓고 [Enter]를 누르세요...")
        
        print(f" ⏳ {angle}도 녹음 중... (3초간 소리 유지)")
        # 순수(Raw) 2채널 오디오 녹음
        raw_recording = sd.rec(int(RECORD_SEC * RATE), samplerate=RATE, channels=CHANNELS)
        sd.wait()
        
        # 0도 타겟 고정 빔포밍 (정면만 보도록 고정)
        # 정면은 도달 시간차(tau)가 0이므로, 두 신호를 지연 없이 그대로 더함
        left = raw_recording[:, 0]
        right = raw_recording[:, 1]
        beamformed_audio = (left + right) / 2.0
        
        # 파워(dB) 계산 후 리스트에 저장
        power_db = get_rms_db(beamformed_audio)
        results_db.append(power_db)
        
        print(f" ✅ 측정 완료: {power_db:.2f} dB\n")

    # 2. 결과 출력 및 정규화(Normalization)
    print("=" * 60)
    print(" 📊 최종 실험 결과 요약 (엑셀 복사용) ")
    print("=" * 60)

    
    # 0도일 때의 파워를 기준점(0 dB)으로 맞추기 위한 보정 작업
    baseline_db = results_db[0] 

    print("각도(도)\t절대 파워(dB)\t정규화 파워(dB)")
    print("-" * 60)
    for i, angle in enumerate(angles):
        abs_db = results_db[i]
        # 정규화 파워: 타겟(0도) 대비 측면 소리가 얼마나 감소했는가?
        norm_db = abs_db - baseline_db  
        print(f"{angle}\t{abs_db:.2f}\t\t{norm_db:.2f}")
    print("=" * 60)

if __name__ == "__main__":
    main()

# import numpy as np
# import matplotlib.pyplot as plt

# # 1. 환경 변수 세팅
# N = 2               # 마이크 개수
# d = 0.075           # 마이크 간격 7.5cm (m)
# f = 1000            # 주파수 1kHz
# v = 343             # 음속 (m/s)

# # 2. 이론적 파워 계산 (푸른색 실선)
# theta_deg = np.linspace(-90, 90, 361)
# theta_rad = np.radians(theta_deg)
# k = (2 * np.pi * f) / v
# psi = k * d * np.sin(theta_rad)

# # 0으로 나누는 오류 방지
# gain = np.divide(np.sin(N * psi / 2), np.sin(psi / 2), 
#                  out=np.full_like(psi, float(N)), where=psi!=0)
# theory_power_db = 20 * np.log10(np.abs(gain) / N)

# # 3. 실험 데이터 준비 (좌우 대칭으로 확장)
# angles = np.array([-90, -60, -45, -30, -15, 0, 15, 30, 45, 60, 90])

# # 1차 실험 데이터
# exp1 = np.array([2.00, -1.92, -3.84, -0.41, -1.28, 0.00, -1.28, -0.41, -3.84, -1.92, 2.00])
# # 2차 실험 데이터
# exp2 = np.array([2.21, -2.20, -4.69, -4.48, -8.13, 0.00, -8.13, -4.48, -4.69, -2.20, 2.21])
# # 3차 실험 데이터
# exp3 = np.array([0.21, -5.54, -10.84, -13.21, -6.32, 0.00, -6.32, -13.21, -10.84, -5.54, 0.21])

# # 4. 그래프 그리기
# plt.figure(figsize=(12, 6))

# # 이론 선
# plt.plot(theta_deg, theory_power_db, color='blue', linewidth=3, label='Theoretical Beam Pattern (Math Model)')

# # 실험 1, 2, 3 선 (구분하기 쉽게 색상과 마커 다르게 설정)
# plt.plot(angles, exp1, color='orange', marker='o', linestyle='--', linewidth=1.5, alpha=0.8, label='Trial 1')
# plt.plot(angles, exp2, color='green', marker='s', linestyle='--', linewidth=1.5, alpha=0.8, label='Trial 2')
# plt.plot(angles, exp3, color='red', marker='^', linestyle='-', linewidth=2, label='Trial 3 (Best Alignment)')

# # 그래프 꾸미기
# plt.title('Theoretical vs Measured Beam Power (All Trials)', fontsize=15)
# plt.xlabel('Angle [degree]', fontsize=13)
# plt.ylabel('Normalized Power [dB]', fontsize=13)

# # 0도 기준선 및 격자
# plt.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
# plt.axhline(y=0, color='gray', linestyle=':', linewidth=1)
# plt.grid(True, linestyle=':', alpha=0.7)

# # 범례 위치 조정
# plt.legend(fontsize=11, loc='lower right')

# # Y축, X축 범위 설정
# plt.ylim(-15, 4)
# plt.xlim(-95, 95)

# plt.tight_layout()
# plt.show()