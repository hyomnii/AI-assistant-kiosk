import numpy as np
import matplotlib.pyplot as plt

# 1. 환경 변수 세팅
N = 2               # 마이크 개수
d = 0.075           # 마이크 간격 7.5cm (m)
f = 1000            # 주파수 1kHz
v = 343             # 음속 (m/s)

# 2. 이론적 파워 계산 (푸른색 실선)
theta_deg = np.linspace(-90, 90, 500)
theta_rad = np.radians(theta_deg)
k = (2 * np.pi * f) / v
psi = k * d * np.sin(theta_rad)

# 0으로 나누는 오류 방지
gain = np.divide(np.sin(N * psi / 2), np.sin(psi / 2), 
                 out=np.full_like(psi, float(N)), where=psi!=0)
theory_power_db = 20 * np.log10(np.abs(gain) / N)

# 3. 실험 데이터 준비 (좌우 대칭으로 확장)
angles = np.array([-90, -60, -45, -30, -15, 0, 15, 30, 45, 60, 90])

# 1차 실험 데이터 (사장님 제공 데이터로 교체)
exp1 = np.array([-20.16, -8.53, -16.90, -13.42, -24.99, 0.00, -24.99, -13.42, -16.90, -8.53, -20.16])
# 2차 실험 데이터 (사장님 제공 데이터로 교체)
exp2 = np.array([-21.92, -12.88, -18.73, -19.15, -22.16, 0.00, -22.16, -19.15, -18.73, -12.88, -21.92])

# 4. 실험값 정규화 (0 dB 기준)
exp1_norm_dB = exp1 - np.max(exp1)  # 정규화하여 0 dB에서 시작하도록
exp2_norm_dB = exp2 - np.max(exp2)  # 정규화하여 0 dB에서 시작하도록

# 5. 그래프 그리기
plt.figure(figsize=(12, 6))

# 이론 선
plt.plot(theta_deg, theory_power_db, color='blue', linewidth=3, label='Theoretical Beam Pattern (Math Model)')

# 실험 1, 2 선 (구분하기 쉽게 색상과 마커 다르게 설정)
plt.plot(angles, exp1_norm_dB, color='orange', marker='o', linestyle='--', linewidth=1.5, alpha=0.8, label='Trial 1')
plt.plot(angles, exp2_norm_dB, color='green', marker='s', linestyle='--', linewidth=1.5, alpha=0.8, label='Trial 2')

# 그래프 꾸미기
plt.title('Theoretical vs Measured Beam Power (All Trials)', fontsize=15)
plt.xlabel('Angle [degree]', fontsize=13)
plt.ylabel('Beamforming Gain [dB]', fontsize=13)

# 0도 기준선 및 격자
plt.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
plt.axhline(y=0, color='gray', linestyle=':', linewidth=1)
plt.grid(True, linestyle=':', alpha=0.7)

# 범례 위치 조정
plt.legend(fontsize=11, loc='lower right')

# Y축, X축 범위 설정
plt.ylim(-25, 5)
plt.xlim(-90, 90)

plt.tight_layout()
plt.show()