import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import time

# ===== 설정 =====
RATE = 16000
CHANNELS = 2
RECORD_SEC = 3.0

angles = [15, 30, 45, 60, 90]
power_results = []

# ===== RMS → dB =====
def get_rms_db(audio_data):
    rms = np.sqrt(np.mean(np.square(audio_data)))
    return 20 * np.log10(rms + 1e-12)

# ===== 기존 beamforming 그대로 사용 =====
def estimate_delay_gcc_phat(sig1, sig2, fs):
    n = len(sig1) + len(sig2) - 1
    N_fft = 1 << (n-1).bit_length()

    X1 = np.fft.rfft(sig1, n=N_fft)
    X2 = np.fft.rfft(sig2, n=N_fft)
    S = X1 * np.conj(X2)

    S_phat = S / (np.abs(S) + 1e-10)

    cc = np.fft.irfft(S_phat, n=N_fft)
    cc = np.concatenate((cc[-N_fft//2:], cc[:N_fft//2]))
    shift = np.argmax(np.abs(cc)) - N_fft//2

    tau = shift / fs
    return tau

def apply_beamforming(stereo_data):
    left = stereo_data[:, 0]
    right = stereo_data[:, 1]

    tau = estimate_delay_gcc_phat(left, right, RATE)
    delay_samples = int(np.round(tau * RATE))

    half_delay = delay_samples // 2

    aligned_left = np.roll(left, -half_delay)
    aligned_right = np.roll(right, half_delay)

    combined = (aligned_left + aligned_right) / 2.0
    return combined.astype(np.float32)

# ===== 측정 루프 =====
print("🎯 각도별 측정 시작")

for angle in angles:
    print(f"\n👉 {angle}도 방향에서 소리 발생 후 Enter 누르세요")
    input()

    print("🔴 녹음 중...")
    recording = sd.rec(int(RECORD_SEC * RATE), samplerate=RATE, channels=CHANNELS)
    sd.wait()

    print("⏳ 처리 중...")

    beamformed = apply_beamforming(recording)
    power_db = get_rms_db(beamformed)

    print(f"📊 {angle}도 Power: {power_db:.2f} dB")

    power_results.append(power_db)

    time.sleep(1)

# ===== 그래프 =====
plt.figure()
plt.plot(angles, power_results, marker='o')
plt.xlabel("Angle (degree)")
plt.ylabel("Power (dB)")
plt.title("Beamforming Directional Response")
plt.grid()

plt.show()