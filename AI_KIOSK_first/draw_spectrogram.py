import numpy as np
import scipy.signal
import sounddevice as sd
import matplotlib.pyplot as plt

# =================================================================
# 1. 환경 및 빔포밍 설정값
# =================================================================
RATE = 16000
MIC_DISTANCE = 0.077
SOUND_SPEED = 343.0
STT_RECORD_SECONDS = 4.0
max_angle = 15
lambda_val = 3.0

def record_and_plot_spectrogram():
    print("\n" + "="*50)
    input("👉 준비가 되면 [Enter] 키를 누르세요. 4초간 녹음이 시작됩니다...")
    print("="*50)
    
    print("🎤 4초간 녹음을 시작합니다.")
    print("👉 팁: 측면(45도나 90도)에서 '아아아~' 하거나 기계 소음을 내보세요!")
    
    stereo_data = sd.rec(int(STT_RECORD_SECONDS * RATE), samplerate=RATE, channels=2, dtype="float32")
    sd.wait()
    print("✅ 녹음 완료! 그림을 생성합니다...")

    # --- 기존 빔포밍 정렬 로직 ---
    left, right = stereo_data[:, 0], stereo_data[:, 1]
    n = len(left) + len(right) - 1
    n_fft = 1 << (n - 1).bit_length()
    x1, x2 = np.fft.rfft(left, n=n_fft), np.fft.rfft(right, n=n_fft)
    s_phat = (x1 * np.conj(x2)) / (np.abs(x1 * np.conj(x2)) + 1e-10)
    cc = np.fft.irfft(s_phat, n=n_fft)
    cc = np.concatenate((cc[-n_fft // 2 :], cc[: n_fft // 2]))
    center = n_fft // 2

    limit_tau = (MIC_DISTANCE * np.sin(np.deg2rad(max_angle))) / SOUND_SPEED
    limit_shift = int(np.ceil(limit_tau * RATE))
    search_range = cc[center - limit_shift : center + limit_shift + 1]
    true_shift = np.argmax(np.abs(search_range)) - limit_shift

    shift_l, shift_r = -(true_shift // 2), true_shift - (true_shift // 2)
    aligned_left, aligned_right = np.roll(left, shift_l), np.roll(right, shift_r)

    y_sum = (aligned_left + aligned_right) / 2.0
    y_diff = (aligned_left - aligned_right) / 2.0

    # =================================================================
    # 2. 핵심: STFT 및 가중치(W) 데이터 추출
    # =================================================================
    f_arr, t_arr, z_sum = scipy.signal.stft(y_sum, fs=RATE, nperseg=512)
    _, _, z_diff = scipy.signal.stft(y_diff, fs=RATE, nperseg=512)

    mag_sum, mag_diff = np.abs(z_sum), np.abs(z_diff)

    # 가중치 수식 W(f, t)
    weight = np.clip(1.0 - lambda_val * (mag_diff / (mag_sum + 1e-10)), 0.05, 1.0)
    
    # 가중치가 적용된 최종 음성 데이터
    z_final = z_sum * weight

    # =================================================================
    # 3. 그림 그리기
    # =================================================================
    plt.rcParams['font.family'] = 'Malgun Gothic' # 맥은 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # (1) 적용 전 스펙트로그램
    S_before = 20 * np.log10(np.abs(z_sum) + 1e-10)
    im1 = axes[0].pcolormesh(t_arr, f_arr, S_before, shading='gouraud', cmap='magma')
    axes[0].set_title('1. 가중치 필터 적용 전')
    axes[0].set_ylabel('주파수 [Hz]')
    fig.colorbar(im1, ax=axes[0], format='%+2.0f dB')

    # (2) 가중치 필터 W(f, t) 히트맵
    # 1.0(보존)은 붉은색, 0.05(삭제)는 파란색으로 표시됩니다.
    im2 = axes[1].pcolormesh(t_arr, f_arr, weight, shading='gouraud', cmap='jet', vmin=0.05, vmax=1.0)
    axes[1].set_title('2. 가중치 필터 W(f, t) 활성화 맵')
    axes[1].set_ylabel('주파수 [Hz]')
    fig.colorbar(im2, ax=axes[1], label='가중치 값 (Weight)')

    # (3) 적용 후 스펙트로그램
    S_after = 20 * np.log10(np.abs(z_final) + 1e-10)
    im3 = axes[2].pcolormesh(t_arr, f_arr, S_after, shading='gouraud', cmap='magma')
    axes[2].set_title('3. 가중치 필터 적용 후')
    axes[2].set_xlabel('시간 [sec]')
    axes[2].set_ylabel('주파수 [Hz]')
    fig.colorbar(im3, ax=axes[2], format='%+2.0f dB')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    record_and_plot_spectrogram()