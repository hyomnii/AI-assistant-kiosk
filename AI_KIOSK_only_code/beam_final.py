import numpy as np
import scipy.signal


RATE = 16000
MIC_DISTANCE = 0.077
SOUND_SPEED = 343.0

def apply_ultimate_beamforming(stereo_data, max_angle=15, lambda_val=3.0):
    left = stereo_data[:, 0]
    right = stereo_data[:, 1]

    n = len(left) + len(right) - 1
    n_fft = 1 << (n - 1).bit_length()
    x1 = np.fft.rfft(left, n=n_fft)
    x2 = np.fft.rfft(right, n=n_fft)
    s_phat = (x1 * np.conj(x2)) / (np.abs(x1 * np.conj(x2)) + 1e-10)
    cc = np.fft.irfft(s_phat, n=n_fft)
    cc = np.concatenate((cc[-n_fft // 2 :], cc[: n_fft // 2]))
    center = n_fft // 2

    limit_tau = (MIC_DISTANCE * np.sin(np.deg2rad(max_angle))) / SOUND_SPEED
    limit_shift = int(np.ceil(limit_tau * RATE))
    search_range = cc[center - limit_shift : center + limit_shift + 1]
    true_shift = np.argmax(np.abs(search_range)) - limit_shift

    shift_l = -(true_shift // 2)
    shift_r = true_shift + shift_l
    aligned_left = np.roll(left, shift_l)
    aligned_right = np.roll(right, shift_r)

    y_sum = (aligned_left + aligned_right) / 2.0
    y_diff = (aligned_left - aligned_right) / 2.0

    _, _, z_sum = scipy.signal.stft(y_sum, fs=RATE, nperseg=512)
    _, _, z_diff = scipy.signal.stft(y_diff, fs=RATE, nperseg=512)

    mag_sum = np.abs(z_sum)
    mag_diff = np.abs(z_diff)

    weight = np.clip(1.0 - lambda_val * (mag_diff / (mag_sum + 1e-10)), 0.05, 1.0)
    z_final = z_sum * weight

    _, combined = scipy.signal.istft(z_final, fs=RATE)
    combined = combined[: len(y_sum)]

    return combined.astype(np.float32)
