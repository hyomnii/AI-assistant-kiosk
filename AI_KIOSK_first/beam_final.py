import numpy as np
import sounddevice as sd
import whisper

from stt_correction_model_hybrid import correct_text
from search_menu import search_menu, WHISPER_MODEL_SIZE

# --- [beamforming 파라미터] ---
RATE = 16000
MIC_DISTANCE = 0.077
SOUND_SPEED = 343.0
STT_RECORD_SECONDS = 4.0
STT_CHANNELS = 2


# --- [beamforming 함수 유지] ---
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

    full_max_tau = MIC_DISTANCE / SOUND_SPEED
    full_max_shift = int(np.ceil(full_max_tau * RATE))
    full_search_range = cc[center - full_max_shift : center + full_max_shift + 1]
    true_shift = np.argmax(np.abs(full_search_range)) - full_max_shift

    target_max_tau = (MIC_DISTANCE * np.sin(np.deg2rad(max_angle))) / SOUND_SPEED
    target_max_shift = int(np.ceil(target_max_tau * RATE))
    g_theta = 1.0 if abs(true_shift) <= target_max_shift else 0.05

    shift_l = -(true_shift // 2)
    shift_r = true_shift + shift_l
    aligned_left = np.roll(left, shift_l)
    aligned_right = np.roll(right, shift_r)

    y_sum = (aligned_left + aligned_right) / 2.0
    y_diff = (aligned_left - aligned_right) / 2.0

    rms_sum = np.sqrt(np.mean(np.square(y_sum)) + 1e-10)
    rms_diff = np.sqrt(np.mean(np.square(y_diff)) + 1e-10)
    penalty = lambda_val * (rms_diff / rms_sum)
    weight = np.clip(1.0 - penalty, 0.05, 1.0)

    combined = g_theta * weight * y_sum
    return combined.astype(np.float32)


def listen_with_beamforming(stt_model):
    print("\n엔터를 누르면 음성 입력을 시작합니다. 종료하려면 Ctrl+C 또는 '종료'를 말하세요.")
    input()
    print(f"{STT_RECORD_SECONDS:.0f}초 동안 듣는 중...")

    stereo_audio = sd.rec(
        int(STT_RECORD_SECONDS * RATE),
        samplerate=RATE,
        channels=STT_CHANNELS,
        dtype="float32",
    )
    sd.wait()

    if stereo_audio.size == 0:
        return ""

    beam_audio = apply_ultimate_beamforming(stereo_audio)

    result = stt_model.transcribe(beam_audio, language="ko", fp16=False)
    return (result.get("text") or "").strip()


if __name__ == "__main__":
    print("Whisper STT 모델을 로딩합니다...")
    stt_model = whisper.load_model(WHISPER_MODEL_SIZE)

    while True:
        try:
            raw_text = listen_with_beamforming(stt_model)
        except KeyboardInterrupt:
            print("\n종료합니다.")
            break
        except Exception as e:
            print(f"음성 입력 오류: {e}")
            continue

        print(f"인식된 문장: {raw_text}")

        if raw_text.strip() in ["종료", "취소", "exit", "quit"]:
            break

        corrected_text = correct_text(raw_text)
        print(f"교정된 문장: {corrected_text}")

        result = search_menu(corrected_text)

        if not result:
            print("조건에 맞는 메뉴가 없습니다.")
        else:
            for r in result:
                print(r)
