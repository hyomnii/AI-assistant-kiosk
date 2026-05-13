import warnings

import numpy as np
import sounddevice as sd
import whisper

from beam_final import RATE, apply_ultimate_beamforming
from search_menu import WHISPER_MODEL_SIZE, extract_temp, search_menu
from stt_correction_model_hybrid import correct_text


warnings.filterwarnings("ignore", category=UserWarning)

RECORD_SECONDS = 5.0
CHANNELS = 2
WHISPER_MODEL = WHISPER_MODEL_SIZE
TRIAL_COUNT = 10
BEAM_MAX_ANGLE = 15
BEAM_LAMBDA = 2.0


def get_rms_db(audio_data):
    audio = np.asarray(audio_data, dtype=np.float32)
    rms = np.sqrt(np.mean(np.square(audio)))
    return 20 * np.log10(rms + 1e-12)


def measure_db_phase(phase_name, duration=3.0):
    print(f"\n[{phase_name} 데시벨 측정]")
    input("준비가 되면 엔터를 눌러주세요...")
    print(f"{duration:.0f}초 동안 측정 중...")

    audio = sd.rec(
        int(duration * RATE),
        samplerate=RATE,
        channels=CHANNELS,
        dtype="float32",
    )
    sd.wait()

    db = get_rms_db(audio)
    print(f"측정 완료: 평균 {db:.2f} dBFS")
    return db


def preserve_temp_for_search(raw_text, corrected_text):
    raw_temp = extract_temp(raw_text)
    corrected_temp = extract_temp(corrected_text)
    if raw_temp == "ICE" and corrected_temp is None:
        return f"ICE {corrected_text}".strip()
    return corrected_text


def normalize_menu_name(menu_name):
    return "".join((menu_name or "").split())


def get_target_menu():
    while True:
        target_query = input("검색하고 싶은 메뉴를 키보드로 입력해주세요: ").strip()
        if target_query:
            break
        print("메뉴명을 한 글자 이상 입력해주세요.")

    try:
        target_results = search_menu(target_query)
    except Exception as exc:
        print(f"목표 메뉴 검색 중 오류가 발생했습니다: {type(exc).__name__}: {exc}")
        target_results = []

    target_menu = target_results[0] if target_results else target_query
    print(f"정확도 비교 기준 메뉴: {target_menu}")
    return target_menu


def transcribe_beamformed_audio(stt_model, mixed_audio):
    beamformed_audio = apply_ultimate_beamforming(
        mixed_audio,
        max_angle=BEAM_MAX_ANGLE,
        lambda_val=BEAM_LAMBDA,
    )
    result = stt_model.transcribe(
        beamformed_audio.astype(np.float32),
        language="ko",
        fp16=False,
    )
    return (result.get("text") or "").strip()


def search_from_transcribed_text(raw_text):
    corrected_text = correct_text(raw_text)
    search_text = preserve_temp_for_search(raw_text, corrected_text)
    menu_results = search_menu(search_text)
    return {
        "raw_text": raw_text,
        "corrected_text": corrected_text,
        "menu_results": menu_results,
        "selected_menu": menu_results[0] if menu_results else "",
    }


def is_correct_menu(target_menu, selected_menu):
    return normalize_menu_name(target_menu) == normalize_menu_name(selected_menu)


def run_experiment():
    print("=" * 60)
    print("동시다발 소음환경 빔포밍 성능 검증 테스트")
    print("=" * 60)
    print("2채널 녹음에 beam_final 빔포밍을 적용한 결과만 테스트합니다.")
    print("10회 테스트 후 목표 메뉴와 검색된 메뉴의 정확도를 비교합니다.\n")

    target_menu = get_target_menu()

    print(f"\nWhisper 모델 로딩 중... ({WHISPER_MODEL})")
    stt_model = whisper.load_model(WHISPER_MODEL)
    print("모델 로딩 완료\n")

    print("--- 1단계: 환경 데시벨 측정 ---")
    print("안내: 측면/주변에서 유튜브, 대화, 음악 등 방해 소음만 재생한 상태로 측정합니다.")
    noise_db = measure_db_phase("방해 소음")

    print("\n안내: 방해 소음을 끄고, 정면에서 주문 음성만 말한 상태로 측정합니다.")
    speech_db = measure_db_phase("정면 음성")

    print("\n--- 2단계: 동시 소음 + 정면 주문 10회 연속 테스트 ---")
    print("안내: 방해 소음을 계속 재생한 상태에서, 각 회차마다 정면에서 주문 문장을 말해주세요.")

    results = []

    for i in range(1, TRIAL_COUNT + 1):
        input(f"\n[{i} / {TRIAL_COUNT} 회차] 준비가 되면 엔터를 누르고 주문을 말해주세요...")
        print(f"{RECORD_SECONDS:.0f}초 동안 녹음 중...")

        mixed_audio = sd.rec(
            int(RECORD_SECONDS * RATE),
            samplerate=RATE,
            channels=CHANNELS,
            dtype="float32",
        )
        sd.wait()

        print("녹음 완료. 빔포밍 결과를 분석합니다...")
        raw_text = transcribe_beamformed_audio(stt_model, mixed_audio)
        trial_result = search_from_transcribed_text(raw_text)
        trial_result["is_correct"] = is_correct_menu(
            target_menu,
            trial_result["selected_menu"],
        )
        results.append(trial_result)

        selected = trial_result["selected_menu"] or "검색 결과 없음"
        verdict = "정답" if trial_result["is_correct"] else "오답"
        print(f"{i}회차 검색된 메뉴: {selected} ({verdict})")

    correct_count = sum(1 for item in results if item["is_correct"])

    print("\n\n" + "=" * 60)
    print("최종 결과")
    print("=" * 60)
    print(f"방해 소음 {noise_db:.1f} dB, 정면 음성 {speech_db:.1f} dB")
    print(f"목표 메뉴 : {target_menu}")

    for idx, trial_result in enumerate(results, 1):
        selected = trial_result["selected_menu"] or "검색 결과 없음"
        print(f"{idx}회차 - 검색된 메뉴 : {selected}")

    print(f"정확도 {correct_count}/{TRIAL_COUNT}")


if __name__ == "__main__":
    run_experiment()
