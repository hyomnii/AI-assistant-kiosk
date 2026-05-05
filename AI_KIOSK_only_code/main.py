import os
import re
import subprocess
import time
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import sounddevice as sd
import whisper
from gtts import gTTS

from beam_final import RATE, apply_ultimate_beamforming
from search_menu import WHISPER_MODEL_SIZE, df, extract_temp, search_menu
from stt_correction_model_hybrid import OPENAI_MODEL, _generate_text, correct_text


STT_RECORD_SECONDS = 4.0
STT_CHANNELS = 2
MAX_NO_SPEECH_PROB = 0.65
MIN_AVG_LOGPROB = -1.2
MIN_VALID_CHAR_RATIO = 0.55
MIN_CORRECTION_SIMILARITY = 0.25

COFFEE_OPTION_TEXT = "선택 안함, 연하게 (1샷) + 0원, 샷 추가 +500원, 시럽 추가 +0원"
ICE_OPTION_TEXT = "선택안함, 얼음 적게, 얼음 많이"
SIZE_OPTION_TEXT = "사이즈 : S -300원, M +0원, L +700원"
MILK_OPTION_TEXT = "선택안함, 두유로 변경 +500원, 저당 우유로 변경 +500원"
DESSERT_RECOMMENDATION_TEXT = (
    "이런 메뉴는 어떠세요?\n"
    "선택 안함, 아메리카노 +1500원, ICE 아메리카노 +2000원"
)


def valid_text_ratio(text):
    if not text:
        return 0.0
    valid_chars = re.findall(r"[0-9A-Za-z가-힣\s]", text)
    return len(valid_chars) / len(text)


def has_too_many_repeats(text):
    compact = text.replace(" ", "")
    return bool(re.search(r"(.)\1{4,}", compact))


def is_unreliable_transcription(text, whisper_result):
    text = (text or "").strip()
    if len(text) < 2:
        return True
    if valid_text_ratio(text) < MIN_VALID_CHAR_RATIO:
        return True
    if has_too_many_repeats(text):
        return True

    segments = whisper_result.get("segments") or []
    if not segments:
        return False

    no_speech_probs = [
        float(segment.get("no_speech_prob", 0.0))
        for segment in segments
        if segment.get("no_speech_prob") is not None
    ]
    avg_logprobs = [
        float(segment.get("avg_logprob", 0.0))
        for segment in segments
        if segment.get("avg_logprob") is not None
    ]

    if no_speech_probs and max(no_speech_probs) >= MAX_NO_SPEECH_PROB:
        return True
    if avg_logprobs and np.mean(avg_logprobs) <= MIN_AVG_LOGPROB:
        return True
    return False


def is_unusable_corrected_text(raw_text, corrected_text):
    raw_text = (raw_text or "").strip()
    corrected_text = (corrected_text or "").strip()
    if len(corrected_text) < 2:
        return True
    if valid_text_ratio(corrected_text) < MIN_VALID_CHAR_RATIO:
        return True
    if has_too_many_repeats(corrected_text):
        return True

    raw_compact = re.sub(r"\s+", "", raw_text)
    corrected_compact = re.sub(r"\s+", "", corrected_text)
    similarity = SequenceMatcher(None, raw_compact, corrected_compact).ratio()
    has_menu_token = any(token in corrected_text for token in ["ICE", "HOT", "메뉴", "추천"])
    return similarity < MIN_CORRECTION_SIMILARITY and not has_menu_token


def listen_with_beamforming(stt_model, wait_for_enter=True, allow_short_confirmation=False):
    if wait_for_enter:
        print("\n엔터를 누르면 음성 입력을 시작합니다. 종료하려면 Ctrl+C 또는 '종료'를 말하세요.")
        input()
    else:
        print("\n바로 음성 입력을 시작합니다. 대답해 주세요.")
    print(f"{STT_RECORD_SECONDS:.0f}초 동안 듣는 중...", flush=True)

    stereo_audio = sd.rec(
        int(STT_RECORD_SECONDS * RATE),
        samplerate=RATE,
        channels=STT_CHANNELS,
        dtype="float32",
    )
    sd.wait()
    print("\n녹음 종료, 음성 인식 중...\n", flush=True)

    if stereo_audio.size == 0:
        return ""

    beam_audio = apply_ultimate_beamforming(stereo_audio)
    result = stt_model.transcribe(beam_audio, language="ko", fp16=False)
    text = (result.get("text") or "").strip()
    if allow_short_confirmation and text:
        return text
    if is_unreliable_transcription(text, result):
        return ""
    return text


def generate_kiosk_response(menu_results, user_query):
    if len(menu_results) == 1:
        return f"{menu_results[0]} - 이 메뉴가 맞으시나요?\n맞으시면 네 맞아요, 아니면 아니요 라고 말씀해주세요."
    if len(menu_results) > 1:
        menu_names = ", ".join(menu_results)
        return f"{menu_names} 중 어떤 메뉴로 드릴까요?"
    return "죄송합니다. 해당 메뉴를 찾지 못했습니다. 다른 메뉴로 말씀해 주세요."


def preserve_temp_for_search(raw_text, corrected_text):
    raw_temp = extract_temp(raw_text)
    corrected_temp = extract_temp(corrected_text)
    if raw_temp == "ICE" and corrected_temp is None:
        return f"ICE {corrected_text}".strip()
    return corrected_text


def is_positive_answer(text):
    compact = text.replace(" ", "")
    return any(word in compact for word in ["네", "예", "맞아", "맞아요", "응", "좋아", "좋아요", "확인"])


def is_negative_answer(text):
    compact = text.replace(" ", "")
    return any(word in compact for word in ["아니", "아니요", "아뇨", "틀려", "틀렸", "취소", "다시"])


def is_positive_answer(text):
    compact = text.replace(" ", "")
    return any(word in compact for word in ["네", "내", "넵", "예", "옙", "맞아", "맞아요", "맞습니다", "응", "좋아", "좋아요", "확인"])


def is_negative_answer(text):
    compact = text.replace(" ", "")
    return any(word in compact for word in ["아니", "아니오", "아니요", "아니야", "아니에요", "아뇨", "틀려", "틀렸", "취소", "다시"])


def correct_confirmation_answer(text):
    compact = (text or "").replace(" ", "").strip()
    negative_misheard = {
        "안녕",
        "안녀",
        "아녕",
        "아뇽",
        "안뇽",
        "아니영",
        "아니여",
        "아니오",
    }
    positive_misheard = {
        "내",
        "넵",
        "냅",
        "녜",
        "예",
        "옙",
    }

    if compact in negative_misheard:
        return "아니요"
    if compact in positive_misheard:
        return "네"
    return text


def classify_confirmation_answer_with_llm(text):
    text = (text or "").strip()
    if not text:
        return "unknown"

    corrected = correct_confirmation_answer(text)
    if is_positive_answer(corrected):
        return "yes"
    if is_negative_answer(corrected):
        return "no"

    prompt = f"""
사용자가 키오스크 메뉴 확인 질문에 답했습니다.
답변이 메뉴가 맞다는 의미면 yes, 아니라는 의미면 no, 판단이 어려우면 unknown만 출력하세요.
음성 인식 오류를 고려하세요. 예를 들어 '안녕'은 이 문맥에서 '아니요'일 가능성이 높습니다.

질문: 이 메뉴가 맞으시나요? 맞으시면 네, 아니면 아니요 라고 말씀해주세요.
사용자 답변: {corrected}

출력 형식: yes 또는 no 또는 unknown
"""

    result = (_generate_text(prompt, OPENAI_MODEL) or "").strip().lower()
    if "yes" in result:
        return "yes"
    if "no" in result:
        return "no"
    if is_positive_answer(text):
        return "yes"
    if is_negative_answer(text):
        return "no"
    return "unknown"


def find_selected_menu(text, candidates):
    compact_text = text.replace(" ", "")

    for menu in candidates:
        compact_menu = menu.replace(" ", "")
        base_menu = compact_menu.replace("ICE", "", 1)
        if compact_menu in compact_text or base_menu in compact_text:
            return menu

    searched = search_menu(text)
    for menu in searched:
        if menu in candidates:
            return menu
    return None


def get_menu_category(menu_name):
    matched = df[df["상품명"] == menu_name]
    if matched.empty:
        return ""
    return str(matched.iloc[0]["카테고리"])


def get_menu_allergy(menu_name):
    matched = df[df["상품명"] == menu_name]
    if matched.empty:
        return ""
    allergy = matched.iloc[0]["알레르기"]
    if not isinstance(allergy, str):
        return ""
    return allergy


def is_coffee_menu(menu_name):
    return get_menu_category(menu_name) == "커피"


def is_ice_menu(menu_name):
    return menu_name.startswith("ICE ")


def is_drink_menu(menu_name):
    return get_menu_category(menu_name) in {"커피", "차"}


def has_milk(menu_name):
    return "우유" in get_menu_allergy(menu_name)


def print_order_options(menu_name):
    category = get_menu_category(menu_name)

    if is_drink_menu(menu_name):
        print(SIZE_OPTION_TEXT)
    if category == "커피":
        print(COFFEE_OPTION_TEXT)
    if is_ice_menu(menu_name):
        print(ICE_OPTION_TEXT)
    if is_drink_menu(menu_name) and has_milk(menu_name):
        print(MILK_OPTION_TEXT)
    if category == "디저트":
        print(DESSERT_RECOMMENDATION_TEXT)


def speak_tts(text):
    response =f"AI 점원 :{text}"
    print(response)
    tts = gTTS(text, lang="ko")
    audio_path = os.path.abspath("kiosk_voice.mp3")
    tts.save(audio_path)
    play_mp3_blocking(audio_path)


def play_mp3_blocking(audio_path):
    audio_uri = Path(audio_path).as_uri()
    script = f"""
Add-Type -AssemblyName PresentationCore
$player = New-Object System.Windows.Media.MediaPlayer
$player.Open([Uri]"{audio_uri}")
while (-not $player.NaturalDuration.HasTimeSpan) {{
    Start-Sleep -Milliseconds 100
}}
$player.Play()
Start-Sleep -Milliseconds ([int]$player.NaturalDuration.TimeSpan.TotalMilliseconds + 300)
$player.Close()
"""
    subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        check=False,
    )


def print_order(text):
    print(f"주문 : {text}")


def reset_order_state():
    return "WAITING_ORDER", None, []


def get_correction_context(conversation_state):
    if conversation_state == "WAITING_ORDER":
        return (
            "사용자는 아직 메뉴를 고르기 전입니다. "
            "카페 키오스크에서 메뉴가 무엇이 있는지 묻거나, 추천을 요청하거나, 특정 메뉴를 주문하려는 상황입니다."
        )
    if conversation_state == "CHOOSE_MENU":
        return (
            "여러 후보 메뉴가 제시된 뒤 사용자가 그중 원하는 실제 메뉴명을 다시 말하는 상황입니다."
        )
    if conversation_state == "CONFIRM_MENU":
        return (
            "AI 점원이 메뉴가 맞는지 확인했고, 사용자는 네 또는 아니요에 해당하는 짧은 답변을 하는 상황입니다."
        )
    return ""


def run_kiosk():
    print("Whisper STT 모델을 로딩합니다...")
    stt_model = whisper.load_model(WHISPER_MODEL_SIZE)
    conversation_state, pending_menu, pending_candidates = reset_order_state()
    wait_for_enter = True

    while True:
        try:
            raw_text = listen_with_beamforming(
                stt_model,
                wait_for_enter=wait_for_enter,
                allow_short_confirmation=conversation_state == "CONFIRM_MENU",
            )
        except KeyboardInterrupt:
            print("\n종료합니다.")
            break
        except Exception as e:
            print(f"음성 입력 오류: {e}")
            continue

        if not raw_text.strip():
            wait_for_enter = conversation_state == "WAITING_ORDER"
            continue

        if raw_text.strip() in ["종료", "취소", "exit", "quit"]:
            speak_tts("프로그램을 종료합니다. 감사합니다.")
            break

        if conversation_state == "CONFIRM_MENU":
            corrected_text = correct_confirmation_answer(raw_text)
        else:
            corrected_text = correct_text(raw_text, context=get_correction_context(conversation_state))
        print_order(corrected_text)

        if conversation_state == "CONFIRM_MENU":
            answer_text = corrected_text
            confirmation = classify_confirmation_answer_with_llm(answer_text)
            if confirmation == "yes":
                speak_tts(f"{pending_menu} - 주문 도와드리겠습니다.")
                print_order_options(pending_menu)
                conversation_state, pending_menu, pending_candidates = reset_order_state()
                wait_for_enter = True
            elif confirmation == "no":
                speak_tts("알겠습니다. 다시 주문하실 메뉴를 말씀해 주세요.")
                conversation_state, pending_menu, pending_candidates = reset_order_state()
                wait_for_enter = False
            else:
                speak_tts("맞으시면 네, 아니면 아니요 라고 말씀해주세요.")
                wait_for_enter = False
            continue

        if is_unusable_corrected_text(raw_text, corrected_text):
            continue

        if conversation_state == "CHOOSE_MENU":
            selected_menu = find_selected_menu(corrected_text, pending_candidates)
            if selected_menu:
                pending_menu = selected_menu
                pending_candidates = []
                conversation_state = "CONFIRM_MENU"
                speak_tts(generate_kiosk_response([pending_menu], corrected_text))
                wait_for_enter = False
            else:
                speak_tts("어떤 메뉴인지 다시 한 번 말씀해 주세요.")
                wait_for_enter = False
            continue

        search_text = preserve_temp_for_search(raw_text, corrected_text)
        result = search_menu(search_text)

        if not result:
            speak_tts(generate_kiosk_response([], search_text))
            wait_for_enter = False
        elif len(result) == 1:
            pending_menu = result[0]
            conversation_state = "CONFIRM_MENU"
            speak_tts(generate_kiosk_response(result, search_text))
            wait_for_enter = False
        else:
            pending_candidates = result
            conversation_state = "CHOOSE_MENU"
            speak_tts(generate_kiosk_response(result, search_text))
            wait_for_enter = False


if __name__ == "__main__":
    run_kiosk()
