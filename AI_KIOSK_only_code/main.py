import os
import re
import subprocess
import time
import asyncio
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import sounddevice as sd
import whisper
import edge_tts

from beam_final import RATE, apply_ultimate_beamforming
from search_menu import WHISPER_MODEL_SIZE, df, extract_temp, search_menu
from stt_correction_model_hybrid import OPENAI_MODEL, _generate_text, correct_text


STT_RECORD_SECONDS = 5.0
STT_CHANNELS = 2
OPTION_WAIT_SECONDS = 10
OPTION_LISTEN_CHUNK_SECONDS = 2
TTS_VOICE = "ko-KR-SunHiNeural"
MAX_NO_SPEECH_PROB = 0.65
MIN_AVG_LOGPROB = -1.2
MIN_VALID_CHAR_RATIO = 0.55
MIN_CORRECTION_SIMILARITY = 0.25

COFFEE_OPTION_TEXT = "선택 안함, 연하게 (1샷) + 0원, 샷 추가 +500원, 시럽 추가 +0원"
ICE_OPTION_TEXT = "선택 안함, 얼음 적게, 얼음 많이"
SIZE_OPTION_TEXT = "사이즈 : S -300원, M +0원, L +700원"
MILK_OPTION_TEXT = "선택 안함, 두유로 변경 +500원, 저당 우유로 변경 +500원"
DESSERT_RECOMMENDATION_TEXT = (
    "이런 메뉴는 어떠세요?\n"
    "선택 안함, 아메리카노 +1500원, ICE 아메리카노 +2000원"
)


# 1. STT 결과에서 한글/영문/숫자 등 유효 문자 비율 반환.
def valid_text_ratio(text):
    if not text:
        return 0.0
    valid_chars = re.findall(r"[0-9A-Za-z가-힣\s]", text)
    return len(valid_chars) / len(text)


# 2. 같은 문자가 과도하게 반복된 비정상 인식 결과 여부 확인.
def has_too_many_repeats(text):
    compact = text.replace(" ", "")
    return bool(re.search(r"(.)\1{4,}", compact))


# 3. Whisper 인식 결과의 주문 처리 가능 여부 판단.
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


# 4. LLM 보정 결과의 사용 가능 여부 판단.
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


# 5. 녹음된 스테레오 음성에 빔포밍 적용 후 Whisper 텍스트 반환.
def transcribe_stereo_audio(stt_model, stereo_audio, allow_short_confirmation=False):
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


# 6. 사용자 음성 녹음 후 빔포밍 STT 결과 반환.
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

    return transcribe_stereo_audio(stt_model, stereo_audio, allow_short_confirmation)


# 7. 옵션 선택 단계에서 카운트다운 표시 및 기본 옵션 발화 대기.
def listen_for_default_options_with_countdown(stt_model, wait_seconds=OPTION_WAIT_SECONDS):
    remaining = wait_seconds
    heard_texts = []

    print("\n옵션 선택 대기중..  ", end="", flush=True)
    while remaining > 0:
        chunk_seconds = min(OPTION_LISTEN_CHUNK_SECONDS, remaining)
        stereo_audio = sd.rec(
            int(chunk_seconds * RATE),
            samplerate=RATE,
            channels=STT_CHANNELS,
            dtype="float32",
        )
        for _ in range(chunk_seconds):
            print(f"{remaining} ", end="", flush=True)
            time.sleep(1)
            remaining -= 1
        sd.wait()

        text = transcribe_stereo_audio(
            stt_model,
            stereo_audio,
            allow_short_confirmation=True,
        )
        if text:
            heard_texts.append(text)
            if is_default_options_answer(text):
                print("\n")
                return text

    print("\n")
    return " ".join(heard_texts).strip()


# 8. 검색된 메뉴 개수에 따른 키오스크 안내 문장 반환.
def display_menu_name(menu_name):
    name = str(menu_name or "").strip()
    if name.startswith("HOT "):
        return name
    if name.startswith("ICE "):
        return name
    return name


def candidate_display_names(menu_results):
    names = []
    seen = set()
    for menu in menu_results:
        name = str(menu or "").strip()
        display = name.replace("HOT ", "", 1).replace("ICE ", "", 1)
        key = display.replace(" ", "")
        if key and key not in seen:
            seen.add(key)
            names.append(display)
    return names


def get_menu_confirmation_question(menu_name):
    return f"{display_menu_name(menu_name)} 맞으실까요?"


def generate_kiosk_response(menu_results, user_query):
    if len(menu_results) == 1:
        return get_menu_confirmation_question(menu_results[0])
    if len(menu_results) > 1:
        menu_names = ", ".join(candidate_display_names(menu_results))
        return f"{menu_names} 중에 어떤 걸로 준비해 드릴까요?"
    return "죄송합니다. 해당 메뉴를 찾지 못했습니다. 다른 메뉴로 말씀해 주세요."


# 9. 화면 출력 문장 중 TTS로 읽을 부분만 반환.
def get_tts_response_text(response_text):
    # 긴 메뉴 목록은 TTS로 읽지 않고 선택 질문만 음성 출력한다.
    if "위 메뉴 중 어떤걸로 드릴까요?" in response_text:
        return "위 메뉴 중 어떤걸로 드릴까요?"
    return response_text


# 10. 보정 과정에서 사라진 ICE 온도 표현 보존.
def preserve_temp_for_search(raw_text, corrected_text):
    raw_temp = extract_temp(raw_text)
    corrected_temp = extract_temp(corrected_text)
    if raw_temp == "ICE" and corrected_temp is None:
        return f"ICE {corrected_text}".strip()
    return corrected_text


# 11. 확인 질문에 대한 긍정 답변 여부 1차 판단.
def is_positive_answer(text):
    compact = text.replace(" ", "")
    return any(word in compact for word in ["네", "예", "맞아", "맞아요", "응", "좋아", "좋아요", "확인"])


# 12. 확인 질문에 대한 부정 답변 여부 1차 판단.
def is_negative_answer(text):
    compact = text.replace(" ", "")
    return any(word in compact for word in ["아니", "아니요", "아뇨", "틀려", "틀렸", "취소", "다시"])


# 13. 확장 표현을 포함한 긍정 답변 여부 판단.
def is_positive_answer(text):
    compact = text.replace(" ", "")
    return any(word in compact for word in ["네", "내", "넵", "예", "옙", "맞아", "맞아요", "맞습니다", "응", "좋아", "좋아요", "확인"])


# 14. 확장 표현을 포함한 부정 답변 여부 판단.
def is_negative_answer(text):
    compact = text.replace(" ", "")
    return any(word in compact for word in ["아니", "아니오", "아니요", "아니야", "아니에요", "아뇨", "틀려", "틀렸", "취소", "다시"])


# 15. 짧은 확인 답변의 STT 오인식 보정.
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


# 16. 메뉴 확인 답변을 yes/no/unknown 중 하나로 분류.
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


# 17. 여러 후보 메뉴 중 온도 표현을 고려한 선택 메뉴 반환.
def find_selected_menu(text, candidates):
    compact_text = text.replace(" ", "")
    normalized_compact_text = compact_text.replace("아이스", "ICE")
    temp = extract_temp(text)

    # ICE/HOT 표현이 있으면 같은 온도 후보를 먼저 검사한다.
    if temp == "ICE":
        preferred_candidates = [menu for menu in candidates if is_ice_menu(menu)]
    elif temp == "HOT":
        preferred_candidates = [menu for menu in candidates if not is_ice_menu(menu)]
    else:
        preferred_candidates = []

    ordered_candidates = preferred_candidates + [
        menu for menu in candidates
        if menu not in preferred_candidates
    ]

    for menu in ordered_candidates:
        compact_menu = menu.replace(" ", "")
        if compact_menu in normalized_compact_text:
            return menu

    searched = search_menu(text)
    for menu in searched:
        if menu in candidates:
            return menu

    base_candidates = preferred_candidates if preferred_candidates else ordered_candidates
    for menu in base_candidates:
        compact_menu = menu.replace(" ", "")
        base_menu = compact_menu.replace("ICE", "", 1)
        if base_menu in normalized_compact_text:
            return menu
    return None


# 18. 메뉴명 기준 CSV 카테고리 조회 결과 반환.
def get_menu_category(menu_name):
    matched = df[df["상품명"] == menu_name]
    if matched.empty:
        return ""
    return str(matched.iloc[0]["카테고리"])


# 19. 메뉴명 기준 CSV 알레르기 정보 조회 결과 반환.
def get_menu_allergy(menu_name):
    matched = df[df["상품명"] == menu_name]
    if matched.empty:
        return ""
    allergy = matched.iloc[0]["알레르기"]
    if not isinstance(allergy, str):
        return ""
    return allergy


# 20. 메뉴의 커피 카테고리 포함 여부 확인.
def is_coffee_menu(menu_name):
    return get_menu_category(menu_name) == "커피"


# 21. 메뉴명의 ICE 메뉴 여부 확인.
def is_ice_menu(menu_name):
    return menu_name.startswith("ICE ")


# 22. 메뉴의 음료 카테고리 여부 확인.
def is_drink_menu(menu_name):
    return get_menu_category(menu_name) in {"커피", "차"}


# 23. 메뉴 알레르기 정보의 우유 포함 여부 확인.
def has_milk(menu_name):
    return "우유" in get_menu_allergy(menu_name)


# 24. 메뉴 종류에 맞는 사이즈/샷/얼음/우유/추천 옵션 출력.
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


# 25. 모든 옵션 기본값 선택 발화 여부 판단.
def is_default_options_answer(text):
    compact = (text or "").replace(" ", "")
    return any(
        word in compact
        for word in ["모두기본", "전부기본", "다기본", "기본으로", "기본"]
    )


# 26. 메뉴 속성에 맞는 기본 옵션 선택 결과 문장 반환.
def get_default_option_summary(menu_name):
    options = []

    if is_drink_menu(menu_name):
        options.append("사이즈 M")
    if is_coffee_menu(menu_name):
        options.append("기본 2샷")
    if is_ice_menu(menu_name):
        options.append("얼음 기본")
    if is_drink_menu(menu_name) and has_milk(menu_name):
        options.append("우유 기본")
    if not options:
        options.append("기본 옵션")

    return ", ".join(options)


def get_default_option_offer(menu_name):
    return (
        f"네, {display_menu_name(menu_name)}는 {get_default_option_summary(menu_name)}으로 제공됩니다.\n"
        "옵션 변경을 원하시면 아래 옵션 변경 버튼을 클릭해주세요."
    )


def get_default_option_complete_response():
    return "네, 기본 옵션으로 주문 도와드리겠습니다. 주문이 완료되었습니다."


def is_default_option_acceptance(text):
    compact = (text or "").replace(" ", "")
    return any(
        word in compact
        for word in ["아니괜찮", "괜찮아", "괜찮습니다", "그냥줘", "그냥주세요", "기본으로", "기본", "그대로", "변경안", "필요없"]
    )


def is_option_change_request(text):
    compact = (text or "").replace(" ", "")
    return any(word in compact for word in ["옵션변경", "변경", "바꿀", "바꿔", "수정"])


# 27. TTS의 ICE/HOT 표기 자연 발음용 텍스트 보정.
def normalize_tts_text(text):
    # 화면 표기는 유지하고 TTS 발음만 자연스럽게 보정한다.
    text = str(text)
    text = text.replace("ICE", "아이스")
    text = text.replace("HOT", "핫")
    return text


async def save_tts_audio(text, audio_path):
    tts = edge_tts.Communicate(
        text,
        voice=TTS_VOICE,
        rate="-6%",
        pitch="+0Hz",
        volume="+0%",
    )
    await tts.save(audio_path)


# 28. AI 점원 문장 화면 출력 및 TTS 음성 재생.
def speak_tts(text, tts_text=None):
    response =f"AI 점원 : {text}"
    print(response)
    tts_source = text if tts_text is None else tts_text
    audio_path = os.path.abspath("kiosk_voice.mp3")
    asyncio.run(save_tts_audio(normalize_tts_text(tts_source), audio_path))
    play_mp3_blocking(audio_path)


# 29. 생성된 MP3 파일의 블로킹 재생.
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
        [
            r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            script,
        ],
        check=False,
    )


# 30. 사용자 최종 인식/보정 주문 문장 콘솔 출력.
def print_order(text):
    print(f"주문 : {text}")


# 31. 대화 상태와 보류 메뉴/후보 목록 초기화 결과 반환.
def reset_order_state():
    return "WAITING_ORDER", None, []


# 32. 현재 대화 상태에 맞는 STT 보정용 문맥 설명 반환.
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
    if conversation_state == "OPTION_SELECT":
        return (
            "AI 점원이 메뉴 옵션을 안내했고, 사용자는 사이즈, 샷, 얼음, 우유 옵션을 선택하거나 "
            "모두 기본으로 선택하려는 상황입니다."
        )
    if conversation_state == "DEFAULT_OPTION_CONFIRM":
        return (
            "AI 점원이 음료의 기본 옵션을 안내했고, 사용자는 그대로 진행하거나 옵션 변경을 요청하는 상황입니다."
        )
    return ""


# 33. 키오스크 주문 대화 루프 실행 및 상태 전환 관리.
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
                allow_short_confirmation=conversation_state in {"CONFIRM_MENU", "DEFAULT_OPTION_CONFIRM", "OPTION_SELECT"},
            )
        except KeyboardInterrupt:
            print("\n종료합니다.")
            break
        except Exception as e:
            print(f"음성 입력 오류: {e}")
            continue

        if not raw_text.strip():
            wait_for_enter = False
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
            if pending_menu and pending_menu in corrected_text:
                print_order(pending_menu)
                response_text = generate_kiosk_response([pending_menu], corrected_text)
                speak_tts(response_text, get_tts_response_text(response_text))
                wait_for_enter = False
                continue
            confirmation = classify_confirmation_answer_with_llm(answer_text)
            if confirmation == "yes":
                if not is_drink_menu(pending_menu):
                    speak_tts(f"네, {display_menu_name(pending_menu)} 준비해 드리겠습니다. 주문이 완료되었습니다.")
                    conversation_state, pending_menu, pending_candidates = reset_order_state()
                    wait_for_enter = False
                else:
                    speak_tts(get_default_option_offer(pending_menu))
                    conversation_state = "DEFAULT_OPTION_CONFIRM"
                    wait_for_enter = False
            elif confirmation == "no":
                speak_tts("아, 죄송합니다. 다시 한 번 말씀해 주시겠어요?")
                conversation_state, pending_menu, pending_candidates = reset_order_state()
                wait_for_enter = False
            else:
                speak_tts("맞으시면 네 맞아요, 아니면 아니요 라고 말씀해주세요.")
                wait_for_enter = False
            continue

        if conversation_state == "DEFAULT_OPTION_CONFIRM":
            if is_option_change_request(raw_text) or is_option_change_request(corrected_text):
                print_order_options(pending_menu)
                speak_tts("원하시는 옵션을 말씀해 주세요.")
                conversation_state = "OPTION_SELECT"
                wait_for_enter = False
            elif (
                is_default_options_answer(raw_text)
                or is_default_options_answer(corrected_text)
                or is_default_option_acceptance(raw_text)
                or is_default_option_acceptance(corrected_text)
            ):
                speak_tts(get_default_option_complete_response())
                conversation_state, pending_menu, pending_candidates = reset_order_state()
                wait_for_enter = False
            else:
                speak_tts("기본 옵션으로 진행할까요? 괜찮으면 그대로 또는 기본으로 달라고 말씀해 주세요.")
                wait_for_enter = False
            continue

        if conversation_state == "OPTION_SELECT":
            if is_default_options_answer(raw_text) or is_default_options_answer(corrected_text):
                speak_tts(get_default_option_complete_response())
                conversation_state, pending_menu, pending_candidates = reset_order_state()
                wait_for_enter = False
            else:
                speak_tts(
                    "옵션을 다시 말씀해 주세요. "
                    "옵션 선택을 원하시지 않으시면 모두 기본으로라고 말씀해주세요."
                )
                print_order_options(pending_menu)
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
                response_text = generate_kiosk_response([pending_menu], corrected_text)
                speak_tts(response_text, get_tts_response_text(response_text))
                wait_for_enter = False
            else:
                speak_tts("어떤 메뉴인지 다시 한 번 말씀해 주세요.")
                wait_for_enter = False
            continue

        search_text = preserve_temp_for_search(raw_text, corrected_text)
        result = search_menu(search_text)

        if not result:
            response_text = generate_kiosk_response([], search_text)
            speak_tts(response_text, get_tts_response_text(response_text))
            wait_for_enter = False
        elif len(result) == 1:
            pending_menu = result[0]
            conversation_state = "CONFIRM_MENU"
            response_text = generate_kiosk_response([pending_menu], search_text)
            speak_tts(response_text, get_tts_response_text(response_text))
            wait_for_enter = False
        else:
            pending_candidates = result
            conversation_state = "CHOOSE_MENU"
            response_text = generate_kiosk_response(result, search_text)
            speak_tts(response_text, get_tts_response_text(response_text))
            wait_for_enter = False


if __name__ == "__main__":
    run_kiosk()
