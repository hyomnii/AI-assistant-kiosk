import asyncio
import csv
import json
import mimetypes
import os
import traceback
import uuid
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


HOST = "127.0.0.1"
PORT = int(os.environ.get("KIOSK_WEB_PORT", "8000"))
SESSION_COOKIE = "kiosk_session"
ROOT_DIR = Path(__file__).resolve().parent
STATIC_DIR = ROOT_DIR / "static"
IMAGE_DIR = ROOT_DIR / "static" / "menu_images"
ICON_DIR = STATIC_DIR / "icons"
PAYMENT_DIR = STATIC_DIR / "payment"
TTS_DIR = STATIC_DIR / "tts"

_kiosk = None
_stt_model = None
_sessions = {}
_menu_cache = None


# main.py를 필요할 때 한 번만 불러와서 무거운 초기화를 늦춘다.
def kiosk():
    global _kiosk
    if _kiosk is None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        patch_chromadb_path()
        import main as main_module

        _kiosk = main_module
    return _kiosk


# 음성 인식 모델도 실제 음성 주문 시점에 한 번만 로드한다.
def stt_model():
    global _stt_model
    if _stt_model is None:
        main = kiosk()
        import whisper

        print(f"Loading Whisper STT model ({main.WHISPER_MODEL_SIZE})...")
        _stt_model = whisper.load_model(main.WHISPER_MODEL_SIZE)
        print("Whisper STT model loaded.")
    return _stt_model


def preload_models():
    print("Preloading kiosk models...")
    kiosk()
    stt_model()
    print("Kiosk models are ready.")


# ChromaDB가 항상 프로젝트 안의 menu_DB를 사용하도록 경로를 고정한다.
def patch_chromadb_path():
    import chromadb

    local_db_path = str(ROOT_DIR / "menu_DB")
    original_client = chromadb.PersistentClient

    if getattr(original_client, "_kiosk_path_patched", False):
        return

    def persistent_client_with_local_path(*args, **kwargs):
        kwargs["path"] = local_db_path
        return original_client(*args, **kwargs)

    persistent_client_with_local_path._kiosk_path_patched = True
    chromadb.PersistentClient = persistent_client_with_local_path


# 입력값을 문자열로 바꾸고 앞뒤 공백을 정리한다.
def normalize_text(text):
    return str(text or "").strip()


# 메뉴 이름과 같은 이미지 파일이 있으면 웹에서 접근할 URL을 만든다.
def image_url_for(menu_name):
    for suffix in (".jpg", ".jpeg", ".png", ".webp"):
        path = IMAGE_DIR / f"{menu_name}{suffix}"
        if path.exists():
            return f"/static/menu_images/{menu_name}{suffix}"
    return ""


# menu.csv를 읽어 직접 주문 화면에서 쓸 메뉴 목록을 캐싱한다.
def load_menu_items():
    global _menu_cache
    if _menu_cache is not None:
        return _menu_cache

    items = []
    with open(ROOT_DIR / "menu.csv", "r", encoding="cp949", newline="") as menu_file:
        reader = csv.DictReader(menu_file)
        for row in reader:
            name = normalize_text(row.get("상품명"))
            category = normalize_text(row.get("카테고리"))
            if not name:
                continue
            items.append(
                {
                    "name": name,
                    "category": category,
                    "calories": normalize_text(row.get("칼로리")),
                    "caffeine": normalize_text(row.get("카페인")),
                    "allergy": normalize_text(row.get("알레르기")) or "없음",
                    "image": image_url_for(name),
                }
            )

    _menu_cache = items
    return _menu_cache


# 한 사용자의 AI 주문 진행 상태를 관리한다.
class Conversation:
    def __init__(self):
        self.reset()

    # 주문 흐름을 처음 상태로 되돌린다.
    def reset(self):
        self.state = "WAITING_ORDER"
        self.pending_menu = None
        self.pending_candidates = []

    # 마이크 입력을 받아 텍스트 주문 처리 흐름으로 넘긴다.
    def handle_voice(self):
        main = kiosk()
        raw_text = main.listen_with_beamforming(
            stt_model(),
            wait_for_enter=False,
            allow_short_confirmation=self.state in {"CONFIRM_MENU", "DEFAULT_OPTION_CONFIRM", "OPTION_SELECT"},
        )
        return self.handle(raw_text, source="voice")

    # 사용자 입력을 보정, 검색, 확인, 옵션 선택 단계에 맞게 처리한다.
    def handle(self, raw_text, source="text"):
        main = kiosk()
        raw_text = normalize_text(raw_text)
        events = [{"role": "user", "text": raw_text or "음성을 인식하지 못했습니다.", "source": source}]

        if not raw_text:
            return self._response(events, "죄송해요. 잘 들리지 않았어요. 다시 한 번 말씀해 주세요.")

        if raw_text.lower() in {"exit", "quit"} or raw_text in {"종료", "취소"}:
            self.reset()
            return self._response(events, "주문을 종료했습니다. 다시 주문하시려면 말씀해 주세요.")

        if self.state == "CONFIRM_MENU":
            corrected_text = main.correct_confirmation_answer(raw_text)
        else:
            corrected_text = main.correct_text(raw_text, context=main.get_correction_context(self.state))
        corrected_text = normalize_text(corrected_text)

        if self.state == "CONFIRM_MENU":
            if self.pending_menu and self.pending_menu in corrected_text:
                events[0]["text"] = self.pending_menu
                return self._response(
                    events,
                    main.get_menu_confirmation_question(self.pending_menu),
                )
            confirmation = main.classify_confirmation_answer_with_llm(corrected_text)
            if confirmation == "yes":
                events[0]["text"] = "네, 맞아요"
                if not main.is_drink_menu(self.pending_menu):
                    completed_menu = self.pending_menu
                    response_text = f"네, {main.display_menu_name(self.pending_menu)} 준비해 드리겠습니다. 주문이 완료되었습니다."
                    self.reset()
                    return self._response(events, response_text, order_items=[completed_menu])
                else:
                    response_text = main.get_default_option_offer(self.pending_menu)
                    self.state = "DEFAULT_OPTION_CONFIRM"
                return self._response(events, response_text)

            if confirmation == "no":
                events[0]["text"] = "아니요"
                self.reset()
                return self._response(events, "아, 죄송합니다. 다시 한 번 말씀해 주시겠어요?")

            return self._response(events, "맞으면 맞아요, 아니면 아니요라고 말씀해 주세요.")

        if self.state == "DEFAULT_OPTION_CONFIRM":
            if main.is_option_change_request(raw_text) or main.is_option_change_request(corrected_text):
                self.state = "OPTION_SELECT"
                response_text = self.order_options(self.pending_menu)
                response_text += "\n\n원하시는 옵션을 말씀해 주세요."
                return self._response(events, response_text)

            if (
                main.is_default_options_answer(raw_text)
                or main.is_default_options_answer(corrected_text)
                or main.is_default_option_acceptance(raw_text)
                or main.is_default_option_acceptance(corrected_text)
            ):
                response_text = main.get_default_option_complete_response()
                self.reset()
                return self._response(events, response_text)

            return self._response(events, "기본 옵션으로 진행할까요? 괜찮으면 그대로 또는 기본으로 달라고 말씀해 주세요.")

        if self.state == "OPTION_SELECT":
            if main.is_default_options_answer(raw_text) or main.is_default_options_answer(corrected_text):
                response_text = main.get_default_option_complete_response()
                self.reset()
                return self._response(events, response_text)

            response_text = "옵션을 다시 말씀해 주세요. 선택이 필요 없으면 모두 기본으로 해달라고 말씀해 주세요."
            options = self.order_options(self.pending_menu)
            if options:
                response_text += "\n\n" + options
            return self._response(events, response_text)

        if main.is_unusable_corrected_text(raw_text, corrected_text):
            return self._response(events, "말씀을 정확히 이해하지 못했어요. 메뉴명을 다시 말씀해 주세요.")

        if self.state == "CHOOSE_MENU":
            selected_menu = main.find_selected_menu(corrected_text, self.pending_candidates)
            if selected_menu:
                events[0]["text"] = selected_menu
                self.pending_menu = selected_menu
                self.pending_candidates = []
                self.state = "CONFIRM_MENU"
                response_text = main.generate_kiosk_response([self.pending_menu], corrected_text)
                return self._response(events, response_text)

            return self._response(events, "어떤 메뉴인지 다시 한 번 말씀해 주세요.")

        ordered_menus = main.extract_multiple_order_menus(raw_text, corrected_text)
        if ordered_menus:
            events[0]["text"] = ", ".join(ordered_menus)
            response_text = main.generate_multi_order_response(ordered_menus)
            self.reset()
            return self._response(events, response_text, order_items=ordered_menus)

        search_text = main.preserve_temp_for_search(raw_text, corrected_text)
        result = main.search_menu(search_text)

        if not result:
            response_text = main.generate_kiosk_response([], search_text)
            return self._response(events, response_text)

        if len(result) == 1:
            events[0]["text"] = result[0]
            self.pending_menu = result[0]
            self.state = "CONFIRM_MENU"
            response_text = main.generate_kiosk_response([self.pending_menu], search_text)
            return self._response(events, response_text)

        self.pending_candidates = result
        self.state = "CHOOSE_MENU"
        response_text = main.generate_kiosk_response(result, search_text)
        return self._response(events, response_text, candidates=result)

    # 선택한 메뉴에 필요한 주문 옵션 안내 문구를 만든다.
    def order_options(self, menu_name):
        main = kiosk()
        category = main.get_menu_category(menu_name)
        parts = []

        if main.is_drink_menu(menu_name):
            parts.append("사이즈는 S, M, L 중 선택하실 수 있어요.")
        if main.is_coffee_menu(menu_name):
            parts.append("샷은 연하게, 기본, 1샷 추가 중 선택하실 수 있어요.")
        if main.is_ice_menu(menu_name):
            parts.append("얼음은 적게, 기본, 많이 중 선택하실 수 있어요.")
        if main.is_drink_menu(menu_name) and main.has_milk(menu_name):
            parts.append("우유는 기본, 두유, 저당 우유 중 선택하실 수 있어요.")

        return "\n".join(parts)

    # 프론트엔드가 바로 사용할 수 있는 응답 형식으로 묶는다.
    def _response(self, events, assistant_text, candidates=None, order_items=None):
        main = kiosk()
        assistant_text = normalize_text(assistant_text)
        events.append({"role": "assistant", "text": assistant_text})
        default_options = ""
        if self.state == "DEFAULT_OPTION_CONFIRM" and self.pending_menu:
            default_options = main.get_default_option_summary(self.pending_menu)
        return {
            "events": events,
            "state": self.state,
            "pending_menu": self.pending_menu,
            "candidates": candidates or self.pending_candidates,
            "default_options": default_options,
            "ai_order_items": order_items or [],
        }


# 브라우저 쿠키에서 세션 ID를 읽고, 없으면 새로 만든다.
def get_session_id(handler):
    jar = cookies.SimpleCookie(handler.headers.get("Cookie", ""))
    if SESSION_COOKIE in jar:
        return jar[SESSION_COOKIE].value, False
    return uuid.uuid4().hex, True


# 세션별 Conversation 객체를 가져오거나 새로 생성한다.
def get_conversation(session_id):
    if session_id not in _sessions:
        _sessions[session_id] = Conversation()
    return _sessions[session_id]


# JSON API 응답을 공통 형식으로 전송한다.
def json_response(handler, status, payload, session_id=None):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    if session_id:
        handler.send_header("Set-Cookie", f"{SESSION_COOKIE}={session_id}; Path=/; SameSite=Lax")
    handler.end_headers()
    handler.wfile.write(body)


# HTML이나 일반 텍스트 응답을 전송한다.
def text_response(handler, status, body, content_type="text/html; charset=utf-8", session_id=None):
    data = body.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
    handler.send_header("Pragma", "no-cache")
    handler.send_header("Expires", "0")
    if session_id:
        handler.send_header("Set-Cookie", f"{SESSION_COOKIE}={session_id}; Path=/; SameSite=Lax")
    handler.end_headers()
    handler.wfile.write(data)


# static 폴더 안의 파일만 안전하게 내려준다.
def file_response(handler, path):
    if not path.exists() or not path.is_file() or STATIC_DIR not in path.parents:
        text_response(handler, 404, "Not found", "text/plain; charset=utf-8")
        return
    data = path.read_bytes()
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


# 안내 문장을 TTS 음성 파일로 저장하고 재생 URL을 반환한다.
def make_tts_audio(text, session_id):
    main = kiosk()
    TTS_DIR.mkdir(parents=True, exist_ok=True)
    audio_name = f"{session_id or uuid.uuid4().hex}.mp3"
    audio_path = TTS_DIR / audio_name
    tts_text = main.normalize_tts_text(text)
    asyncio.run(main.save_tts_audio(tts_text, str(audio_path)))
    return f"/static/tts/{audio_name}?v={uuid.uuid4().hex}"


# HTTP 요청을 받아 화면, API, 정적 파일을 처리하는 웹 서버 핸들러다.
class KioskHandler(BaseHTTPRequestHandler):
    # 화면 HTML, 메뉴 데이터, 정적 파일 요청을 처리한다.
    def do_GET(self):
        parsed = urlparse(self.path)
        session_id, is_new = get_session_id(self)

        if parsed.path == "/":
            text_response(self, 200, INDEX_HTML, session_id=session_id if is_new else None)
            return

        if parsed.path == "/api/menu":
            json_response(self, 200, {"items": load_menu_items()}, session_id=session_id if is_new else None)
            return

        if parsed.path == "/api/state":
            conversation = get_conversation(session_id)
            json_response(
                self,
                200,
                {
                    "state": conversation.state,
                    "pending_menu": conversation.pending_menu,
                    "candidates": conversation.pending_candidates,
                },
                session_id=session_id if is_new else None,
            )
            return

        if parsed.path.startswith("/static/menu_images/"):
            name = unquote(parsed.path.removeprefix("/static/menu_images/"))
            file_response(self, IMAGE_DIR / name)
            return

        if parsed.path.startswith("/static/tts/"):
            name = unquote(parsed.path.removeprefix("/static/tts/"))
            file_response(self, TTS_DIR / name)
            return

        if parsed.path.startswith("/static/icons/"):
            name = unquote(parsed.path.removeprefix("/static/icons/"))
            file_response(self, ICON_DIR / name)
            return

        if parsed.path.startswith("/static/payment/"):
            name = unquote(parsed.path.removeprefix("/static/payment/"))
            file_response(self, PAYMENT_DIR / name)
            return

        if parsed.path == "/static/ai_staff.png":
            file_response(self, STATIC_DIR / "ai_staff.png")
            return

        if parsed.path == "/static/home_bg_v2.png":
            file_response(self, STATIC_DIR / "home_bg_v2.png")
            return

        if parsed.path.startswith("/static/fonts/"):
            name = unquote(parsed.path.removeprefix("/static/fonts/"))
            file_response(self, STATIC_DIR / "fonts" / name)
            return

        text_response(self, 404, "Not found", "text/plain; charset=utf-8")

    # 메시지, 음성 인식, TTS, 초기화 같은 API 요청을 처리한다.
    def do_POST(self):
        parsed = urlparse(self.path)
        session_id, is_new = get_session_id(self)
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length).decode("utf-8") if length else "{}"

        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            json_response(self, 400, {"error": "Invalid JSON"}, session_id=session_id if is_new else None)
            return

        try:
            conversation = get_conversation(session_id)

            if parsed.path == "/api/message":
                result = conversation.handle(payload.get("message", ""))
                json_response(self, 200, result, session_id=session_id if is_new else None)
                return

            if parsed.path == "/api/voice":
                result = conversation.handle_voice()
                json_response(self, 200, result, session_id=session_id if is_new else None)
                return

            if parsed.path == "/api/tts":
                audio_url = make_tts_audio(payload.get("text", ""), session_id)
                json_response(self, 200, {"audio_url": audio_url}, session_id=session_id if is_new else None)
                return

            if parsed.path == "/api/reset":
                conversation.reset()
                json_response(
                    self,
                    200,
                    {
                        "events": [{"role": "assistant", "text": "새 주문을 시작할게요. 원하시는 메뉴를 말씀해 주세요."}],
                        "state": conversation.state,
                        "pending_menu": conversation.pending_menu,
                        "candidates": conversation.pending_candidates,
                    },
                    session_id=session_id if is_new else None,
                )
                return

            json_response(self, 404, {"error": "Not found"}, session_id=session_id if is_new else None)
        except Exception as exc:
            traceback.print_exc()
            json_response(
                self,
                500,
                {"error": "서버 처리 중 오류가 발생했습니다.", "detail": str(exc)},
                session_id=session_id if is_new else None,
            )

    # 기본 서버 로그 형식을 간단하게 출력한다.
    def log_message(self, format, *args):
        print("%s - %s" % (self.address_string(), format % args))


# 브라우저에 표시할 전체 키오스크 화면 HTML/CSS/JS다.
INDEX_HTML = r"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI 키오스크</title>
  <style>
    /* 화면 전체: 키오스크에서 사용할 글꼴을 불러온다. */
    @font-face {
      font-family: "Jua";
      src: url("/static/fonts/Jua-Regular.ttf") format("truetype");
      font-weight: 400;
      font-style: normal;
      font-display: swap;
    }

    /* 화면 전체: 배경색, 글자색, 강조색 같은 공통 색상을 정한다. */
    :root {
      --bg: #d8e3ee;
      --paper: rgba(255, 255, 255, 0.94);
      --paper-solid: #ffffff;
      --ink: #082b63;
      --muted: #52657c;
      --line: #d9e1ed;
      --green: #0b4fa8;
      --green-dark: #073d86;
      --amber: #d0902d;
      --cream: #eef5ff;
      --blue: #0b4fa8;
      --red: #c23a3a;
      --shadow: 0 16px 34px rgba(9, 42, 94, 0.14);
      --soft-shadow: 0 8px 22px rgba(9, 42, 94, 0.10);
      --radius-lg: 18px;
      --radius-md: 14px;
      --radius-sm: 10px;
    }

    * { box-sizing: border-box; }

    /* 화면 전체: 기본 여백과 글꼴을 정한다. */
    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Jua", "Noto Sans KR", "Malgun Gothic", sans-serif;
      background:
        radial-gradient(circle at 18% 10%, rgba(255, 255, 255, .8) 0 7%, transparent 8%),
        linear-gradient(135deg, #c2d9ef 0%, #edf6ff 48%, #d4e0eb 100%);
      color: var(--ink);
    }

    button, textarea { font: inherit; }
    button { cursor: pointer; }

    /* 화면 전환: 홈, 직접 주문, AI 주문 화면 중 active 화면만 보이게 한다. */
    .screen {
      display: none;
      min-height: 100vh;
      padding: 28px;
      position: relative;
      overflow: hidden;
    }

    .screen.active { display: flex; }

    /* 첫 화면 중앙: 앱 이름과 주문 방식 선택 버튼 영역이다. */
    .home {
      align-items: center;
      justify-content: center;
      background:
        linear-gradient(180deg, rgba(236, 247, 255, .12), rgba(236, 247, 255, .28)),
        url("/static/home_bg_v2.png") center / cover no-repeat,
        linear-gradient(180deg, #9ed4fb, #d9ebf7);
    }

    .home::before {
      content: none;
    }

    .home::after {
      content: "";
      position: absolute;
      inset: 0;
      background: linear-gradient(90deg, rgba(255,255,255,.22), rgba(255,255,255,.02) 45%, rgba(255,255,255,.16));
      pointer-events: none;
    }

    .home-inner {
      width: min(1040px, 100%);
      display: grid;
      gap: 34px;
      text-align: center;
      position: relative;
      z-index: 1;
    }

    .brand {
      display: grid;
      gap: 10px;
      position: relative;
    }

    .brand::before {
      content: "CAFE\A KIOSK";
      white-space: pre;
      justify-self: start;
      text-align: left;
      color: #082b63;
      font-size: 18px;
      font-weight: 900;
      line-height: 1.05;
      letter-spacing: 0;
      margin-bottom: 24px;
    }

    .brand h1 {
      margin: 0;
      font-size: clamp(52px, 7vw, 88px);
      letter-spacing: 0;
      color: #082b63;
      text-shadow: 0 2px 0 rgba(255,255,255,.42);
    }

    .brand p {
      margin: 0;
      color: #082b63;
      font-size: clamp(18px, 2.2vw, 25px);
      font-weight: 800;
    }

    .choice-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 24px;
    }

    /* 첫 화면 중앙: 직접 주문 / AI 주문 선택 카드 모양을 정한다. */
    .choice {
      min-height: 280px;
      border: 1px solid rgba(161, 181, 207, .9);
      border-radius: var(--radius-lg);
      background: linear-gradient(180deg, rgba(255,255,255,.96), rgba(248,251,255,.9));
      box-shadow: var(--shadow);
      padding: 34px;
      display: grid;
      align-content: center;
      justify-items: center;
      gap: 18px;
      color: var(--ink);
      backdrop-filter: blur(4px);
      transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease;
    }

    .choice.ai {
      border-color: rgba(161, 181, 207, .9);
    }

    .choice:hover {
      transform: translateY(-3px);
      border-color: #0b4fa8;
      box-shadow: 0 20px 44px rgba(9, 42, 94, .18);
    }

    .choice.ai:hover {
      border-color: #0b4fa8;
    }

    .choice-icon {
      width: clamp(118px, 14vw, 156px);
      height: clamp(118px, 14vw, 156px);
      border-radius: 0;
      display: grid;
      place-items: center;
      background: transparent;
      box-shadow: none;
      overflow: visible;
    }

    .choice.ai .choice-icon { background: transparent; }

    .choice-icon img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
    }

    .choice strong {
      font-size: clamp(25px, 3.2vw, 38px);
      line-height: 1.25;
      letter-spacing: 0;
      color: #082b63;
    }

    /* 주문 화면 상단: 처음으로 버튼, 화면 제목, 새 주문 버튼이 있는 바다. */
    .kiosk {
      flex-direction: column;
      gap: 18px;
      margin: 3px;
      min-height: calc(100vh - 6px);
      background: linear-gradient(145deg, rgba(255,255,255,.96), rgba(247,251,255,.94));
      border: 1px solid #cbd8e8;
      border-radius: var(--radius-lg);
      box-shadow: inset 0 0 0 1px rgba(255,255,255,.75), var(--shadow);
    }

    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 16px;
    }

    .topbar h2 {
      margin: 0;
      font-size: clamp(28px, 4vw, 48px);
      letter-spacing: 0;
      color: #082b63;
    }

    .back {
      min-width: 120px;
      min-height: 52px;
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      background: linear-gradient(180deg, #fff, #f7fbff);
      color: var(--blue);
      font-weight: 700;
      box-shadow: var(--soft-shadow);
    }

    .topbar-spacer {
      width: 120px;
      min-height: 52px;
      pointer-events: none;
    }

    /* 직접 주문 화면: 카테고리 탭, 메뉴 목록, 주문 내역을 세로로 배치한다. */
    .direct-layout {
      flex: 1;
      min-height: 0;
      display: grid;
      grid-template-columns: 1fr;
      grid-template-rows: auto minmax(0, 1fr) auto;
      gap: 18px;
    }

    /* 직접 주문 화면 상단: 메뉴 카테고리 탭 버튼 영역이다. */
    .tabs {
      display: flex;
      flex-direction: row;
      gap: 10px;
      overflow: auto;
    }

    .tab {
      min-height: 62px;
      min-width: 150px;
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      background: linear-gradient(180deg, #fff, #f7fbff);
      color: var(--blue);
      font-weight: 800;
      font-size: 18px;
      box-shadow: 0 4px 12px rgba(9, 42, 94, .06);
    }

    .tab.active {
      border-color: var(--green);
      background: linear-gradient(180deg, #115fc2, #083f96);
      color: #fff;
    }

    /* 직접 주문 화면 중앙: 좌우 화살표와 메뉴 카드 목록 영역이다. */
    .menu-browser {
      min-height: 0;
      display: grid;
      grid-template-columns: 64px minmax(0, 1fr) 64px;
      align-items: stretch;
      gap: 12px;
    }

    .menu-arrow {
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      background: rgba(255, 255, 255, .78);
      color: var(--blue);
      font-size: 42px;
      font-weight: 900;
      box-shadow: var(--soft-shadow);
    }

    .menu-arrow:disabled {
      color: #b8b8b8;
      background: #f3f3f3;
      cursor: not-allowed;
      box-shadow: none;
    }

    /* 직접 주문 화면 중앙: 실제 메뉴 카드들이 들어가는 그리드다. */
    .menu-grid {
      min-height: 0;
      overflow: hidden;
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
      grid-auto-rows: minmax(320px, auto);
      gap: 14px;
      padding-right: 4px;
    }

    .menu-card {
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      background: var(--paper-solid);
      overflow: hidden;
      display: grid;
      grid-template-rows: 210px 1fr;
      text-align: left;
      color: var(--ink);
      box-shadow: var(--soft-shadow);
    }

    .menu-card:hover { border-color: var(--blue); }

    .photo {
      width: 100%;
      height: 210px;
      background: linear-gradient(180deg, #fff, #f9fbff);
      display: grid;
      place-items: center;
      color: var(--muted);
      font-size: 14px;
      overflow: hidden;
    }

    .photo img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      padding: 10px;
      display: block;
    }

    .menu-info {
      padding: 13px;
      display: grid;
      gap: 8px;
      align-content: start;
    }

    .menu-name {
      font-weight: 800;
      font-size: 18px;
      line-height: 1.25;
      word-break: keep-all;
      overflow-wrap: anywhere;
    }

    .menu-meta {
      color: var(--muted);
      font-size: 14px;
      line-height: 1.35;
    }

    /* 직접 주문 화면 하단: 선택한 메뉴가 쌓이는 주문 내역 패널이다. */
    .order-panel {
      border-top: 1px solid var(--line);
      padding-top: 18px;
      display: grid;
      grid-template-rows: auto 1fr auto;
      min-height: 0;
      gap: 14px;
    }

    .order-panel h3 {
      margin: 0;
      font-size: 24px;
      color: var(--blue);
    }

    .cart {
      overflow: auto;
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      align-content: start;
      gap: 10px;
    }

    .cart-item {
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      background: #fff;
      padding: 12px;
      display: grid;
      gap: 8px;
    }

    .cart-item strong { font-size: 17px; }
    .cart-item span { color: var(--muted); line-height: 1.4; }

    .cart-head {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 10px;
    }

    .cart-head strong {
      min-width: 0;
      line-height: 1.25;
      word-break: keep-all;
      overflow-wrap: anywhere;
    }

    .cart-qty {
      display: grid;
      grid-template-columns: 38px 34px 38px 38px;
      align-items: center;
      gap: 4px;
      flex: 0 0 auto;
    }

    .qty-btn,
    .trash-btn {
      width: 38px;
      height: 38px;
      border: 1px solid var(--line);
      border-radius: var(--radius-sm);
      background: #fff;
      color: var(--ink);
      font-size: 22px;
      font-weight: 900;
      line-height: 1;
      display: grid;
      place-items: center;
    }

    .qty-btn:disabled {
      color: #b8b8b8;
      background: #f3f3f3;
      border-color: #e0e0e0;
      cursor: not-allowed;
    }

    .qty-count {
      text-align: center;
      font-size: 18px;
      font-weight: 900;
      color: #000;
    }

    .trash-btn {
      color: var(--red);
      font-size: 18px;
    }

    .trash-btn svg {
      width: 18px;
      height: 18px;
      display: block;
      stroke-width: 2.4;
    }

    .panel-actions {
      min-height: 76px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
      padding: 12px 16px;
      border-radius: var(--radius-md);
      background: #fff;
      border: 1px solid var(--line);
      box-shadow: var(--soft-shadow);
    }

    .cancel-order {
      min-height: 48px;
      border: 0;
      background: transparent;
      color: var(--blue);
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 20px;
      font-weight: 900;
    }

    .cancel-order::before {
      content: "\00d7";
      width: 28px;
      height: 28px;
      border-radius: 50%;
      background: var(--blue);
      color: #fff;
      display: grid;
      place-items: center;
      font-size: 25px;
      line-height: 1;
    }

    #finishDirect {
      min-width: 190px;
      min-height: 56px;
      border: 0;
      border-radius: var(--radius-md);
      background: linear-gradient(180deg, #115fc2, #073f98);
      color: #fff;
      font-size: 22px;
      font-weight: 900;
      box-shadow: 0 10px 24px rgba(9, 73, 166, .26);
    }

    #finishDirect:hover {
      background: var(--green-dark);
    }

    .primary {
      min-height: 56px;
      border: 1px solid var(--green);
      border-radius: var(--radius-md);
      background: linear-gradient(180deg, #115fc2, #073f98);
      color: #fff;
      font-weight: 800;
      font-size: 18px;
    }

    .primary:hover { background: var(--green-dark); }

    .secondary {
      min-height: 52px;
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      background: linear-gradient(180deg, #fff, #f7fbff);
      color: var(--blue);
      font-weight: 800;
      box-shadow: var(--soft-shadow);
    }

    /* 결제 화면: 결제 수단 버튼과 쿠폰 사용 영역을 배치한다. */
    .payment-layout {
      flex: 1;
      min-height: 0;
      width: min(940px, 100%);
      margin: 0 auto;
      display: grid;
      align-content: start;
      gap: 24px;
      padding: 10px 0 28px;
    }

    .payment-methods {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }

    .payment-button,
    .coupon-box {
      min-height: 150px;
      border: 1px solid var(--line);
      border-radius: 22px;
      background: #fff;
      box-shadow: var(--soft-shadow);
      color: var(--ink);
      font-weight: 900;
    }

    .payment-button {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 220px;
      align-items: center;
      gap: 20px;
      padding: 18px 24px;
      text-align: left;
      font-size: clamp(23px, 3vw, 32px);
      line-height: 1.18;
      transition: transform .16s ease, border-color .16s ease, box-shadow .16s ease;
    }

    .payment-label {
      min-height: 100%;
      display: flex;
      align-items: center;
      gap: 14px;
    }

    .card-icon {
      width: 54px;
      height: 38px;
      flex: 0 0 54px;
      border: 3px solid currentColor;
      border-radius: 8px;
      position: relative;
    }

    .card-icon::before {
      content: "";
      position: absolute;
      left: 0;
      right: 0;
      top: 9px;
      height: 6px;
      background: currentColor;
    }

    .card-icon::after {
      content: "";
      position: absolute;
      left: 8px;
      bottom: 7px;
      width: 18px;
      height: 4px;
      border-radius: 999px;
      background: currentColor;
    }

    .payment-button:hover {
      transform: translateY(-2px);
      border-color: var(--blue);
      box-shadow: var(--shadow);
    }

    .payment-button.logo-payment {
      grid-template-columns: 1fr;
      padding: 0;
      overflow: hidden;
      background-position: center;
      background-repeat: no-repeat;
      background-size: cover;
    }

    .payment-button.logo-payment.payco {
      background-image: url("/static/payment/payco.jpg");
    }

    .payment-button.logo-payment.kakao-pay {
      background-image: url("/static/payment/kakao_pay.jpg");
    }

    .payment-button.logo-payment.naver-pay {
      background-image: url("/static/payment/naver_pay.png");
    }

    .payment-logo-slot {
      width: 220px;
      height: 118px;
      border: 0;
      border-radius: 18px;
      background: transparent;
      background-position: center;
      background-repeat: no-repeat;
      background-size: contain;
      overflow: hidden;
    }

    .payment-combo {
      display: grid;
      align-content: center;
      gap: 4px;
    }

    .coupon-section {
      border-top: 2px dashed #aeb8c6;
      padding-top: 24px;
      display: grid;
      gap: 14px;
    }

    .coupon-box {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 22px 26px;
      font-size: clamp(23px, 3vw, 32px);
      text-align: left;
    }

    .coupon-box::after {
      content: "+";
      width: 54px;
      height: 54px;
      flex: 0 0 54px;
      border-radius: 50%;
      background: var(--blue);
      color: #fff;
      display: flex;
      align-items: center;
      justify-content: center;
      font-family: Arial, sans-serif;
      font-size: 36px;
      font-weight: 900;
      line-height: 54px;
    }

    /* AI 주문 화면: 왼쪽 음성 주문 영역과 오른쪽 대화 기록 영역을 나눈다. */
    .ai-layout {
      flex: 1;
      min-height: 0;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      gap: 18px;
    }

    /* AI 주문 화면 왼쪽: 직원 이미지, 안내 문구, 음성 버튼 영역이다. */
    .voice-stage {
      display: grid;
      place-items: center;
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      background: var(--paper-solid);
      padding: 30px;
      box-shadow: var(--soft-shadow);
    }

    .voice-inner {
      display: grid;
      gap: 22px;
      justify-items: center;
      text-align: center;
      width: min(680px, 100%);
    }

    .assistant-face {
      width: min(240px, 54vw);
      height: min(320px, 68vw);
      border-radius: var(--radius-md);
      background: transparent;
      box-shadow: none;
      overflow: hidden;
      display: grid;
      place-items: center;
    }

    .assistant-face img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
    }

    .voice-inner h3 {
      margin: 0;
      font-size: clamp(26px, 4vw, 44px);
      letter-spacing: 0;
    }

    .voice-inner p {
      margin: 0;
      color: var(--muted);
      font-size: 20px;
      line-height: 1.45;
    }

    .mic {
      width: min(360px, 100%);
      min-height: 86px;
      border: 0;
      border-radius: var(--radius-md);
      background: linear-gradient(180deg, #115fc2, #073f98);
      color: #fff;
      font-size: 24px;
      font-weight: 900;
    }

    .mic.listening {
      background: var(--amber);
      animation: pulse 1s infinite alternate;
    }

    @keyframes pulse {
      from { transform: scale(1); }
      to { transform: scale(1.025); }
    }

    /* AI 주문 화면 오른쪽: AI 안내 메시지 기록과 음성 안내 토글 영역이다. */
    .ai-log {
      border-left: 1px solid var(--line);
      padding-left: 18px;
      display: grid;
      grid-template-rows: auto 1fr auto;
      gap: 12px;
      min-height: 0;
    }

    .ai-log h3 {
      margin: 0;
      font-size: 24px;
      color: var(--blue);
    }

    .ai-order-box[hidden] {
      display: none;
    }

    .ai-order-box {
      display: grid;
      gap: 12px;
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      background: #fff;
      padding: 14px;
      box-shadow: var(--soft-shadow);
    }

    .ai-order-list {
      display: grid;
      gap: 10px;
    }

    .ai-order-item {
      display: grid;
      gap: 10px;
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      background: #fff;
      padding: 12px;
    }

    .ai-order-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }

    .ai-order-head strong {
      font-size: 20px;
      color: var(--ink);
    }

    .ai-order-options {
      color: var(--muted);
      line-height: 1.45;
      white-space: pre-wrap;
    }

    .ai-item-actions {
      display: grid;
      grid-template-columns: 1fr;
      gap: 10px;
    }

    .ai-actions {
      display: grid;
      gap: 10px;
    }

    .messages {
      overflow: auto;
      display: grid;
      align-content: start;
      gap: 10px;
    }

    .message {
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      padding: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
      word-break: keep-all;
      overflow-wrap: anywhere;
      background: #fff;
    }

    .message.assistant { background: var(--cream); }
    .message.user { background: #edf8f5; }
    .message.system { color: var(--muted); font-size: 14px; }

    .message-help {
      display: block;
      margin-top: 6px;
      color: var(--muted);
      font-size: 0.82em;
      line-height: 1.35;
    }

    /* 메뉴 선택 후 화면 중앙에 뜨는 옵션 선택 팝업이다. */
    .modal {
      position: fixed;
      inset: 0;
      display: none;
      align-items: center;
      justify-content: center;
      background: rgba(8, 28, 61, .38);
      padding: 20px;
    }

    .modal.active { display: flex; }

    .modal-box {
      width: min(560px, 100%);
      border-radius: var(--radius-lg);
      background: var(--paper-solid);
      box-shadow: var(--shadow);
      padding: 22px;
      display: grid;
      gap: 18px;
    }

    .modal-box h3 {
      margin: 0;
      font-size: 28px;
    }

    .option-group {
      display: grid;
      gap: 8px;
    }

    .option-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    .option {
      min-height: 46px;
      border: 1px solid var(--line);
      border-radius: var(--radius-sm);
      background: #fff;
      padding: 0 14px;
      font-weight: 700;
    }

    .option.active {
      border-color: var(--green);
      background: linear-gradient(180deg, #115fc2, #073f98);
      color: #fff;
    }

    /* 작은 화면: 모바일에서는 주요 영역을 한 줄 세로 배치로 바꾼다. */
    @media (max-width: 900px) {
      .screen { padding: 18px; }
      .choice-grid, .direct-layout, .ai-layout, .payment-methods { grid-template-columns: 1fr; }
      .choice { min-height: 220px; }
      .tabs { flex-direction: row; overflow: auto; }
      .tab { min-width: 130px; }
      .order-panel, .ai-log { border-left: 0; padding-left: 0; border-top: 1px solid var(--line); padding-top: 14px; }
      .panel-actions { gap: 12px; }
      #finishDirect { min-width: 150px; }
    }

    @media (max-width: 560px) {
      .panel-actions {
        align-items: stretch;
        flex-direction: column;
      }

      .cancel-order,
      #finishDirect {
        width: 100%;
        justify-content: center;
      }

      .payment-button,
      .coupon-box {
        min-height: 128px;
        padding: 16px;
      }

      .payment-logo-slot {
        width: 160px;
        height: 96px;
      }
    }
  </style>
</head>
<body>
  <section id="homeScreen" class="screen home active">
    <div class="home-inner">
      <div class="brand">
        <h1>AI 키오스크</h1>
        <p>원하시는 주문 방식을 선택해 주세요.</p>
      </div>
      <div class="choice-grid">
        <button class="choice" id="directStart">
          <span class="choice-icon"><img src="/static/icons/menu_order_icon_cutout.png" alt=""></span>
          <strong>직접 선택해서<br>주문할게요!</strong>
        </button>
        <button class="choice ai" id="aiStart">
          <span class="choice-icon"><img src="/static/icons/ai_order_icon_cutout.png" alt=""></span>
          <strong>AI 점원과 함께<br>주문할게요!</strong>
        </button>
      </div>
    </div>
  </section>

  <section id="directScreen" class="screen kiosk">
    <div class="topbar">
      <button class="back" data-go-home>처음으로</button>
      <h2>직접 선택 주문</h2>
      <span class="topbar-spacer" aria-hidden="true"></span>
    </div>
    <div class="direct-layout">
      <nav id="categoryTabs" class="tabs"></nav>
      <div class="menu-browser">
        <button class="menu-arrow" id="menuPrev" type="button" aria-label="Previous menu page">&#8249;</button>
        <div id="menuGrid" class="menu-grid"></div>
        <button class="menu-arrow" id="menuNext" type="button" aria-label="Next menu page">&#8250;</button>
      </div>
      <aside class="order-panel">
        <h3>주문 내역</h3>
        <div id="cart" class="cart"></div>
        <div class="panel-actions">
          <button class="cancel-order" id="clearCart" type="button">전체취소</button>
          <button class="primary" id="finishDirect" type="button">결제하기</button>
        </div>
      </aside>
    </div>
  </section>

  <section id="aiScreen" class="screen kiosk">
    <div class="topbar">
      <button class="back" data-go-home>처음으로</button>
      <h2>AI 점원 음성 주문</h2>
      <button class="back" id="resetAi">새 주문</button>
    </div>
    <div class="ai-layout">
      <div class="voice-stage">
        <div class="voice-inner">
          <div class="assistant-face" aria-hidden="true">
            <img src="/static/ai_staff.png?v=2" alt="">
          </div>
          <h3 id="voiceTitle">원하시는 메뉴를 말씀해 주세요.</h3>
          <p id="voiceHelp">버튼을 누르시면 주문을 도와드릴게요!</p>
          <button class="mic" id="voiceOrder">음성 안내 시작</button>
        </div>
      </div>
      <aside class="ai-log">
        <h3>AI 점원 안내</h3>
        <div id="messages" class="messages"></div>
        <div class="ai-order-box" id="aiOrderBox" hidden>
          <div id="aiOrderList" class="ai-order-list"></div>
          <div class="ai-actions">
            <button class="primary" id="payAi" type="button">결제하기</button>
          </div>
        </div>
      </aside>
    </div>
  </section>

  <section id="paymentScreen" class="screen kiosk">
    <div class="topbar">
      <button class="back" data-go-home>처음으로</button>
      <h2>결제하기</h2>
      <span class="topbar-spacer" aria-hidden="true"></span>
    </div>
    <div class="payment-layout">
      <div class="payment-methods">
        <button class="payment-button" type="button">
          <span class="payment-label">
            <span class="card-icon" aria-hidden="true"></span>
            <span>카드결제</span>
          </span>
        </button>
        <button class="payment-button logo-payment payco" type="button" aria-label="PAYCO"></button>
        <button class="payment-button logo-payment kakao-pay" type="button" aria-label="카카오페이"></button>
        <button class="payment-button logo-payment naver-pay" type="button" aria-label="네이버페이"></button>
      </div>
      <div class="coupon-section">
        <button class="coupon-box" type="button">쿠폰 사용</button>
      </div>
    </div>
  </section>

  <div id="optionModal" class="modal" role="dialog" aria-modal="true">
    <div class="modal-box">
      <h3 id="modalName"></h3>
      <div id="modalOptions"></div>
      <button class="primary" id="addCart">담기</button>
      <button class="secondary" id="closeModal">닫기</button>
    </div>
  </div>

  <script>
    const screens = {
      home: document.querySelector("#homeScreen"),
      direct: document.querySelector("#directScreen"),
      ai: document.querySelector("#aiScreen"),
      payment: document.querySelector("#paymentScreen")
    };
    const categoryTabs = document.querySelector("#categoryTabs");
    const menuGrid = document.querySelector("#menuGrid");
    const menuPrev = document.querySelector("#menuPrev");
    const menuNext = document.querySelector("#menuNext");
    const cartEl = document.querySelector("#cart");
    const messages = document.querySelector("#messages");
    const modal = document.querySelector("#optionModal");
    const modalName = document.querySelector("#modalName");
    const modalOptions = document.querySelector("#modalOptions");
    const voiceOrder = document.querySelector("#voiceOrder");
    const voiceTitle = document.querySelector("#voiceTitle");
    const voiceHelp = document.querySelector("#voiceHelp");
    const aiOrderBox = document.querySelector("#aiOrderBox");
    const aiOrderList = document.querySelector("#aiOrderList");
    const payAi = document.querySelector("#payAi");

    let menuItems = [];
    let activeCategory = "전체";
    let menuPage = 0;
    let selectedMenu = null;
    let selectedOptions = {};
    let cart = [];
    let voiceEnabled = true;
    let currentAudio = null;
    let currentAiMenu = null;
    let currentAiItems = [];
    let editingAiIndex = null;
    let autoVoiceActive = false;
    let voiceFlowRunning = false;
    let voicePausedForOptions = false;

    // 선택한 화면만 보이고 재생 중인 음성은 멈춘다.
    function showScreen(name) {
      Object.values(screens).forEach((screen) => screen.classList.remove("active"));
      screens[name].classList.add("active");
      currentAudio?.pause();
    }

    // AI 주문 화면의 메시지와 안내 상태를 초기화한다.
    function resetAiUi() {
      currentAudio?.pause();
      messages.innerHTML = "";
      voiceTitle.textContent = "원하시는 메뉴를 말씀해 주세요.";
      voiceHelp.textContent = "음성 안내 시작 버튼을 누르면 주문을 시작할게요.";
      currentAiMenu = null;
      currentAiItems = [];
      editingAiIndex = null;
      autoVoiceActive = false;
      voiceFlowRunning = false;
      voicePausedForOptions = false;
      aiOrderBox.hidden = true;
      setListening(false);
    }

    // 서버에 현재 AI 대화 상태 초기화를 요청한다.
    async function resetConversation() {
      try {
        await post("/api/reset", {});
      } catch (error) {
        console.error("AI reset failed", error);
      }
    }

    // 직접 주문 장바구니와 옵션 선택 상태를 비운다.
    function resetDirectOrder() {
      cart = [];
      selectedMenu = null;
      selectedOptions = {};
      modal.classList.remove("active");
      renderCart();
    }

    // 직접 주문 메뉴 목록을 첫 카테고리와 첫 페이지로 돌린다.
    function resetDirectBrowser() {
      activeCategory = "전체";
      menuPage = 0;
      if (menuItems.length) {
        renderCategories();
        renderMenu();
      }
    }

    // 직접 주문 화면으로 이동하고 메뉴 데이터를 준비한다.
    document.querySelector("#directStart").addEventListener("click", () => {
      resetDirectBrowser();
      showScreen("direct");
      loadMenu();
    });

    // AI 주문 화면으로 이동하고 첫 안내 메시지를 음성으로 재생한다.
    document.querySelector("#aiStart").addEventListener("click", async () => {
      resetAiUi();
      await resetConversation();
      showScreen("ai");
      addMessage("assistant", "안녕하세요. 원하시는 메뉴를 말씀해 주세요.");
      await speak("안녕하세요. 원하시는 메뉴를 말씀해 주세요.");
      voiceTitle.textContent = "원하시는 메뉴를 말씀해 주세요.";
      voiceHelp.textContent = "음성 안내 시작 버튼을 누르면 주문을 시작할게요.";
      voiceOrder.disabled = false;
      voiceOrder.textContent = "음성 안내 시작";
    });

    // 처음 화면으로 돌아갈 때 주문 상태를 함께 정리한다.
    document.querySelectorAll("[data-go-home]").forEach((button) => {
      button.addEventListener("click", () => {
        resetAiUi();
        resetDirectOrder();
        resetConversation();
        showScreen("home");
      });
    });

    // 이전 메뉴 페이지로 이동한다.
    menuPrev.addEventListener("click", () => {
      if (menuPage <= 0) return;
      menuPage -= 1;
      renderMenu();
    });

    // 다음 메뉴 페이지로 이동한다.
    menuNext.addEventListener("click", () => {
      menuPage += 1;
      renderMenu();
    });

    // 화면 크기가 바뀌면 한 페이지에 보일 메뉴 수를 다시 계산한다.
    window.addEventListener("resize", () => {
      renderMenu();
    });

    // 서버에서 메뉴 목록을 가져오고 화면에 그린다.
    async function loadMenu() {
      if (menuItems.length) {
        renderCategories();
        renderMenu();
        return;
      }
      const response = await fetch("/api/menu");
      const data = await response.json();
      menuItems = data.items || [];
      renderCategories();
      renderMenu();
    }

    // 메뉴 카테고리 탭 버튼들을 만든다.
    function renderCategories() {
      const categories = ["전체", ...new Set(menuItems.map((item) => item.category).filter(Boolean))];
      categoryTabs.innerHTML = "";
      categories.forEach((category) => {
        const button = document.createElement("button");
        button.className = `tab ${category === activeCategory ? "active" : ""}`;
        button.textContent = category;
        button.addEventListener("click", () => {
          activeCategory = category;
          menuPage = 0;
          renderCategories();
          renderMenu();
        });
        categoryTabs.appendChild(button);
      });
    }

    // 현재 카테고리와 페이지에 맞는 메뉴 카드를 출력한다.
    function renderMenu() {
      const items = activeCategory === "전체"
        ? menuItems
        : menuItems.filter((item) => item.category === activeCategory);
      const sortedItems = sortMenuItems(items);
      const pageSize = getMenuPageSize();
      const pageCount = Math.max(1, Math.ceil(sortedItems.length / pageSize));
      menuPage = Math.min(menuPage, pageCount - 1);
      const pageItems = sortedItems.slice(menuPage * pageSize, (menuPage + 1) * pageSize);

      menuPrev.disabled = menuPage <= 0;
      menuNext.disabled = menuPage >= pageCount - 1;
      menuGrid.innerHTML = "";
      pageItems.forEach((item) => {
        const card = document.createElement("button");
        card.className = "menu-card";
        card.innerHTML = `
          <div class="photo">${item.image ? `<img src="${encodeURI(item.image)}" alt="${item.name}">` : "사진 준비 중"}</div>
          <div class="menu-info">
            <div class="menu-name">${escapeHtml(item.name)}</div>
            <div class="menu-meta">${escapeHtml(item.category)} · ${escapeHtml(item.calories)} kcal</div>
            <div class="menu-meta">카페인 ${escapeHtml(item.caffeine)} mg</div>
          </div>
        `;
        card.addEventListener("click", () => openOptions(item));
        menuGrid.appendChild(card);
      });
    }

    // 화면 너비에 따라 메뉴 한 페이지의 카드 개수를 정한다.
    function getMenuPageSize() {
      if (window.innerWidth <= 560) return 2;
      if (window.innerWidth <= 900) return 4;
      return 8;
    }

    // 카테고리와 메뉴 종류 기준으로 보기 좋은 순서로 정렬한다.
    function sortMenuItems(items) {
      const categoryOrder = ["커피", "차", "디저트"];
      const coffeeOrder = [
        "아메리카노",
        "카페라떼",
        "바닐라라떼",
        "연유라떼",
        "카페모카",
        "카라멜마끼아또",
        "에스프레소",
        "디카페인 아메리카노",
        "디카페인 카페라떼"
      ];

      return [...items].sort((a, b) => {
        const categoryCompare =
          categoryRank(a.category, categoryOrder) - categoryRank(b.category, categoryOrder);
        if (categoryCompare !== 0) return categoryCompare;

        if (a.category === "커피" && b.category === "커피") {
          const baseA = baseMenuName(a.name);
          const baseB = baseMenuName(b.name);
          const coffeeCompare =
            menuRank(baseA, coffeeOrder) - menuRank(baseB, coffeeOrder);
          if (coffeeCompare !== 0) return coffeeCompare;
          return tempOrder(a.name) - tempOrder(b.name);
        }

        if (a.category === "차" && b.category === "차") {
          const teaCompare = teaMenuRank(a.name) - teaMenuRank(b.name);
          if (teaCompare !== 0) return teaCompare;
          return tempOrder(a.name) - tempOrder(b.name);
        }

        return menuItems.indexOf(a) - menuItems.indexOf(b);
      });
    }

    // 지정된 카테고리 순서에서 몇 번째인지 반환한다.
    function categoryRank(category, order) {
      const index = order.indexOf(category);
      return index === -1 ? order.length : index;
    }

    // 지정된 메뉴 순서에서 몇 번째인지 반환한다.
    function menuRank(name, order) {
      const index = order.indexOf(name);
      return index === -1 ? order.length : index;
    }

    // HOT/ICE 접두어를 뺀 기본 메뉴명을 만든다.
    function baseMenuName(name) {
      return String(name || "").replace(/^ICE\s+/, "").replace(/^HOT\s+/, "");
    }

    // 차 메뉴 정렬에 쓸 기본 이름을 만든다.
    function teaBaseName(name) {
      const base = baseMenuName(name).replace(/^핫/, "");
      return base === "초코" ? "초코" : base;
    }

    // 차 메뉴가 원본 메뉴 목록에서 등장하는 순서를 찾는다.
    function teaMenuRank(name) {
      const base = teaBaseName(name);
      const index = menuItems.findIndex((item) => item.category === "차" && teaBaseName(item.name) === base);
      return index === -1 ? menuItems.length : index;
    }

    // HOT 메뉴가 ICE 메뉴보다 먼저 오도록 온도 순서를 정한다.
    function tempOrder(name) {
      if (String(name).startsWith("HOT ")) return 0;
      if (String(name).startsWith("ICE ")) return 1;
      return 0;
    }

    // 메뉴를 클릭하면 옵션 선택 모달을 연다.
    function openOptions(item) {
      selectedMenu = item;
      selectedOptions = {};
      modalName.textContent = item.name;
      modalOptions.innerHTML = "";

      const groups = optionGroups(item);
      groups.forEach((group) => {
        selectedOptions[group.name] = group.values[0];
        const wrap = document.createElement("div");
        wrap.className = "option-group";
        wrap.innerHTML = `<strong>${group.name}</strong>`;
        const row = document.createElement("div");
        row.className = "option-row";
        group.values.forEach((value, index) => {
          const button = document.createElement("button");
          button.className = `option ${index === 0 ? "active" : ""}`;
          button.textContent = value;
          button.addEventListener("click", () => {
            selectedOptions[group.name] = value;
            row.querySelectorAll(".option").forEach((el) => el.classList.remove("active"));
            button.classList.add("active");
          });
          row.appendChild(button);
        });
        wrap.appendChild(row);
        modalOptions.appendChild(wrap);
      });

      modal.classList.add("active");
    }

    // 메뉴 종류에 따라 필요한 옵션 그룹을 만든다.
    function optionGroups(item) {
      const isDrink = ["커피", "차"].includes(item.category);
      const isCoffee = item.category === "커피";
      const isIce = item.name.startsWith("ICE ");
      const hasMilk = item.allergy.includes("우유");
      const groups = [];
      if (isDrink) groups.push({name: "사이즈", values: ["M", "S - 300원", "L + 700원"]});
      if (isCoffee) groups.push({name: "샷", values: ["기본", "연하게 + 0원", "1샷 추가 + 500원", "시럽 추가 + 0원"]});
      if (isIce) groups.push({name: "얼음", values: ["기본", "적게", "많이"]});
      if (isDrink && hasMilk) groups.push({name: "우유", values: ["기본", "두유 변경 + 500원", "저당 우유 변경 + 500원"]});
      if (item.category === "디저트") groups.push({name: "추천 음료", values: ["선택 안 함", "아메리카노 + 1500원", "ICE 아메리카노 + 2000원"]});
      return groups.length ? groups : [{name: "옵션", values: ["기본"]}];
    }

    // 선택한 메뉴와 옵션을 장바구니에 추가한다.
    document.querySelector("#addCart").addEventListener("click", () => {
      if (screens.ai.classList.contains("active") && selectedMenu) {
        const nextItem = {
          menu: selectedMenu,
          options: {...selectedOptions},
          quantity: editingAiIndex === null ? 1 : currentAiItems[editingAiIndex].quantity
        };
        if (editingAiIndex === null) {
          currentAiItems = [nextItem];
        } else {
          currentAiItems[editingAiIndex] = nextItem;
        }
        currentAiMenu = selectedMenu.name;
        editingAiIndex = null;
        renderAiOrderBox();
        modal.classList.remove("active");
        return;
      }
      cart.push({menu: selectedMenu, options: {...selectedOptions}, quantity: 1});
      modal.classList.remove("active");
      renderCart();
    });

    document.querySelector("#closeModal").addEventListener("click", () => {
      editingAiIndex = null;
      modal.classList.remove("active");
    });
    document.querySelector("#clearCart").addEventListener("click", () => {
      resetDirectOrder();
    });

    // 직접 주문 완료 버튼을 처리한다.
    document.querySelector("#finishDirect").addEventListener("click", () => {
      if (!cart.length) {
        alert("메뉴를 먼저 선택해 주세요.");
        return;
      }
      showScreen("payment");
    });

    // 장바구니 목록과 수량 조절 버튼을 다시 그린다.
    function renderCart() {
      cartEl.innerHTML = "";
      if (!cart.length) {
        cartEl.innerHTML = `<div class="message">선택한 메뉴가 없습니다.</div>`;
        return;
      }
      cart.forEach((item, index) => {
        const optionText = Object.entries(item.options).map(([key, value]) => `${key}: ${value}`).join(" · ");
        const row = document.createElement("div");
        row.className = "cart-item";
        row.innerHTML = `
          <div class="cart-head">
            <strong>${escapeHtml(item.menu.name)}</strong>
            <div class="cart-qty" aria-label="Quantity controls">
              <button class="qty-btn" type="button" data-action="decrease" ${item.quantity <= 1 ? "disabled" : ""} aria-label="Decrease quantity">-</button>
              <span class="qty-count">${item.quantity}</span>
              <button class="qty-btn" type="button" data-action="increase" aria-label="Increase quantity">+</button>
              <button class="trash-btn" type="button" data-action="remove" aria-label="Remove item">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" aria-hidden="true">
                  <path d="M3 6h18"></path>
                  <path d="M8 6V4h8v2"></path>
                  <path d="M19 6l-1 14H6L5 6"></path>
                  <path d="M10 11v5"></path>
                  <path d="M14 11v5"></path>
                </svg>
              </button>
            </div>
          </div>
          <span>${escapeHtml(optionText)}</span>
        `;
        row.querySelector('[data-action="decrease"]').addEventListener("click", () => {
          if (item.quantity <= 1) return;
          item.quantity -= 1;
          renderCart();
        });
        row.querySelector('[data-action="increase"]').addEventListener("click", () => {
          item.quantity += 1;
          renderCart();
        });
        row.querySelector('[data-action="remove"]').addEventListener("click", () => {
          cart.splice(index, 1);
          renderCart();
        });
        cartEl.appendChild(row);
      });
    }

    // 서버 API에 JSON POST 요청을 보내고 결과를 반환한다.
    async function post(path, payload = {}) {
      const response = await fetch(path, {
        method: "POST",
        headers: {"Content-Type": "application/json; charset=utf-8"},
        body: JSON.stringify(payload)
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || data.error || "요청 실패");
      return data;
    }

    // 음성 주문 버튼을 누르면 서버의 음성 인식 API를 호출한다.
    voiceOrder.addEventListener("click", async () => {
      if (voicePausedForOptions) {
        await resetConversation();
        currentAiMenu = null;
        currentAiItems = [];
        voicePausedForOptions = false;
      }
      autoVoiceActive = true;
      voiceOrder.disabled = true;
      voiceOrder.textContent = "자동 진행 중";
      runVoiceOrderFlow();
    });

    async function runVoiceOrderFlow() {
      if (voiceFlowRunning) return;
      voiceFlowRunning = true;

      while (autoVoiceActive) {
        const shouldContinue = await runVoiceTurn();
        if (!shouldContinue) {
          stopVoiceOrderFlow();
          break;
        }
        await delay(350);
      }

      voiceFlowRunning = false;
      if (!autoVoiceActive) {
        setListening(false);
      }
    }

    async function runVoiceTurn() {
      setListening(true);
      try {
        const data = await post("/api/voice", {});
        setListening(false);
        await handleAiEvents(data);
        return shouldAutoContinue(data);
      } catch (error) {
        setListening(false);
        addMessage("assistant", "처리 중 문제가 생겼어요. 다시 시도해 주세요.");
        await speak("처리 중 문제가 생겼어요. 다시 시도해 주세요.");
        return true;
      }
    }

    function shouldAutoContinue(data) {
      if (data.state === "DEFAULT_OPTION_CONFIRM") {
        voicePausedForOptions = true;
        return false;
      }
      const assistantText = (data.events || [])
        .filter((event) => event.role === "assistant")
        .map((event) => event.text || "")
        .join(" ");
      return !/주문이 완료|주문을 종료|완료되었습니다/.test(assistantText);
    }

    function stopVoiceOrderFlow() {
      autoVoiceActive = false;
      voiceOrder.classList.remove("listening");
      if (voicePausedForOptions) {
        voiceOrder.disabled = false;
        voiceOrder.textContent = "음성 안내 시작";
        return;
      }
      voiceOrder.disabled = false;
      voiceOrder.textContent = "음성 안내 시작";
    }

    function delay(ms) {
      return new Promise((resolve) => setTimeout(resolve, ms));
    }

    function queueNextVoiceTurn(shouldContinue) {
      if (!shouldContinue) {
        autoVoiceActive = false;
        stopVoiceOrderFlow();
        return;
      }
      autoVoiceActive = true;
      runVoiceOrderFlow();
    }

    // AI 주문 흐름을 초기화하고 안내 메시지를 다시 표시한다.
    document.querySelector("#resetAi").addEventListener("click", async () => {
      resetAiUi();
      await resetConversation();
      addMessage("assistant", "새 주문을 시작할게요. 원하시는 메뉴를 말씀해 주세요.");
    });

    payAi.addEventListener("click", async () => {
      stopVoiceOrderFlow();
      await resetConversation();
      showScreen("payment");
    });

    async function showAiOrderBox(data) {
      const orderItems = data.ai_order_items || [];
      await loadMenu();

      if (orderItems.length) {
        currentAiItems = orderItems
          .map((menuName) => createAiOrderItem(menuName))
          .filter(Boolean);
        currentAiMenu = null;
        renderAiOrderBox();
        return;
      }

      const menuName = data.pending_menu || currentAiMenu || "";
      currentAiMenu = menuName || currentAiMenu;
      currentAiItems = menuName ? [createAiOrderItem(menuName)].filter(Boolean) : [];
      renderAiOrderBox();
    }

    function createAiOrderItem(menuName) {
      const menu = menuItems.find((item) => item.name === menuName);
      if (!menu) return null;
      return {
        menu,
        options: defaultOptionsForMenu(menu),
        quantity: 1
      };
    }

    function defaultOptionsForMenu(menu) {
      const options = {};
      optionGroups(menu).forEach((group) => {
        options[group.name] = group.values[0];
      });
      return options;
    }

    function renderAiOrderBox() {
      aiOrderList.innerHTML = "";
      if (!currentAiItems.length) {
        aiOrderBox.hidden = true;
        return;
      }

      currentAiItems.forEach((item, index) => {
        const optionText = optionTextForItem(item);
        const row = document.createElement("div");
        row.className = "ai-order-item";
        row.innerHTML = `
          <div class="ai-order-head">
            <strong>${escapeHtml(item.menu.name)}</strong>
            <div class="cart-qty" aria-label="AI order quantity controls">
              <button class="qty-btn" type="button" data-action="decrease" ${item.quantity <= 1 ? "disabled" : ""} aria-label="Decrease quantity">-</button>
              <span class="qty-count">${item.quantity}</span>
              <button class="qty-btn" type="button" data-action="increase" aria-label="Increase quantity">+</button>
            </div>
          </div>
          <div class="ai-order-options">${escapeHtml(optionText)}</div>
          <div class="ai-item-actions">
            <button class="secondary" type="button" data-action="options">옵션 변경</button>
          </div>
        `;
        row.querySelector('[data-action="decrease"]').addEventListener("click", () => {
          if (item.quantity <= 1) return;
          item.quantity -= 1;
          renderAiOrderBox();
        });
        row.querySelector('[data-action="increase"]').addEventListener("click", () => {
          item.quantity += 1;
          renderAiOrderBox();
        });
        row.querySelector('[data-action="options"]').addEventListener("click", () => {
          openAiOptionEditor(index);
        });
        aiOrderList.appendChild(row);
      });

      aiOrderBox.hidden = false;
    }

    function optionTextForItem(item) {
      const text = Object.entries(item.options)
        .map(([key, value]) => `${key}: ${value}`)
        .join(" · ");
      return text || "기본 옵션";
    }

    function openAiOptionEditor(index) {
      stopVoiceOrderFlow();
      setListening(false);
      currentAudio?.pause();

      const orderItem = currentAiItems[index];
      if (!orderItem) {
        addMessage("assistant", "옵션을 변경할 메뉴를 찾지 못했어요.");
        return;
      }

      editingAiIndex = index;
      selectedMenu = orderItem.menu;
      selectedOptions = {...orderItem.options};
      modalName.textContent = orderItem.menu.name;
      modalOptions.innerHTML = "";

      optionGroups(orderItem.menu).forEach((group) => {
        if (!selectedOptions[group.name]) {
          selectedOptions[group.name] = group.values[0];
        }
        const wrap = document.createElement("div");
        wrap.className = "option-group";
        wrap.innerHTML = `<strong>${group.name}</strong>`;
        const row = document.createElement("div");
        row.className = "option-row";
        group.values.forEach((value) => {
          const button = document.createElement("button");
          button.className = `option ${selectedOptions[group.name] === value ? "active" : ""}`;
          button.textContent = value;
          button.addEventListener("click", () => {
            selectedOptions[group.name] = value;
            row.querySelectorAll(".option").forEach((el) => el.classList.remove("active"));
            button.classList.add("active");
          });
          row.appendChild(button);
        });
        wrap.appendChild(row);
        modalOptions.appendChild(wrap);
      });

      modal.classList.add("active");
    }

    // 서버에서 받은 대화 이벤트를 화면과 음성 안내에 반영한다.
    async function handleAiEvents(data) {
      if ((data.ai_order_items || []).length) {
        currentAiMenu = null;
      } else if (data.pending_menu) {
        currentAiMenu = data.pending_menu;
      } else if (data.state === "WAITING_ORDER") {
        if (!currentAiItems.length) {
          currentAiMenu = null;
        }
      }
      const spokenTexts = [];
      (data.events || []).forEach((event) => {
        if (event.role === "system") return;
        addMessage(event.role, event.text, data.state);
        if (event.role === "assistant") spokenTexts.push(ttsTextForEvent(event.text, data.state));
      });
      for (const text of spokenTexts) {
        await speak(text);
      }
      if (data.state === "CONFIRM_MENU") {
        aiOrderBox.hidden = true;
        voiceTitle.textContent = "말씀해 주세요.";
        voiceHelp.textContent = "해당 메뉴가 맞으시면 네 맞아요, 아니시면 아니요 라고 말씀해주세요.";
      } else if (data.state === "DEFAULT_OPTION_CONFIRM") {
        await showAiOrderBox(data);
        voiceTitle.textContent = "새로운 주문을 시작하시겠어요?";
        voiceHelp.textContent = "음성 안내 시작 버튼을 누르면 새로운 주문을 시작할게요.";
      } else if (data.state === "OPTION_SELECT") {
        aiOrderBox.hidden = true;
        voiceTitle.textContent = "옵션을 선택해 주세요.";
        voiceHelp.textContent = "원하시는 옵션을 말하거나 모두 기본으로 진행할 수 있어요.";
      } else if (data.state === "CHOOSE_MENU") {
        aiOrderBox.hidden = true;
        voiceTitle.textContent = "후보 메뉴 중 선택해 주세요.";
        voiceHelp.textContent = (data.candidates || []).join(", ");
      } else if ((data.ai_order_items || []).length) {
        await showAiOrderBox(data);
        voiceTitle.textContent = "주문 목록에 담았어요.";
        voiceHelp.textContent = "결제하기를 누르시면 결제 화면으로 이동합니다.";
      } else {
        aiOrderBox.hidden = true;
        voiceTitle.textContent = "원하시는 메뉴를 말씀해 주세요.";
        voiceHelp.textContent = "음성 안내 시작 버튼을 누르면 주문을 시작할게요.";
      }
    }

    // 음성 인식 중인지에 따라 버튼과 안내 문구를 바꾼다.
    function setListening(isListening) {
      voiceOrder.disabled = isListening || autoVoiceActive;
      voiceOrder.classList.toggle("listening", isListening);
      voiceOrder.textContent = isListening
        ? "듣고 있어요..."
        : (autoVoiceActive ? "자동 진행 중" : "음성 안내 시작");
      if (isListening) {
        voiceTitle.textContent = "말씀해 주세요.";
        voiceHelp.textContent = "말씀이 끝나면 자동으로 인식하고 다음 단계로 이어갈게요.";
      }
    }

    function setSpeaking(isSpeaking) {
      if (!isSpeaking) return;
      voiceOrder.disabled = true;
      voiceOrder.classList.remove("listening");
      voiceOrder.textContent = "말하는 중...";
      voiceTitle.textContent = "말하는 중...";
      voiceHelp.textContent = "AI 점원이 안내하고 있어요.";
    }

    // 대화 로그에 메시지 한 줄을 추가한다.
    function ttsTextForEvent(text, state) {
      if (state !== "CONFIRM_MENU") return text;
      return String(text || "").split("\n")[0];
    }

    function addMessage(role, text, state = "") {
      if (role === "system") return;
      const el = document.createElement("div");
      el.className = `message ${role}`;
      if (role === "assistant" && state === "CONFIRM_MENU") {
        const [mainText] = String(text || "").split("\n");
        el.textContent = mainText;
        const help = document.createElement("span");
        help.className = "message-help";
        help.textContent = "맞으시면 네 맞아요, 아니라면 아니요 라고 말씀해주세요.";
        el.appendChild(help);
      } else {
        el.textContent = text;
      }
      messages.appendChild(el);
      messages.scrollTop = messages.scrollHeight;
    }

    // 서버에서 TTS 파일을 만들고 브라우저에서 재생한다.
    async function speak(text) {
      if (!voiceEnabled) return;
      currentAudio?.pause();
      try {
        setSpeaking(true);
        const data = await post("/api/tts", {text});
        if (!voiceEnabled) return;
        currentAudio = new Audio(data.audio_url);
        await new Promise((resolve) => {
          currentAudio.addEventListener("ended", resolve, {once: true});
          currentAudio.addEventListener("error", resolve, {once: true});
          currentAudio.play().catch(resolve);
        });
      } catch (error) {
        console.error("TTS playback failed", error);
      }
    }

    // 메뉴명처럼 화면에 넣는 값을 HTML 특수문자로 안전하게 바꾼다.
    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, (char) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;"
      }[char]));
    }
    renderCart();
  </script>
</body>
</html>
"""


# 서버 시작 전 필요한 폴더를 만들고 웹 서버를 실행한다.
def run():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    preload_models()
    server = ThreadingHTTPServer((HOST, PORT), KioskHandler)
    print(f"AI Assistant Kiosk web app: http://{HOST}:{PORT}")
    print(f"Menu image folder: {IMAGE_DIR}")
    server.serve_forever()


if __name__ == "__main__":
    run()
