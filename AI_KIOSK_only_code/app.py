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
TTS_DIR = STATIC_DIR / "tts"

_kiosk = None
_stt_model = None
_sessions = {}
_menu_cache = None


def kiosk():
    global _kiosk
    if _kiosk is None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        patch_chromadb_path()
        import main as main_module

        _kiosk = main_module
    return _kiosk


def stt_model():
    global _stt_model
    if _stt_model is None:
        main = kiosk()
        import whisper

        _stt_model = whisper.load_model(main.WHISPER_MODEL_SIZE)
    return _stt_model


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


def normalize_text(text):
    return str(text or "").strip()


def image_url_for(menu_name):
    for suffix in (".jpg", ".jpeg", ".png", ".webp"):
        path = IMAGE_DIR / f"{menu_name}{suffix}"
        if path.exists():
            return f"/static/menu_images/{menu_name}{suffix}"
    return ""


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


class Conversation:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = "WAITING_ORDER"
        self.pending_menu = None
        self.pending_candidates = []

    def handle_voice(self):
        main = kiosk()
        raw_text = main.listen_with_beamforming(
            stt_model(),
            wait_for_enter=False,
            allow_short_confirmation=self.state in {"CONFIRM_MENU", "OPTION_SELECT"},
        )
        return self.handle(raw_text, source="voice")

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

        if corrected_text and corrected_text != raw_text:
            events.append({"role": "system", "text": f"인식 보정: {corrected_text}"})

        if self.state == "CONFIRM_MENU":
            confirmation = main.classify_confirmation_answer_with_llm(corrected_text)
            if confirmation == "yes":
                response_text = f"{self.pending_menu} 주문 도와드릴게요."
                options = self.order_options(self.pending_menu)
                if options:
                    self.state = "OPTION_SELECT"
                    response_text += "\n\n" + options
                    response_text += "\n\n원하시는 옵션을 말씀해 주세요. 옵션이 필요 없으면 모두 기본으로 해달라고 말씀해 주세요."
                else:
                    self.reset()
                    response_text += "\n주문이 완료되었습니다."
                return self._response(events, response_text)

            if confirmation == "no":
                self.reset()
                return self._response(events, "알겠습니다. 다시 주문하실 메뉴를 말씀해 주세요.")

            return self._response(events, "맞으면 맞아요, 아니면 아니요라고 말씀해 주세요.")

        if self.state == "OPTION_SELECT":
            if main.is_default_options_answer(raw_text) or main.is_default_options_answer(corrected_text):
                response_text = main.get_default_option_summary(self.pending_menu)
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
                self.pending_menu = selected_menu
                self.pending_candidates = []
                self.state = "CONFIRM_MENU"
                response_text = main.generate_kiosk_response([self.pending_menu], corrected_text)
                return self._response(events, response_text)

            return self._response(events, "어떤 메뉴인지 다시 한 번 말씀해 주세요.")

        search_text = main.preserve_temp_for_search(raw_text, corrected_text)
        result = main.search_menu(search_text)

        if not result:
            response_text = main.generate_kiosk_response([], search_text)
            return self._response(events, response_text)

        if len(result) == 1:
            self.pending_menu = result[0]
            self.state = "CONFIRM_MENU"
            response_text = main.generate_kiosk_response([self.pending_menu], search_text)
            return self._response(events, response_text)

        self.pending_candidates = result
        self.state = "CHOOSE_MENU"
        response_text = main.generate_kiosk_response(result, search_text)
        return self._response(events, response_text, candidates=result)

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
        if category == "디저트":
            parts.append("디저트와 함께 드실 아메리카노를 추천드릴 수 있어요.")

        return "\n".join(parts)

    def _response(self, events, assistant_text, candidates=None):
        assistant_text = normalize_text(assistant_text)
        events.append({"role": "assistant", "text": assistant_text})
        return {
            "events": events,
            "state": self.state,
            "pending_menu": self.pending_menu,
            "candidates": candidates or self.pending_candidates,
        }


def get_session_id(handler):
    jar = cookies.SimpleCookie(handler.headers.get("Cookie", ""))
    if SESSION_COOKIE in jar:
        return jar[SESSION_COOKIE].value, False
    return uuid.uuid4().hex, True


def get_conversation(session_id):
    if session_id not in _sessions:
        _sessions[session_id] = Conversation()
    return _sessions[session_id]


def json_response(handler, status, payload, session_id=None):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    if session_id:
        handler.send_header("Set-Cookie", f"{SESSION_COOKIE}={session_id}; Path=/; SameSite=Lax")
    handler.end_headers()
    handler.wfile.write(body)


def text_response(handler, status, body, content_type="text/html; charset=utf-8", session_id=None):
    data = body.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    if session_id:
        handler.send_header("Set-Cookie", f"{SESSION_COOKIE}={session_id}; Path=/; SameSite=Lax")
    handler.end_headers()
    handler.wfile.write(data)


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


def make_tts_audio(text, session_id):
    main = kiosk()
    TTS_DIR.mkdir(parents=True, exist_ok=True)
    audio_name = f"{session_id or uuid.uuid4().hex}.mp3"
    audio_path = TTS_DIR / audio_name
    tts_text = main.normalize_tts_text(text)
    asyncio.run(main.save_tts_audio(tts_text, str(audio_path)))
    return f"/static/tts/{audio_name}?v={uuid.uuid4().hex}"


class KioskHandler(BaseHTTPRequestHandler):
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

        if parsed.path == "/static/ai_staff.png":
            file_response(self, STATIC_DIR / "ai_staff.png")
            return

        if parsed.path.startswith("/static/fonts/"):
            name = unquote(parsed.path.removeprefix("/static/fonts/"))
            file_response(self, STATIC_DIR / "fonts" / name)
            return

        text_response(self, 404, "Not found", "text/plain; charset=utf-8")

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

    def log_message(self, format, *args):
        print("%s - %s" % (self.address_string(), format % args))


INDEX_HTML = r"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI 키오스크</title>
  <style>
    @font-face {
      font-family: "Gaegu";
      src: url("/static/fonts/Gaegu-Light.ttf") format("truetype");
      font-weight: 300;
      font-style: normal;
      font-display: swap;
    }

    @font-face {
      font-family: "Gaegu";
      src: url("/static/fonts/Gaegu-Regular.ttf") format("truetype");
      font-weight: 400;
      font-style: normal;
      font-display: swap;
    }

    @font-face {
      font-family: "Gaegu";
      src: url("/static/fonts/Gaegu-Bold.ttf") format("truetype");
      font-weight: 700 900;
      font-style: normal;
      font-display: swap;
    }

    :root {
      --bg: #f4efe6;
      --paper: #fffdf8;
      --ink: #1f2421;
      --muted: #71685d;
      --line: #ded2c0;
      --green: #1f7a63;
      --green-dark: #145846;
      --amber: #b96e22;
      --cream: #fff7e9;
      --blue: #246a8f;
      --shadow: 0 16px 40px rgba(44, 35, 25, 0.12);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Gaegu", "Noto Sans KR", "Malgun Gothic", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }

    button, textarea { font: inherit; }
    button { cursor: pointer; }

    .screen {
      display: none;
      min-height: 100vh;
      padding: 28px;
    }

    .screen.active { display: flex; }

    .home {
      align-items: center;
      justify-content: center;
    }

    .home-inner {
      width: min(1040px, 100%);
      display: grid;
      gap: 34px;
      text-align: center;
    }

    .brand {
      display: grid;
      gap: 10px;
    }

    .brand h1 {
      margin: 0;
      font-size: clamp(52px, 7vw, 88px);
      letter-spacing: 0;
    }

    .brand p {
      margin: 0;
      color: var(--muted);
      font-size: clamp(18px, 2.2vw, 25px);
    }

    .choice-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 24px;
    }

    .choice {
      min-height: 280px;
      border: 2px solid #08765d;
      border-radius: 8px;
      background: var(--paper);
      box-shadow: var(--shadow);
      padding: 34px;
      display: grid;
      align-content: center;
      justify-items: center;
      gap: 18px;
      color: var(--ink);
      transition: transform .18s ease, border-color .18s ease;
    }

    .choice.ai {
      border-color: #1f64b2;
    }

    .choice:hover {
      transform: translateY(-3px);
      border-color: #08765d;
    }

    .choice.ai:hover {
      border-color: #1f64b2;
    }

    .choice-icon {
      width: clamp(118px, 14vw, 156px);
      height: clamp(118px, 14vw, 156px);
      border-radius: 30px;
      display: grid;
      place-items: center;
      background: transparent;
      box-shadow: 0 12px 26px rgba(31, 45, 38, .14);
      overflow: hidden;
    }

    .choice.ai .choice-icon { background: transparent; }

    .choice-icon img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }

    .choice strong {
      font-size: clamp(25px, 3.2vw, 38px);
      line-height: 1.25;
      letter-spacing: 0;
    }

    .kiosk {
      flex-direction: column;
      gap: 18px;
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
    }

    .back {
      min-width: 120px;
      min-height: 52px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--ink);
      font-weight: 700;
    }

    .topbar-spacer {
      width: 120px;
      min-height: 52px;
      pointer-events: none;
    }

    .direct-layout {
      flex: 1;
      min-height: 0;
      display: grid;
      grid-template-columns: 1fr;
      grid-template-rows: auto minmax(0, 1fr) auto;
      gap: 18px;
    }

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
      border-radius: 8px;
      background: #fff;
      color: var(--ink);
      font-weight: 800;
      font-size: 18px;
    }

    .tab.active {
      border-color: var(--green);
      background: var(--green);
      color: #fff;
    }

    .menu-browser {
      min-height: 0;
      display: grid;
      grid-template-columns: 64px minmax(0, 1fr) 64px;
      align-items: stretch;
      gap: 12px;
    }

    .menu-arrow {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--ink);
      font-size: 42px;
      font-weight: 900;
      box-shadow: 0 8px 22px rgba(44, 35, 25, 0.08);
    }

    .menu-arrow:disabled {
      color: #b8b8b8;
      background: #f3f3f3;
      cursor: not-allowed;
      box-shadow: none;
    }

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
      border-radius: 8px;
      background: var(--paper);
      overflow: hidden;
      display: grid;
      grid-template-rows: 210px 1fr;
      text-align: left;
      color: var(--ink);
      box-shadow: 0 8px 22px rgba(44, 35, 25, 0.08);
    }

    .menu-card:hover { border-color: var(--green); }

    .photo {
      width: 100%;
      height: 210px;
      background: #fff;
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
      border-radius: 8px;
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
      border-radius: 8px;
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
      border-radius: 8px;
      background: #fff;
      border: 1px solid var(--line);
    }

    .cancel-order {
      min-height: 48px;
      border: 0;
      background: transparent;
      color: var(--ink);
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
      background: var(--green);
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
      border-radius: 28px;
      background: var(--green);
      color: #fff;
      font-size: 22px;
      font-weight: 900;
      box-shadow: 0 10px 24px rgba(13, 118, 93, .22);
    }

    #finishDirect:hover {
      background: var(--green-dark);
    }

    .primary {
      min-height: 56px;
      border: 1px solid var(--green);
      border-radius: 8px;
      background: var(--green);
      color: #fff;
      font-weight: 800;
      font-size: 18px;
    }

    .primary:hover { background: var(--green-dark); }

    .secondary {
      min-height: 52px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--ink);
      font-weight: 800;
    }

    .ai-layout {
      flex: 1;
      min-height: 0;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      gap: 18px;
    }

    .voice-stage {
      display: grid;
      place-items: center;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--paper);
      padding: 30px;
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
      border-radius: 8px;
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
      border-radius: 8px;
      background: #1f64b2;
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
    }

    .messages {
      overflow: auto;
      display: grid;
      align-content: start;
      gap: 10px;
    }

    .message {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
      word-break: keep-all;
      overflow-wrap: anywhere;
      background: #fff;
    }

    .message.assistant { background: var(--cream); }
    .message.user { background: #e8f3ef; }
    .message.system { color: var(--muted); font-size: 14px; }

    .modal {
      position: fixed;
      inset: 0;
      display: none;
      align-items: center;
      justify-content: center;
      background: rgba(24, 20, 15, .44);
      padding: 20px;
    }

    .modal.active { display: flex; }

    .modal-box {
      width: min(560px, 100%);
      border-radius: 8px;
      background: var(--paper);
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
      border-radius: 8px;
      background: #fff;
      padding: 0 14px;
      font-weight: 700;
    }

    .option.active {
      border-color: var(--green);
      background: var(--green);
      color: #fff;
    }

    @media (max-width: 900px) {
      .screen { padding: 18px; }
      .choice-grid, .direct-layout, .ai-layout { grid-template-columns: 1fr; }
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
          <button class="mic" id="voiceOrder">음성 주문 시작</button>
        </div>
      </div>
      <aside class="ai-log">
        <h3>AI 점원 안내</h3>
        <div id="messages" class="messages"></div>
        <button class="secondary" id="stopVoice">음성 안내 끄기</button>
      </aside>
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
      ai: document.querySelector("#aiScreen")
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

    let menuItems = [];
    let activeCategory = "전체";
    let menuPage = 0;
    let selectedMenu = null;
    let selectedOptions = {};
    let cart = [];
    let voiceEnabled = true;
    let currentAudio = null;

    function showScreen(name) {
      Object.values(screens).forEach((screen) => screen.classList.remove("active"));
      screens[name].classList.add("active");
      currentAudio?.pause();
    }

    function resetAiUi() {
      messages.innerHTML = "";
      voiceTitle.textContent = "원하시는 메뉴를 말씀해 주세요.";
      voiceHelp.textContent = "버튼을 누르시면 주문을 도와드릴게요!";
      setListening(false);
    }

    async function resetConversation() {
      try {
        await post("/api/reset", {});
      } catch (error) {
        console.error("AI reset failed", error);
      }
    }

    function resetDirectOrder() {
      cart = [];
      selectedMenu = null;
      selectedOptions = {};
      modal.classList.remove("active");
      renderCart();
    }

    function resetDirectBrowser() {
      activeCategory = "전체";
      menuPage = 0;
      if (menuItems.length) {
        renderCategories();
        renderMenu();
      }
    }

    document.querySelector("#directStart").addEventListener("click", () => {
      resetDirectBrowser();
      showScreen("direct");
      loadMenu();
    });

    document.querySelector("#aiStart").addEventListener("click", () => {
      resetAiUi();
      showScreen("ai");
      addMessage("assistant", "안녕하세요. 원하시는 메뉴를 말씀해 주세요.");
      speak("안녕하세요. 원하시는 메뉴를 말씀해 주세요.");
    });

    document.querySelectorAll("[data-go-home]").forEach((button) => {
      button.addEventListener("click", () => {
        resetAiUi();
        resetDirectOrder();
        resetConversation();
        showScreen("home");
      });
    });

    menuPrev.addEventListener("click", () => {
      if (menuPage <= 0) return;
      menuPage -= 1;
      renderMenu();
    });

    menuNext.addEventListener("click", () => {
      menuPage += 1;
      renderMenu();
    });

    window.addEventListener("resize", () => {
      renderMenu();
    });

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

    function getMenuPageSize() {
      if (window.innerWidth <= 560) return 2;
      if (window.innerWidth <= 900) return 4;
      return 8;
    }

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

    function categoryRank(category, order) {
      const index = order.indexOf(category);
      return index === -1 ? order.length : index;
    }

    function menuRank(name, order) {
      const index = order.indexOf(name);
      return index === -1 ? order.length : index;
    }

    function baseMenuName(name) {
      return String(name || "").replace(/^ICE\s+/, "").replace(/^HOT\s+/, "");
    }

    function teaBaseName(name) {
      const base = baseMenuName(name).replace(/^핫/, "");
      return base === "초코" ? "초코" : base;
    }

    function teaMenuRank(name) {
      const base = teaBaseName(name);
      const index = menuItems.findIndex((item) => item.category === "차" && teaBaseName(item.name) === base);
      return index === -1 ? menuItems.length : index;
    }

    function tempOrder(name) {
      if (String(name).startsWith("HOT ")) return 0;
      if (String(name).startsWith("ICE ")) return 1;
      return 0;
    }

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

    document.querySelector("#addCart").addEventListener("click", () => {
      cart.push({menu: selectedMenu, options: {...selectedOptions}, quantity: 1});
      modal.classList.remove("active");
      renderCart();
    });

    document.querySelector("#closeModal").addEventListener("click", () => modal.classList.remove("active"));
    document.querySelector("#clearCart").addEventListener("click", () => {
      resetDirectOrder();
    });

    document.querySelector("#finishDirect").addEventListener("click", () => {
      if (!cart.length) {
        alert("메뉴를 먼저 선택해 주세요.");
        return;
      }
      alert("주문이 완료되었습니다.");
      resetDirectOrder();
      showScreen("home");
    });

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

    voiceOrder.addEventListener("click", async () => {
      setListening(true);
      try {
        const data = await post("/api/voice", {});
        handleAiEvents(data);
      } catch (error) {
        addMessage("system", error.message);
        speak("처리 중 문제가 생겼어요. 다시 시도해 주세요.");
      } finally {
        setListening(false);
      }
    });

    document.querySelector("#resetAi").addEventListener("click", async () => {
      const data = await post("/api/reset", {});
      resetAiUi();
      handleAiEvents(data);
    });

    document.querySelector("#stopVoice").addEventListener("click", () => {
      voiceEnabled = !voiceEnabled;
      currentAudio?.pause();
      document.querySelector("#stopVoice").textContent = voiceEnabled ? "음성 안내 끄기" : "음성 안내 켜기";
    });

    function handleAiEvents(data) {
      (data.events || []).forEach((event) => {
        addMessage(event.role, event.text);
        if (event.role === "assistant") speak(event.text);
      });
      if (data.state === "CONFIRM_MENU") {
        voiceTitle.textContent = "메뉴가 맞는지 확인해 주세요.";
        voiceHelp.textContent = "맞으면 맞아요, 아니면 아니요라고 말씀해 주세요.";
      } else if (data.state === "OPTION_SELECT") {
        voiceTitle.textContent = "옵션을 선택해 주세요.";
        voiceHelp.textContent = "원하시는 옵션을 말하거나 모두 기본으로 진행할 수 있어요.";
      } else if (data.state === "CHOOSE_MENU") {
        voiceTitle.textContent = "후보 메뉴 중 선택해 주세요.";
        voiceHelp.textContent = (data.candidates || []).join(", ");
      } else {
        voiceTitle.textContent = "원하시는 메뉴를 말씀해 주세요.";
        voiceHelp.textContent = "버튼을 누르시면 주문을 도와드릴게요!";
      }
    }

    function setListening(isListening) {
      voiceOrder.disabled = isListening;
      voiceOrder.classList.toggle("listening", isListening);
      voiceOrder.textContent = isListening ? "듣고 있어요..." : "음성 주문 시작";
      if (isListening) {
        voiceTitle.textContent = "말씀해 주세요.";
        voiceHelp.textContent = "녹음이 끝나면 자동으로 인식하고 주문을 이어갑니다.";
      }
    }

    function addMessage(role, text) {
      const el = document.createElement("div");
      el.className = `message ${role}`;
      el.textContent = text;
      messages.appendChild(el);
      messages.scrollTop = messages.scrollHeight;
    }

    async function speak(text) {
      if (!voiceEnabled) return;
      currentAudio?.pause();
      try {
        const data = await post("/api/tts", {text});
        if (!voiceEnabled) return;
        currentAudio = new Audio(data.audio_url);
        await currentAudio.play();
      } catch (error) {
        console.error("TTS playback failed", error);
      }
    }

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


def run():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((HOST, PORT), KioskHandler)
    print(f"AI Assistant Kiosk web app: http://{HOST}:{PORT}")
    print(f"Menu image folder: {IMAGE_DIR}")
    server.serve_forever()


if __name__ == "__main__":
    run()
