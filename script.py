"""
Meet Caption Overlay
====================

A transparent, click-through fullscreen overlay that:
  1. Receives live captions from a browser extension via a local Flask API,
  2. Displays them on top of every other window using a Qt WebEngine view,
  3. On a hotkey, sends the running transcript + context to Gemini and renders
     an AI-designed HTML summary in the same overlay.

Single-file by design so it can be launched with `python script.py`.
"""

from __future__ import annotations

import datetime as dt
import html
import logging
import os
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from google import genai

import keyboard  # NB: on macOS/Linux this typically requires root.

from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import QApplication, QMainWindow

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent
SAVE_FILE = SCRIPT_DIR / "meet_captions.txt"
CONTEXT_FILE = SCRIPT_DIR / "context.txt"

FLASK_HOST = "127.0.0.1"
FLASK_PORT = 5000

ANALYSIS_HOTKEY = "0"
QUIT_HOTKEY = "ctrl+shift+q"

CAPTION_TTL_SECONDS = 15
ANALYSIS_TTL_SECONDS = 30
ERROR_TTL_SECONDS = 8
MAX_SPEAKER_LEN = 200
MAX_TEXT_LEN = 2_000
MAX_TRANSCRIPT_CHARS = 16_000  # Trim oldest transcript content beyond this.

GEMINI_MODEL = "gemini-3.1-flash-lite-preview"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("overlay")
# Werkzeug's per-request log spams once per caption; quiet it down.
logging.getLogger("werkzeug").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------
_api_key = os.getenv("GEMINI_API_KEY")
if not _api_key:
    log.warning("GEMINI_API_KEY not set; the analysis hotkey will show an error.")
gemini_client: genai.Client | None = genai.Client(api_key=_api_key) if _api_key else None

# ---------------------------------------------------------------------------
# Qt thread-safe signals
# ---------------------------------------------------------------------------
class UICommunicator(QObject):
    # html, auto_clear_seconds (0 = persist until next update)
    update_screen = pyqtSignal(str, int)
    quit_app = pyqtSignal()


communicator = UICommunicator()

# ---------------------------------------------------------------------------
# Analysis state
# ---------------------------------------------------------------------------
# Used as both a "one analysis at a time" gate AND a "summary is on screen,
# don't overwrite with captions" flag. The lock is held until the on-screen
# message has been displayed for its full TTL, so live captions resume only
# after the summary auto-clears.
_analysis_lock = threading.Lock()


def is_analyzing() -> bool:
    return _analysis_lock.locked()


# ---------------------------------------------------------------------------
# HTML templates
# ---------------------------------------------------------------------------
BLANK_HTML = (
    "<!DOCTYPE html><html><body "
    "style='background-color: transparent !important; overflow: hidden;'></body></html>"
)

STARTUP_HTML = """
<!DOCTYPE html>
<html>
<head><style>
  body {
    background-color: transparent !important; overflow: hidden;
    display: flex; justify-content: center; align-items: center;
    height: 100vh; margin: 0;
    font-family: system-ui, -apple-system, Segoe UI, sans-serif;
    color: #FFEB3B;
    text-shadow: 0 2px 6px rgba(0,0,0,0.9);
    font-size: 24px; font-weight: 600;
  }
</style></head>
<body>Listening for subtitles…</body>
</html>
"""


def render_caption_html(speaker: str, text: str) -> str:
    """Build the caption overlay. All user-supplied text is HTML-escaped."""
    safe_speaker = html.escape(speaker)
    safe_text = html.escape(text)
    return f"""<!DOCTYPE html>
<html><head><style>
  body {{
    background-color: transparent !important; overflow: hidden;
    display: flex; justify-content: center; align-items: flex-end;
    height: 100vh; margin: 0; padding: 0 0 8vh 0;
    font-family: system-ui, -apple-system, Segoe UI, sans-serif;
  }}
  .caption {{
    max-width: 80%;
    text-align: center;
    color: #FFEB3B;
    font-size: 32px;
    font-weight: 700;
    line-height: 1.3;
    text-shadow:
      2px 2px 0 #000, -2px -2px 0 #000,
      2px -2px 0 #000, -2px 2px 0 #000,
      0 2px 8px rgba(0,0,0,0.9);
  }}
  .speaker {{ color: #80DEEA; margin-right: 0.4em; }}
</style></head>
<body><div class="caption">
  <span class="speaker">{safe_speaker}:</span>{safe_text}
</div></body></html>"""


def render_status_html(message: str, color: str = "#00FFFF") -> str:
    safe_msg = html.escape(message)
    return f"""<!DOCTYPE html>
<html><head><style>
  body {{
    background-color: transparent !important; overflow: hidden;
    display: flex; justify-content: center; align-items: center;
    height: 100vh; margin: 0;
    font-family: system-ui, -apple-system, Segoe UI, sans-serif;
    color: {color}; font-size: 40px; font-weight: 700;
    text-shadow: 0 2px 8px rgba(0,0,0,0.9);
  }}
</style></head>
<body>{safe_msg}</body></html>"""


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)


@app.route("/api/captions", methods=["POST"])
def save_caption():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"status": "error", "reason": "invalid json"}), 400

    speaker = str(data.get("speaker") or "Unknown")[:MAX_SPEAKER_LEN]
    text = str(data.get("text") or "")[:MAX_TEXT_LEN]

    if not text.strip():
        return jsonify({"status": "ok", "noop": True}), 200

    # Don't overwrite the AI summary while it's still on screen.
    if not is_analyzing():
        communicator.update_screen.emit(
            render_caption_html(speaker, text), CAPTION_TTL_SECONDS
        )

    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with SAVE_FILE.open("a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] [{speaker}] {text}\n")
    except OSError:
        log.exception("Could not write to %s", SAVE_FILE)

    return jsonify({"status": "ok"}), 200


def run_flask() -> None:
    log.info(
        "Caption API listening on http://%s:%d/api/captions", FLASK_HOST, FLASK_PORT
    )
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, use_reloader=False)


# ---------------------------------------------------------------------------
# Gemini analysis
# ---------------------------------------------------------------------------
def _read_text_safely(path: Path, tail_chars: int | None = None) -> str:
    if not path.exists():
        return ""
    try:
        with path.open("r", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        log.exception("Could not read %s", path)
        return ""
    if tail_chars is not None and len(text) > tail_chars:
        text = "…[earlier transcript truncated]…\n" + text[-tail_chars:]
    return text


def _strip_markdown_fences(s: str) -> str:
    """Robustly strip ```html ... ``` fences if the model added any."""
    s = s.strip()
    if s.startswith("```"):
        # Drop the opening fence (and optional language tag on the same line).
        s = s.split("\n", 1)[1] if "\n" in s else ""
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()


def _build_analysis_prompt(context: str, transcript: str) -> str:
    return f"""\
You are designing a transparent, fullscreen HUD overlay (1920x1080) that
summarises the meeting below for the wearer at a glance.

--- CONTEXT (background notes) ---
{context}

--- LIVE TRANSCRIPT ---
{transcript}

Surface, in order of priority:
  • Key concepts as short tags (single words or short phrases — NOT sentences)
  • Action items as concise bullets
  • Decisions and open questions, if any

Hard requirements (failure to comply breaks the overlay):
  1. Output VALID, RAW HTML only. No markdown fences, no commentary, no <html> wrapper boilerplate beyond what you need.
  2. Include a <style> block.
  3. body MUST include: background-color: transparent !important; overflow: hidden;
  4. Every text element must remain readable over a busy desktop. Use either strong text-shadow OR rgba(0,0,0,0.7) padded plates.
  5. Center content in the viewport. Assume 1920x1080.
  6. Use system-ui or other web-safe fonts only."""


def trigger_analysis() -> None:
    if not _analysis_lock.acquire(blocking=False):
        log.info("Analysis already in progress; ignoring hotkey.")
        return

    hold_seconds = 0
    try:
        communicator.update_screen.emit(
            render_status_html("Analyzing meeting context…"), 0
        )

        if gemini_client is None:
            communicator.update_screen.emit(
                render_status_html("GEMINI_API_KEY missing", color="#FF5252"),
                ERROR_TTL_SECONDS,
            )
            hold_seconds = ERROR_TTL_SECONDS
            return

        context_text = _read_text_safely(CONTEXT_FILE)
        transcript_text = _read_text_safely(SAVE_FILE, tail_chars=MAX_TRANSCRIPT_CHARS)

        if not transcript_text.strip():
            communicator.update_screen.emit(
                render_status_html("No transcript captured yet", color="#FFA726"),
                ERROR_TTL_SECONDS,
            )
            hold_seconds = ERROR_TTL_SECONDS
            return

        prompt = _build_analysis_prompt(context_text, transcript_text)
        try:
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL, contents=prompt
            )
        except Exception as e:  # noqa: BLE001 - surface any SDK error to the UI
            log.exception("Gemini API call failed")
            communicator.update_screen.emit(
                render_status_html(f"API error: {e}", color="#FF5252"),
                ERROR_TTL_SECONDS,
            )
            hold_seconds = ERROR_TTL_SECONDS
            return

        raw_html = _strip_markdown_fences(getattr(response, "text", "") or "")
        if not raw_html:
            communicator.update_screen.emit(
                render_status_html("Empty response from model", color="#FF5252"),
                ERROR_TTL_SECONDS,
            )
            hold_seconds = ERROR_TTL_SECONDS
            return

        communicator.update_screen.emit(raw_html, ANALYSIS_TTL_SECONDS)
        hold_seconds = ANALYSIS_TTL_SECONDS
    finally:
        # Hold the lock while the message is on screen, so live captions
        # don't immediately overwrite the summary.
        if hold_seconds > 0:
            time.sleep(hold_seconds)
        _analysis_lock.release()


# ---------------------------------------------------------------------------
# Hotkeys
# ---------------------------------------------------------------------------
def setup_hotkeys() -> None:
    try:
        # Run analysis on its own thread so the keyboard callback returns
        # immediately and doesn't block the global hook.
        keyboard.add_hotkey(
            ANALYSIS_HOTKEY,
            lambda: threading.Thread(target=trigger_analysis, daemon=True).start(),
        )
        keyboard.add_hotkey(QUIT_HOTKEY, communicator.quit_app.emit)
        log.info(
            "Hotkeys registered: '%s' = analyze, '%s' = quit",
            ANALYSIS_HOTKEY,
            QUIT_HOTKEY,
        )
    except Exception:
        # On macOS / Linux the `keyboard` package usually requires root.
        log.exception(
            "Could not register global hotkeys "
            "(on macOS/Linux you may need to run with sudo)."
        )


# ---------------------------------------------------------------------------
# Overlay window
# ---------------------------------------------------------------------------
class OverlayWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.webview = QWebEngineView()
        self.webview.page().setBackgroundColor(QColor(0, 0, 0, 0))
        self.webview.setHtml(STARTUP_HTML)
        self.setCentralWidget(self.webview)

        # Single shared timer, restarted on every update. Eliminates the race
        # where a stale auto-clear thread wipes a fresh caption.
        self._clear_timer = QTimer(self)
        self._clear_timer.setSingleShot(True)
        self._clear_timer.timeout.connect(self._clear_display)

        communicator.update_screen.connect(self._on_update)
        communicator.quit_app.connect(QApplication.instance().quit)

        self.showFullScreen()

    def _on_update(self, raw_html: str, auto_clear_seconds: int) -> None:
        self.webview.setHtml(raw_html)
        self._clear_timer.stop()
        if auto_clear_seconds > 0:
            self._clear_timer.start(auto_clear_seconds * 1000)

    def _clear_display(self) -> None:
        # Defensive: if a summary is still active, leave it alone.
        if not is_analyzing():
            self.webview.setHtml(BLANK_HTML)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    threading.Thread(target=run_flask, daemon=True).start()
    setup_hotkeys()

    qt_app = QApplication(sys.argv)
    window = OverlayWindow()  # noqa: F841 - kept alive by Qt
    return qt_app.exec()


if __name__ == "__main__":
    sys.exit(main())
