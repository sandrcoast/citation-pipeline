# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Citation Pipeline Contributors
"""
app.py — Interactive REPL client for the Citation Pipeline middleware.

Type a prompt and press Enter. The client builds the JSON body, prints the
full curl command it's about to run, executes it, and pretty-prints the
JSON response. Shortcuts begin with '/'. Exit with Ctrl+C or /quit.

This client is intentionally stateless and minimal:
  - does NOT manage the Python venv
  - does NOT start/stop/health-check the uvicorn middleware server
  - does NOT touch git, Ollama, or Playwright
  - uses only the Python standard library

Run (with the middleware already running on http://localhost:8000):

    python app.py
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
_URL_RE = re.compile(r"https?://[^\s<>\"'`\])}]+", re.IGNORECASE)
_SLUG_STRIP = re.compile(r"[\s/&?#=:]+")


def _slug_from_prompt(prompt: str) -> str:
    m = _URL_RE.search(prompt)
    if m:
        raw = re.sub(r"^https?://", "", m.group(0), flags=re.IGNORECASE)
    else:
        raw = prompt
    cleaned = _SLUG_STRIP.sub("", raw)
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "", cleaned)
    return cleaned[:15] or "prompt"


def _result_filename(prompt: str) -> Path:
    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    return RESULTS_DIR / f"{ts}_{_slug_from_prompt(prompt)}.json"


def _save_full_response(prompt: str, parsed: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _result_filename(prompt)
    path.write_text(
        json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path

DEFAULT_URL = "http://localhost:8000"
DEFAULT_MODEL = "gemma3:1b"

HELP = """
Shortcuts (start with /):
  /help                 show this help
  /status               show current url / model / citations flag
  /citations on|off     toggle the citations flag for subsequent requests
  /model <name>         set model (default: gemma3:1b)
  /url <base>           set middleware base URL (default: http://localhost:8000)
  /paste                send system clipboard contents as the prompt (alias: /p)
  /quit                 exit (aliases: /q, /exit)

Tip: if Shift+Insert / Ctrl+Insert don't paste in your terminal,
use Ctrl+V (Windows Terminal), right-click (legacy conhost),
or type /paste to pull the prompt from the clipboard.

Inline per-request flags (append to the end of your prompt):
  --citations:y|n       override citations for this one request
  --cit:y|n             short form
  -cit:y|n              shortest form

Anything else is sent as a prompt to POST <url>/api/generate.
Press Ctrl+C at any time to exit.
"""


def parse_inline_flags(line: str) -> tuple[str, dict]:
    """
    Strip trailing --flag:value tokens from the prompt text and return
    (clean_prompt, overrides). Unknown tokens are left in the prompt.
    """
    overrides: dict = {}
    kept: list[str] = []
    for tok in line.split():
        low = tok.lower()
        if (
            low.startswith("--citations:")
            or low.startswith("--cit:")
            or low.startswith("-cit:")
        ):
            val = tok.split(":", 1)[1].lower()
            overrides["citations"] = val in ("y", "yes", "true", "on", "1")
        else:
            kept.append(tok)
    return " ".join(kept), overrides


def send_prompt(prompt: str, model: str, citations: bool, base_url: str) -> None:
    body = {"model": model, "prompt": prompt, "citations": citations}
    body_json = json.dumps(body, ensure_ascii=False)
    cmd = [
        "curl", "-s", "-X", "POST",
        f"{base_url}/api/generate",
        "-H", "Content-Type: application/json",
        "-d", body_json,
    ]
    # Show exactly what's being executed (POSIX-quoted for readability)
    printable = " ".join(shlex.quote(c) for c in cmd)
    print(f"\n$ {printable}\n")

    try:
        res = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8"
        )
    except FileNotFoundError:
        print("[error] curl executable not found on PATH.")
        print("        On Windows 10/11 it ships at C:\\Windows\\System32\\curl.exe.")
        return

    if res.returncode != 0:
        print(f"[curl exited {res.returncode}]")
        if res.stderr:
            print(res.stderr.strip())
        return

    # Pretty-print JSON if possible, otherwise dump raw body
    try:
        parsed = json.loads(res.stdout)
    except json.JSONDecodeError:
        print(res.stdout)
        return

    if isinstance(parsed, dict) and "error" in parsed:
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
        return

    # Pull the LLM answer out of the JSON for a clean text-first display.
    answer_text = ""
    if isinstance(parsed.get("response"), str):
        answer_text = parsed["response"]
    elif isinstance(parsed.get("message"), dict):
        answer_text = parsed["message"].get("content", "") or ""

    print("--- LLM answer ---")
    print(answer_text.strip() or "[empty answer — model returned no text]")
    print()

    summary_keys = (
        "model", "_prompt_id", "_total_ms",
        "citation_records_count", "_fetched_sources",
    )
    summary = {k: parsed[k] for k in summary_keys if k in parsed}
    print("--- response metadata ---")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    citations = []
    meta = parsed.get("citation_metadata") if isinstance(parsed, dict) else None
    if isinstance(meta, dict):
        citations = meta.get("citations") or []
    if citations:
        print("\n--- first 3 citation records ---")
        print(json.dumps(citations[:3], ensure_ascii=False, indent=2))

    try:
        path = _save_full_response(prompt, parsed)
        print(
            f"\nFull collection of citation records is available in following file "
            f"in results/{path.name}"
        )
    except OSError as e:
        print(f"\n[warn] could not save full response: {e}")


def read_clipboard() -> str | None:
    """Return system clipboard text via tkinter (stdlib). None on failure."""
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        try:
            text = root.clipboard_get()
        finally:
            root.destroy()
        return text
    except Exception as e:
        print(f"[clipboard unavailable: {e}]")
        return None


def handle_shortcut(line: str, state: dict) -> bool:
    """Run a /shortcut. Return False to exit the REPL, True to continue."""
    parts = line[1:].split(maxsplit=1)
    if not parts:
        return True
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("quit", "exit", "q"):
        return False
    if cmd in ("paste", "p"):
        text = read_clipboard()
        if not text:
            return True
        text = text.strip()
        if not text:
            print("[clipboard is empty]")
            return True
        print(f"[pasted {len(text)} chars from clipboard]")
        prompt, overrides = parse_inline_flags(text)
        citations = overrides.get("citations", state["citations"])
        if not prompt:
            print("[empty prompt after flag stripping — nothing to send]")
            return True
        send_prompt(prompt, state["model"], citations, state["url"])
        return True

    if cmd == "help":
        print(HELP)
    elif cmd == "status":
        print(f"url        = {state['url']}")
        print(f"model      = {state['model']}")
        print(f"citations  = {'on' if state['citations'] else 'off'}")
    elif cmd == "citations":
        low = arg.lower()
        if low in ("on", "true", "yes", "y", "1"):
            state["citations"] = True
        elif low in ("off", "false", "no", "n", "0"):
            state["citations"] = False
        else:
            print("usage: /citations on|off")
            return True
        print(f"citations = {'on' if state['citations'] else 'off'}")
    elif cmd == "model":
        if arg:
            state["model"] = arg
        print(f"model = {state['model']}")
    elif cmd == "url":
        if arg:
            state["url"] = arg.rstrip("/")
        print(f"url = {state['url']}")
    else:
        print(f"unknown shortcut: /{cmd}  (type /help)")
    return True


def main() -> int:
    state = {"url": DEFAULT_URL, "model": DEFAULT_MODEL, "citations": True}
    print("Citation Pipeline REPL")
    print(
        f"Middleware: {state['url']}   "
        f"Model: {state['model']}   "
        f"Citations: {'on' if state['citations'] else 'off'}"
    )
    print("Type your prompt and press Enter. Press Ctrl+C to exit.")
    print("Type /help for shortcuts.\n")

    try:
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                print()
                return 0
            if not line:
                continue
            if line.startswith("/"):
                if not handle_shortcut(line, state):
                    return 0
                continue

            prompt, overrides = parse_inline_flags(line)
            citations = overrides.get("citations", state["citations"])
            if not prompt:
                print("[empty prompt after flag stripping — nothing to send]")
                continue
            send_prompt(prompt, state["model"], citations, state["url"])
    except KeyboardInterrupt:
        print("\n[exit]")
        return 0


if __name__ == "__main__":
    sys.exit(main())
