"""Real Whisper verification - downloads a real Whisper model and transcribes real speech.

Used by CI (GitHub's runners can reach Hugging Face) and usable on any PC with internet:

    python tools/verify_whisper.py --model small --bangla-model large-v3-turbo --report whisper-report.json

Checks, with the app's own code:
1. local Whisper transcribes tests/data/jfk_speech.mp4 (real human speech, public domain) - timestamps included;
2. audio longer than 3 minutes is cut to the first 180 s; silence is reported as "no speech", never as a transcript;
3. optional: a Bangla clip (synthetic TTS voice - an approximation of real Bangla speech);
4. the real GUI: Analyze Selected Videos shows the transcript, Generate Preview sends it to title generation,
   unsupported suggestions are rejected, another suggestion can be chosen, Apply renames and Undo restores.
   (Title generation uses a local stand-in AI here - no AI key on CI - so AI title *quality* is not measured.)
Exit code 0 = everything passed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
JFK = ROOT / "tests" / "data" / "jfk_speech.mp4"
BANGLA = ROOT / "tests" / "data" / "bangla_tts_metro.mp4"
JFK_PHRASE = "ask not what your country can do for you ask what you can do for your country"
BANGLA_WORDS = ["ঢাকার", "মেট্রোরেল", "ট্রেন", "প্রতিদিন", "অফিসে", "ভাড়া", "স্টেশনে"]

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""), flush=True)
    return ok


def coverage(text: str, phrase_words: list[str]) -> float:
    from app.ai.title_quality import words

    said = set(words(text))
    return sum(w in said for w in phrase_words) / len(phrase_words)


def write_wav(path: Path, seconds: float, frames: bytes = b"") -> None:
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(frames + b"\x00\x00" * int(seconds * 16000 - len(frames) // 2))


def pipeline_checks(ctx, work: Path, model: str) -> str:  # type: ignore[no-untyped-def]
    from app.ai.title_quality import words
    from app.ai.transcription import make_transcript_service
    from app.files.audio import AudioExtractor

    ctx.settings.update(transcription_service="auto", whisper_model=model)
    service = make_transcript_service(ctx)
    check("local Whisper selected by Automatic", service.label.startswith(f"Whisper {model} (local"), service.label)
    stages: list[str] = []
    started = time.time()
    result = service.get(JFK, 180, "auto", step=stages.append)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), flush=True)
    score = coverage(result.text, words(JFK_PHRASE))
    check("real speech transcribed (JFK)", result.ok and score >= 0.8,
          f"{time.time() - started:.0f}s, stages={stages}, language={result.language}, phrase coverage={score:.0%}: {result.text!r}")  # fmt: skip
    check("timestamps returned", bool(result.segments) and all("start" in s for s in result.segments), "; ".join(result.timestamped()[:3]))

    with AudioExtractor().extracted_audio(JFK, 180) as clip, wave.open(str(clip.path)) as w:
        speech = w.readframes(w.getnframes())
    long_audio = work / "long_200s.wav"
    write_wav(long_audio, 200, speech)
    long_result = service.get(long_audio, 180, "auto")
    check("longer than 3 minutes: only the first 180 s analysed", long_result.ok and abs(long_result.analyzed_seconds - 180) < 0.5,
          f"analysed {long_result.analyzed_range}")  # fmt: skip
    silent = work / "silent_5s.wav"
    write_wav(silent, 5)
    silent_result = service.get(silent, 180, "auto")
    check("silence is not reported as a transcript", silent_result.status in ("empty", "insufficient") and not silent_result.ok,
          f"{silent_result.status}: {silent_result.error}")  # fmt: skip
    cached = service.get(JFK, 180, "auto")
    check("transcript cache reused", cached.from_cache and cached.text == result.text)
    return result.text


def _bangla_share(text: str) -> float:
    letters = [c for c in text if c.isalpha() or "\u0980" <= c <= "\u09ff"]
    return sum("\u0980" <= c <= "\u09ff" for c in letters) / len(letters) if letters else 0.0


def bangla_check(ctx, model: str) -> None:  # type: ignore[no-untyped-def]
    """Synthetic (robotic) Bangla voice: automatic language detection is reported; with the language set it must work."""
    from app.ai.transcription import make_transcript_service

    ctx.settings.update(whisper_model=model)
    for language in ("auto", "bn"):
        started = time.time()
        result = make_transcript_service(ctx).get(BANGLA, 180, language)
        score, script = coverage(result.text, BANGLA_WORDS), _bangla_share(result.text)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), flush=True)
        detail = f"{time.time() - started:.0f}s, language={result.language}, Bangla script={script:.0%}, key words found={score:.0%}: {result.text!r}"
        if language == "auto":  # informational: a robotic TTS voice is not representative of real speakers
            print(f"[INFO] Bangla with automatic language detection ({model}) - {detail}", flush=True)
            continue
        check(f"Bangla speech (synthetic voice) with language set to Bangla, {model}",
              result.ok and script >= 0.8 and score >= 0.3, detail)  # fmt: skip


def gui_checks(ctx, work: Path, transcript: str, model: str) -> None:  # type: ignore[no-untyped-def]
    from PySide6.QtWidgets import QApplication, QMessageBox

    from app.ai.model_config import ModelConfig
    from app.ui.content_analysis import COL_TITLE, COL_TRANSCRIPT
    from app.ui.main_window import MainWindow
    from app.ui.preview_window import COL_NEW
    from tests.fake_ai_server import FakeAIServer, openai_ok
    from tests.helpers import wait_until

    app = QApplication.instance() or QApplication([])
    QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok)  # type: ignore[method-assign]
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)  # type: ignore[method-assign]
    MainWindow._ask = lambda *a, **k: True  # type: ignore[method-assign]  # confirm Apply
    folder = work / "Videos"
    folder.mkdir()
    shutil.copy(JFK, folder / "clip_0001.mp4")
    ctx.settings.update(transcription_service="auto", whisper_model=model)
    good = {"title": "Ask Not What Your Country Can Do For You", "evidence": "ask not what your country can do for you", "names": []}
    second = {"title": "My Fellow Americans, Ask What You Can Do For Your Country", "evidence": "my fellow americans", "names": []}
    invented = {"title": "President Lincoln Promises Free Healthcare For Everyone", "evidence": "free healthcare for everyone", "names": ["Lincoln"]}
    invented_name = {"title": "Lincoln Asks What Your Country Can Do", "evidence": "ask not what your country", "names": ["Lincoln"]}
    answer = {"titles": [{"n": 1, "analysis": {"topic": "a call to public service", "key_points": ["ask not what your country can do"]},
                          "suggestions": [good, invented, invented_name, second]}]}  # fmt: skip
    with FakeAIServer() as chat:
        ctx.registry.add(ModelConfig.new("openai_compatible", "stand-in-ai", base_url=chat.base, max_retries=0))
        window = MainWindow(ctx)
        window.files_panel.set_filter("all")
        window.show()
        window.refresh_models_ui()
        window.load_folder(folder)
        wait_until(lambda: window.files_panel.model.rowCount() == 1, timeout=20)
        panel = window.title_panel
        panel.enabled.setChecked(True)
        panel.source.setCurrentIndex(panel.source.findData("transcript"))
        content = panel.content
        window.analyze_videos()
        wait_until(lambda: window._analysis_worker is None, timeout=900)
        row = content.results.topLevelItem(0)
        lines = [row.child(i).text(0) for i in range(row.childCount())]
        check("GUI: transcript shown after Analyze Selected Videos", row.text(COL_TRANSCRIPT).startswith("Completed")
              and any(transcript[:30] in line for line in lines), f"{row.text(COL_TRANSCRIPT)} | {lines[:1]}")  # fmt: skip
        for _ in range(3):
            chat.queue(200, openai_ok(json.dumps(answer)))
        window.command_panel.set_command("")
        window.generate_preview()
        wait_until(lambda: window._plan_worker is None and not window.command_panel._busy, timeout=300)
        prompt = json.loads(chat.requests[0]["body"])["messages"][1]["content"] if chat.requests else ""
        check("title generation received the real transcript", transcript[:40] in prompt and "clip_0001" not in prompt,
              prompt[prompt.find('"transcript"'):][:160])  # fmt: skip
        model_rows = window.preview_panel.model
        new_name = model_rows.index(0, COL_NEW).data() if model_rows.rowCount() else ""
        check("preview uses a transcript-supported title", new_name == f"{good['title']}.mp4", new_name)
        content.results.expandAll()
        texts = []
        for i in range(row.childCount()):
            texts += [row.child(i).text(0)] + [row.child(i).child(j).text(0) for j in range(row.child(i).childCount())]
        rejected = [t for t in texts if "Lincoln" in t and " — " in t]
        check("invented claims rejected (made-up quote, made-up name)", len(rejected) == 2
              and any("'Lincoln' is not mentioned in the transcript" in t for t in rejected), " | ".join(rejected))  # fmt: skip
        if not new_name:
            check("apply / undo with a chosen suggestion", False, "skipped: there is no preview row")
            window.close()
            return
        window.use_suggestion("clip_0001.mp4", second["title"])
        new_name = model_rows.index(0, COL_NEW).data()
        check("choosing another suggestion updates the preview", new_name == f"{second['title'].replace(',', ',')}.mp4"
              and row.text(COL_TITLE) == second["title"], new_name)  # fmt: skip
        window.apply_changes()
        wait_until(lambda: not window._operation_running and (folder / new_name).exists(), timeout=30)
        wait_until(lambda: not window._operation_running, timeout=30)
        check("apply renamed the file", (folder / new_name).exists() and not (folder / "clip_0001.mp4").exists())
        window.undo_last()
        wait_until(lambda: (folder / "clip_0001.mp4").exists() and not window._operation_running, timeout=30)
        check("undo restored the original name", (folder / "clip_0001.mp4").exists() and not (folder / new_name).exists())
        window.close()
        app.processEvents()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="small")
    parser.add_argument("--bangla-model", default="", help="e.g. large-v3-turbo (empty: skip the Bangla check)")
    parser.add_argument("--report", default="")
    parser.add_argument("--data-dir", default="", help="app data folder (keeps downloaded models between runs)")
    args = parser.parse_args()
    work = Path(tempfile.mkdtemp(prefix="aivfa-whisper-"))
    os.environ["AIVFA_DATA_DIR"] = args.data_dir or str(work / "data")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from app.context import AppContext

    _app = QApplication.instance() or QApplication([])
    ctx = AppContext.create()
    transcript = ""
    try:
        transcript = pipeline_checks(ctx, work, args.model)
        if args.bangla_model:
            bangla_check(ctx, args.bangla_model)
        gui_checks(ctx, work, transcript, args.model)
    except Exception as exc:  # noqa: BLE001 - report, then fail
        import traceback

        traceback.print_exc()
        check("verification ran to the end", False, f"{type(exc).__name__}: {exc}")
    failed = [name for name, ok, _ in results if not ok]
    report = {"passed": not failed, "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in results], "transcript": transcript}
    if args.report:
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed" + (f"; failed: {failed}" if failed else ""), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
