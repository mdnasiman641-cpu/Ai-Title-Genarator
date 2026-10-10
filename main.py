"""AI Video File Assistant - application entry point.

Run normally to start the app. ``--self-test`` runs a headless end-to-end check
(resources, translations, theme, database, scan, plan, apply, undo) in a
temporary folder and exits with status 0 on success; ``build.bat`` uses it to
verify the packaged executable.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


def self_test() -> int:
    """Exercise the real code paths against a throw-away folder. Returns a process exit code."""
    work = Path(tempfile.mkdtemp(prefix="aivfa-selftest-"))
    os.environ["AIVFA_DATA_DIR"] = str(work / "data")  # never touch the real user profile
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # no display needed (CI, services, SSH)
    report: list[str] = []
    try:
        from PySide6.QtWidgets import QApplication

        from app.ai.schemas import NumberingAction
        from app.config.constants import APP_NAME, APP_VERSION
        from app.context import AppContext
        from app.files.operation_manager import OperationManager
        from app.files.planner import Planner
        from app.files.scanner import scan_folder
        from app.i18n import all_keys, set_language, tr
        from app.ui.icons import get_icon
        from app.ui.main_window import MainWindow
        from app.ui.theme import apply_theme
        from app.utils.helpers import resource_path

        app = QApplication.instance() or QApplication(["self-test"])
        for theme in ("dark", "light"):
            apply_theme(app, theme)
        report.append(f"{APP_NAME} {APP_VERSION}: themes OK")
        assert not get_icon("folder", "#ffffff", 16).isNull(), "icons were not bundled"
        assert resource_path("icons", "app.png").exists(), "application icon missing"
        set_language("bn")
        assert tr("folder.title") == "ফোল্ডার" and len(all_keys()) > 100
        set_language("en")
        report.append("resources + translations OK")

        folder = work / "Videos"
        folder.mkdir()
        for i in (1, 2, 3):
            (folder / f"VID_00{i}_FINAL_1080P_WEB-DL.mp4").write_text(str(i))
        ctx = AppContext.create(work / "data")
        entries = scan_folder(folder).entries
        plan = Planner(folder, entries).build(
            [NumberingAction(start=1, width=2, position="replace", template="Blue Bloods - Episode {n}")]
        )
        manager = OperationManager(ctx.db)
        result = manager.execute(plan, command="self-test", provider="offline")
        expected = {f"Blue Bloods - Episode 0{i}.mp4" for i in (1, 2, 3)}
        assert result.ok and {p.name for p in folder.iterdir()} == expected, f"rename failed: {result.message}"
        assert manager.undo(result.operation_id or 0).ok, "undo failed"
        assert len(list(folder.iterdir())) == 3 and (folder / "VID_001_FINAL_1080P_WEB-DL.mp4").exists()
        report.append("scan → plan → apply → undo OK")

        window = MainWindow(ctx)  # builds every panel (not shown)
        window.close()
        report.append("main window OK")
    except Exception as exc:  # noqa: BLE001 - report any failure as a non-zero exit
        report.append(f"SELF-TEST FAILED: {type(exc).__name__}: {exc}")
        (work / "selftest.txt").write_text("\n".join(report), encoding="utf-8")
        print("\n".join(report), file=sys.stderr)
        return 1
    report.append("SELF-TEST OK")
    (work / "selftest.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Create the Qt application, apply settings and show the main window."""
    argv = sys.argv if argv is None else argv
    if "--self-test" in argv:
        return self_test()

    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    from app.config.constants import APP_NAME, APP_ORGANIZATION
    from app.context import AppContext
    from app.i18n import set_language
    from app.ui.error_hook import install_exception_hooks
    from app.ui.gc_guard import MainThreadGC
    from app.ui.main_window import MainWindow
    from app.ui.theme import apply_theme
    from app.utils.helpers import resource_path
    from app.utils.logger import get_logger, setup_logging

    app = QApplication(argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_ORGANIZATION)
    app.setQuitOnLastWindowClosed(True)
    gc_guard = MainThreadGC(app)  # never let Python's GC destroy Qt objects from a worker thread
    icon = resource_path("icons", "app.png")
    if icon.exists():
        app.setWindowIcon(QIcon(str(icon)))

    ctx = AppContext.create()
    settings = ctx.settings.load()
    setup_logging(settings.log_level)
    install_exception_hooks()
    set_language(settings.language)
    apply_theme(app, settings.theme)
    get_logger("app").info("Startup complete (theme=%s, language=%s)", settings.theme, settings.language)

    window = MainWindow(ctx)
    window.show()
    window.raise_()
    window.activateWindow()
    code = app.exec()
    gc_guard.stop()
    return code


if __name__ == "__main__":
    sys.exit(main())
