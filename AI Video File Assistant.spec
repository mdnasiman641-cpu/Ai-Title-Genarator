# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build recipe.

Default: a **one-folder** build  ->  dist/AI Video File Assistant/AI Video File Assistant.exe
Optional: ``set AIVFA_ONEFILE=1`` (or ``build.bat onefile``) -> dist/AI Video File Assistant.exe

One-folder is the default because a one-file exe unpacks itself to a temp folder on every
launch (slow start, more antivirus false positives) and would have to carry ffmpeg/ffprobe
(~100 MB) inside. See README.md ("Packaging trade-offs").
"""

import os
import sys
from pathlib import Path

ROOT = Path(SPECPATH)  # noqa: F821 - provided by PyInstaller
NAME = "AI Video File Assistant"
ONEFILE = os.environ.get("AIVFA_ONEFILE") == "1"
ON_WINDOWS = sys.platform == "win32"

# Bundled resources keep their ``app/resources/...`` layout so ``resource_path()`` works
# identically from source and from the frozen app.
datas = [
    (str(ROOT / "app" / "resources" / "icons"), "app/resources/icons"),
    (str(ROOT / "app" / "resources" / "styles"), "app/resources/styles"),
]
bin_dir = ROOT / "app" / "resources" / "bin"  # optional: drop ffmpeg.exe / ffprobe.exe here
if not ONEFILE and bin_dir.is_dir() and any(p.suffix.lower() in (".exe", ".dll") for p in bin_dir.iterdir()):
    datas.append((str(bin_dir), "app/resources/bin"))

icon_file = ROOT / "app" / "resources" / "icons" / "app.ico"
version_file = ROOT / "version_info.txt"
exe_options = dict(
    name=NAME,
    debug=False,
    strip=False,
    upx=False,  # UPX-packed exes trigger far more antivirus false positives
    console=False,  # GUI application: no console window
    icon=str(icon_file) if ON_WINDOWS and icon_file.exists() else None,
    version=str(version_file) if ON_WINDOWS and version_file.exists() else None,
)

a = Analysis(  # noqa: F821
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    datas=datas,
    hiddenimports=[],
    excludes=["tkinter", "unittest", "pytest", "_pytest", "pip", "setuptools", "wheel", "distutils", "pydoc_data"],
    noarchive=False,
)
pyz = PYZ(a.pure)  # noqa: F821

if ONEFILE:
    exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], **exe_options)  # noqa: F821
else:
    exe = EXE(pyz, a.scripts, [], exclude_binaries=True, **exe_options)  # noqa: F821
    coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name=NAME)  # noqa: F821
