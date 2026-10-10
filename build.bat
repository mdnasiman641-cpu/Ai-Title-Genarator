@echo off
rem ===========================================================================
rem  AI Video File Assistant - build script (Windows)
rem
rem    build.bat            one-folder build  -> dist\AI Video File Assistant\AI Video File Assistant.exe
rem    build.bat onefile    single-file build -> dist\AI Video File Assistant.exe
rem
rem  Steps: create/activate .venv -> install dependencies -> run tests -> PyInstaller
rem         -> self-test of the finished exe.
rem  Set SKIP_TESTS=1 to skip the test run (not recommended).
rem ===========================================================================
setlocal EnableExtensions
cd /d "%~dp0"

set "MODE=%~1"
set "APP_NAME=AI Video File Assistant"

echo.
echo [1/5] Looking for Python 3.12 or newer...
set "PY="
py -3.12 --version >nul 2>&1 && set "PY=py -3.12"
if not defined PY py -3 --version >nul 2>&1 && set "PY=py -3"
if not defined PY python --version >nul 2>&1 && set "PY=python"
if not defined PY goto :nopython
%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)"
if errorlevel 1 goto :nopython

echo [2/5] Preparing the virtual environment (.venv)...
if not exist ".venv\Scripts\python.exe" (
    %PY% -m venv .venv
    if errorlevel 1 goto :fail
)
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip --quiet
if errorlevel 1 goto :fail
python -m pip install -r requirements-dev.txt --quiet
if errorlevel 1 goto :fail

echo [3/5] Running the tests...
if "%SKIP_TESTS%"=="1" (
    echo       skipped because SKIP_TESTS=1
) else (
    set "QT_QPA_PLATFORM=offscreen"
    python -m pytest -q
    if errorlevel 1 (
        set "QT_QPA_PLATFORM="
        echo.
        echo *** Tests failed - the executable was NOT built. ***
        goto :fail
    )
    set "QT_QPA_PLATFORM="
)

echo [4/5] Building the executable with PyInstaller...
if /i "%MODE%"=="onefile" (set "AIVFA_ONEFILE=1") else (set "AIVFA_ONEFILE=")
if not exist "app\resources\bin\ffprobe.exe" echo       note: no ffprobe.exe in app\resources\bin - video durations/thumbnails need ffmpeg on PATH.
python -m PyInstaller --noconfirm --clean "%APP_NAME%.spec"
if errorlevel 1 goto :fail

if /i "%MODE%"=="onefile" (
    set "EXE=dist\%APP_NAME%.exe"
) else (
    set "EXE=dist\%APP_NAME%\%APP_NAME%.exe"
)
if not exist "%EXE%" goto :fail

echo [5/5] Self-testing the built executable...
start "" /wait "%EXE%" --self-test
if errorlevel 1 (
    echo *** The built executable failed its self-test. ***
    goto :fail
)

echo.
echo ===========================================================================
echo  Build finished:  %EXE%
echo ===========================================================================
endlocal
exit /b 0

:nopython
echo.
echo *** Python 3.12 or newer was not found. Install it from https://www.python.org/downloads/
echo     (tick "Add python.exe to PATH" and keep the "py launcher" option), then run build.bat again.
goto :fail

:fail
echo.
echo Build failed.
endlocal
exit /b 1
