Optional: place ffmpeg.exe and ffprobe.exe in this folder BEFORE running build.bat to ship them
inside the application (one-folder build). They enable durations, resolution/codec details and
thumbnails. Download a Windows build from https://ffmpeg.org/download.html (e.g. the "essentials"
build from gyan.dev) - mind its licence if you redistribute the result.

If this folder is empty the app still works: it uses an ffprobe/ffmpeg found on PATH, or the
folder configured in Settings -> General, and otherwise simply hides those features.
