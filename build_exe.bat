@echo off
echo ============================================
echo  Media Analyzer ^& Organizer - Build .exe
echo ============================================
echo.

echo [1/3] Installing dependencies...
pip install customtkinter Pillow pyinstaller
echo.

echo [2/3] Building executable (this takes ~60s)...
pyinstaller ^
  --onefile ^
  --windowed ^
  --name "MediaAnalyzer" ^
  --collect-all customtkinter ^
  media_analyzer_app.py

echo.
echo [3/3] Done!
echo.
echo  Executable : dist\MediaAnalyzer.exe
echo  Double-click it — no Python needed.
echo.
pause
