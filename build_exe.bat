@echo off
echo ============================================
echo  Media Analyzer - Build Windows .exe
echo ============================================

REM Install deps if needed
echo [1/3] Installing dependencies...
pip install customtkinter Pillow pyinstaller

REM Build the exe
echo [2/3] Building executable...
pyinstaller ^
  --onefile ^
  --windowed ^
  --name "MediaAnalyzer" ^
  --icon NONE ^
  --collect-all customtkinter ^
  media_analyzer_app.py

echo [3/3] Done!
echo.
echo Executable: dist\MediaAnalyzer.exe
echo.
pause
