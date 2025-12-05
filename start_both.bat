@echo off
echo ========================================
echo Starting Spooky Relight + Flask App
echo ========================================
echo.

echo Starting Flask Backend (Port 5000)...
start "Flask Backend" cmd /k "cd light && D:\new1\venv_py310\Scripts\python.exe app.py"

timeout /t 3 /nobreak >nul

echo Starting Spooky Landing Page (Port 5173)...
start "Spooky Landing" cmd /k "cd spooky-relight-master-5c3ffe48-main && npm run dev"

echo.
echo ========================================
echo Both servers are starting!
echo ========================================
echo.
echo Flask Backend: http://localhost:5000
echo Spooky Landing: http://localhost:5173
echo.
echo Press any key to exit this window...
pause >nul
