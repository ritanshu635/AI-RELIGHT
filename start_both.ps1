# Spooky Relight + Flask App Launcher
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting Spooky Relight + Flask App" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Start Flask Backend
Write-Host "Starting Flask Backend (Port 5000)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd light; D:\new1\venv_py310\Scripts\python.exe app.py"

# Wait a moment
Start-Sleep -Seconds 3

# Start Spooky Landing Page
Write-Host "Starting Spooky Landing Page (Port 5173)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd spooky-relight-master-5c3ffe48-main; npm run dev"

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Both servers are starting!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Flask Backend: " -NoNewline
Write-Host "http://localhost:5000" -ForegroundColor Cyan
Write-Host "Spooky Landing: " -NoNewline
Write-Host "http://localhost:5173" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
