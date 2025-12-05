@echo off
echo ========================================
echo Setting up Spooky Landing Page
echo ========================================
echo.

cd spooky-relight-master-5c3ffe48-main

echo Installing npm dependencies...
echo This may take a few minutes...
echo.

npm install

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo You can now run: start_both.bat
echo Or manually: npm run dev
echo.
pause
