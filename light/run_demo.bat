@echo off
echo ============================================================
echo IC-Light Demo Launcher
echo ============================================================
echo.
echo Select which demo to run:
echo   1. Text-Conditioned Relighting (gradio_demo.py)
echo   2. Background-Conditioned Relighting (gradio_demo_bg.py)
echo   3. Run System Test
echo   4. Exit
echo.
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo.
    echo Starting Text-Conditioned Demo...
    echo.
    D:\new1\venv_py310\Scripts\python.exe gradio_demo.py
) else if "%choice%"=="2" (
    echo.
    echo Starting Background-Conditioned Demo...
    echo.
    D:\new1\venv_py310\Scripts\python.exe gradio_demo_bg.py
) else if "%choice%"=="3" (
    echo.
    echo Running System Test...
    echo.
    D:\new1\venv_py310\Scripts\python.exe test_system.py
    echo.
    pause
) else if "%choice%"=="4" (
    echo.
    echo Exiting...
    exit /b 0
) else (
    echo.
    echo Invalid choice. Please run the script again.
    pause
)
