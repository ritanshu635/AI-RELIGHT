# IC-Light Fix and Run Script
# This script fixes all issues and runs the application

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "IC-Light Fix and Run Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Activate virtual environment
Write-Host "[1/5] Activating virtual environment..." -ForegroundColor Yellow
$venvPath = "..\venv_py310\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    & $venvPath
    Write-Host "✓ Virtual environment activated" -ForegroundColor Green
} else {
    Write-Host "✗ Virtual environment not found at $venvPath" -ForegroundColor Red
    exit 1
}

# Step 2: Verify dependencies
Write-Host ""
Write-Host "[2/5] Verifying dependencies..." -ForegroundColor Yellow
python -c "import torch, gradio, openai, realesrgan; print('✓ All dependencies installed')" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Missing dependencies. Installing..." -ForegroundColor Red
    pip install -r requirements.txt
}

# Step 3: Check environment variables
Write-Host ""
Write-Host "[3/5] Checking environment variables..." -ForegroundColor Yellow
if ($env:OPENAI_API_KEY) {
    Write-Host "✓ OPENAI_API_KEY is set" -ForegroundColor Green
} else {
    Write-Host "⚠ OPENAI_API_KEY not set. AI recommendations will use defaults." -ForegroundColor Yellow
    Write-Host "  To set it: `$env:OPENAI_API_KEY='your-key-here'" -ForegroundColor Gray
}

# Step 4: Check model files
Write-Host ""
Write-Host "[4/5] Checking model files..." -ForegroundColor Yellow
if (Test-Path "models\RealESRGAN_x4plus.pth") {
    Write-Host "✓ RealESRGAN model found" -ForegroundColor Green
} else {
    Write-Host "⚠ RealESRGAN model not found. Upscaling will be disabled." -ForegroundColor Yellow
    Write-Host "  Download from: https://github.com/xinntao/Real-ESRGAN/releases" -ForegroundColor Gray
}

# Step 5: Create outputs directory
Write-Host ""
Write-Host "[5/5] Creating outputs directory..." -ForegroundColor Yellow
if (!(Test-Path "outputs")) {
    New-Item -ItemType Directory -Path "outputs" | Out-Null
}
Write-Host "✓ Outputs directory ready" -ForegroundColor Green

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting IC-Light Application..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Choose which demo to run:" -ForegroundColor Yellow
Write-Host "1. Text-conditioned relighting (gradio_demo.py)" -ForegroundColor White
Write-Host "2. Background-conditioned relighting (gradio_demo_bg.py)" -ForegroundColor White
Write-Host ""
$choice = Read-Host "Enter choice (1 or 2)"

if ($choice -eq "1") {
    Write-Host ""
    Write-Host "Starting text-conditioned demo..." -ForegroundColor Green
    python gradio_demo.py
} elseif ($choice -eq "2") {
    Write-Host ""
    Write-Host "Starting background-conditioned demo..." -ForegroundColor Green
    python gradio_demo_bg.py
} else {
    Write-Host "Invalid choice. Defaulting to text-conditioned demo..." -ForegroundColor Yellow
    python gradio_demo.py
}
