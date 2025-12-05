# IC-Light Enhancement Setup Guide

## Prerequisites

- Python 3.10
- CUDA 12.1 compatible GPU
- Minimum 8GB VRAM (12GB+ recommended)

## Installation Steps

### 1. Activate Virtual Environment

```bash
D:\new1\venv_py310\Scripts\Activate.ps1
```

### 2. Install Dependencies

All required dependencies have been installed:
- ✅ openai (v2.8.1)
- ✅ basicsr (v1.4.2)
- ✅ realesrgan (v0.3.0)
- ✅ facexlib (v0.3.0)
- ✅ gfpgan (v1.3.8)

### 3. Verify Model Files

The Real-ESRGAN model is located at:
```
D:\new1\light\models\RealESRGAN_x4plus.pth
```

### 4. Configure OpenAI API Key

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

3. Get your API key from: https://platform.openai.com/api-keys

### 5. Set Environment Variable (Alternative)

If not using .env file, set the environment variable directly:

**PowerShell:**
```powershell
$env:OPENAI_API_KEY="sk-your-actual-api-key-here"
```

**CMD:**
```cmd
set OPENAI_API_KEY=sk-your-actual-api-key-here
```

## Verification

To verify the installation, run:

```bash
python -c "import openai, basicsr, realesrgan; print('All packages imported successfully!')"
```

## Next Steps

Once setup is complete, you can proceed with implementing the remaining tasks:
- Task 2: Implement Real-ESRGAN upscaling module
- Task 3: Implement GPT Vision API client
- Task 4: Update gradio_demo.py with modern UI
- And more...

## Troubleshooting

### SSL Certificate Issues

If you encounter SSL certificate errors during pip install, use:
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org <package-name>
```

### GPU Memory Issues

If you run out of GPU memory:
- Close other GPU-intensive applications
- Reduce batch size or image resolution
- Consider using CPU mode (slower but works with limited VRAM)
