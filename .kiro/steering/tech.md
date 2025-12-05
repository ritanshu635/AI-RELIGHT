# Technology Stack

## Core Framework

- **Python 3.10**: Primary language
- **PyTorch**: Deep learning framework with CUDA 12.1 support
- **Gradio 3.41.2**: Web UI framework
- **Diffusers 0.27.2**: Stable Diffusion pipeline management
- **Transformers 4.36.2**: CLIP text encoding

## AI Models

- **Stable Diffusion 1.5**: Base model (stablediffusionapi/realistic-vision-v51)
- **IC-Light Models**: Custom relighting models
  - `iclight_sd15_fc.safetensors` - Text + foreground conditioned
  - `iclight_sd15_fcon.safetensors` - With offset noise
  - `iclight_sd15_fbc.safetensors` - Text + foreground + background conditioned
- **BRIA RMBG 1.4**: Background removal
- **Real-ESRGAN x4plus**: Image upscaling
- **GPT-4o-mini**: AI lighting recommendations via OpenAI API

## Key Dependencies

```
diffusers==0.27.2
transformers==4.36.2
torch (CUDA 12.1)
gradio==3.41.2
openai>=1.0.0
realesrgan>=0.3.0
basicsr>=1.4.2
opencv-python
safetensors
pillow==10.2.0
```

## Environment Setup

### Virtual Environment
```bash
# Activate environment
D:\new1\venv_py310\Scripts\Activate.ps1
```

### Required Environment Variables
```bash
# OpenAI API key for AI recommendations
set OPENAI_API_KEY=your-api-key-here          # Windows CMD
$env:OPENAI_API_KEY="your-api-key-here"       # PowerShell
```

### Model Files
Models are auto-downloaded to `./models/` except:
- `RealESRGAN_x4plus.pth` must be manually placed in `./models/`

## Common Commands

### Run Applications
```bash
# Text-conditioned relighting demo
python gradio_demo.py

# Background-conditioned demo
python gradio_demo_bg.py
```

### Testing
```bash
# Run specific test file
python test_upscaler.py
python test_gpt_recommendations.py

# Run all tests
python -m unittest discover -s . -p "test_*.py"

# Integration validation
python validate_integration.py
```

### Development
```bash
# Install dependencies
pip install -r requirements.txt

# Check imports
python -c "import openai, basicsr, realesrgan; print('OK')"
```

## GPU Requirements

- **Minimum**: 8GB VRAM
- **Recommended**: 12GB+ VRAM
- **Models Memory Usage**:
  - IC-Light: ~4GB
  - Real-ESRGAN: ~2GB
  - Total pipeline: ~6-8GB

## Performance Notes

- Processing time: ~20-30 seconds per image (full pipeline)
- AI recommendations: ~2-5 seconds (API call)
- Upscaling: ~5-10 seconds
- CPU fallback available but significantly slower
