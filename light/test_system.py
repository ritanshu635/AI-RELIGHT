"""
Comprehensive system test for IC-Light
Tests all components and their integration
"""
import os
import sys
import numpy as np
from PIL import Image

print("=" * 60)
print("IC-Light System Test")
print("=" * 60)

# Test 1: Python environment
print("\n1. Testing Python environment...")
print(f"   Python: {sys.version}")
print(f"   Executable: {sys.executable}")

# Test 2: Core dependencies
print("\n2. Testing core dependencies...")
try:
    import torch
    print(f"   ✓ PyTorch: {torch.__version__}")
    print(f"   ✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   ✓ CUDA device: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"   ✗ PyTorch error: {e}")
    sys.exit(1)

try:
    import gradio as gr
    print(f"   ✓ Gradio: {gr.__version__}")
except Exception as e:
    print(f"   ✗ Gradio error: {e}")
    sys.exit(1)

try:
    import diffusers
    print(f"   ✓ Diffusers: {diffusers.__version__}")
except Exception as e:
    print(f"   ✗ Diffusers error: {e}")
    sys.exit(1)

# Test 3: Custom modules
print("\n3. Testing custom modules...")
try:
    from upscaler import ImageUpscaler
    print("   ✓ upscaler.py imported")
except Exception as e:
    print(f"   ✗ upscaler.py error: {e}")
    sys.exit(1)

try:
    from gpt_recommendations import GPTRecommendationClient
    print("   ✓ gpt_recommendations.py imported")
except Exception as e:
    print(f"   ✗ gpt_recommendations.py error: {e}")
    sys.exit(1)

try:
    from briarmbg import BriaRMBG
    print("   ✓ briarmbg.py imported")
except Exception as e:
    print(f"   ✗ briarmbg.py error: {e}")
    sys.exit(1)

# Test 4: Model files
print("\n4. Checking model files...")
models_dir = "./models"
if not os.path.exists(models_dir):
    print(f"   ! Creating models directory: {models_dir}")
    os.makedirs(models_dir)

model_files = {
    "iclight_sd15_fc.safetensors": "Text-conditioned model",
    "iclight_sd15_fbc.safetensors": "Background-conditioned model",
    "RealESRGAN_x4plus.pth": "Upscaling model"
}

for model_file, description in model_files.items():
    model_path = os.path.join(models_dir, model_file)
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"   ✓ {description}: {model_file} ({size_mb:.1f} MB)")
    else:
        print(f"   ✗ {description}: {model_file} NOT FOUND")
        if "RealESRGAN" in model_file:
            print(f"      Download from: https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth")
        else:
            print(f"      Will be auto-downloaded on first run")

# Test 5: Environment variables
print("\n5. Checking environment variables...")
openai_key = os.getenv('OPENAI_API_KEY')
if openai_key:
    masked_key = openai_key[:10] + "..." + openai_key[-4:] if len(openai_key) > 14 else "***"
    print(f"   ✓ OPENAI_API_KEY: {masked_key}")
else:
    print("   ! OPENAI_API_KEY not set (AI recommendations will use defaults)")

# Test 6: Upscaler initialization
print("\n6. Testing upscaler initialization...")
try:
    upscaler = ImageUpscaler(model_path='./models/RealESRGAN_x4plus.pth', device='cuda')
    if upscaler.model_available:
        print("   ✓ Upscaler initialized successfully")
        
        # Test with a small image
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = upscaler.upscale_to_1080p(test_img)
        print(f"   ✓ Test upscale: {test_img.shape} -> {result.shape}")
    else:
        print("   ! Upscaler model not available (will use simple upscaling)")
except Exception as e:
    print(f"   ✗ Upscaler error: {e}")

# Test 7: GPT client initialization
print("\n7. Testing GPT client initialization...")
try:
    gpt_client = GPTRecommendationClient()
    if gpt_client.client:
        print("   ✓ GPT client initialized with API key")
    else:
        print("   ! GPT client initialized without API key (will use defaults)")
    
    # Test with a dummy image
    test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    suggestions = gpt_client.get_lighting_recommendations(test_img)
    print(f"   ✓ Test recommendations: {suggestions}")
except Exception as e:
    print(f"   ✗ GPT client error: {e}")

# Test 8: Output directory
print("\n8. Checking output directory...")
output_dir = "./outputs"
if not os.path.exists(output_dir):
    print(f"   ! Creating output directory: {output_dir}")
    os.makedirs(output_dir)
else:
    print(f"   ✓ Output directory exists: {output_dir}")

# Test 9: Static files
print("\n9. Checking static files...")
static_files = ["static/light_pointer.js"]
for static_file in static_files:
    if os.path.exists(static_file):
        print(f"   ✓ {static_file}")
    else:
        print(f"   ✗ {static_file} NOT FOUND")

# Summary
print("\n" + "=" * 60)
print("System Test Complete!")
print("=" * 60)
print("\nTo run the application:")
print("  Text-conditioned demo:       python gradio_demo.py")
print("  Background-conditioned demo: python gradio_demo_bg.py")
print("\nThe application will be available at: http://localhost:7860")
print("=" * 60)
