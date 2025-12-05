# 🚀 IC-Light Complete Tech Stack

## 📋 Overview
IC-Light is an AI-powered image relighting application built with Python, deep learning models, and modern web technologies.

---

## 🎨 Frontend Technologies

### **Web Framework**
- **Flask 2.x** - Python web framework for the main application
  - Lightweight and flexible
  - RESTful API endpoints
  - File upload handling
  - Session management

### **UI/UX**
- **HTML5** - Modern semantic markup
- **CSS3** - Custom styling with:
  - CSS Grid & Flexbox layouts
  - Gradient backgrounds
  - Animations (pulse, glow effects)
  - Dark theme design
  - Responsive design
- **Vanilla JavaScript** - Interactive features:
  - Drag-and-drop file upload
  - Draggable light source pointer
  - Real-time direction detection
  - AJAX API calls
  - Progress indicators
  - Touch event support (mobile)

### **Alternative Frontend (Legacy)**
- **Gradio 3.41.2** - Original UI framework
  - Python-based web UI
  - Built-in components
  - Auto-generated interface
  - (Note: Replaced with Flask due to timeout issues)

---

## 🧠 AI/ML Models & Frameworks

### **Core Deep Learning Framework**
- **PyTorch 2.x** - Primary deep learning framework
  - CUDA 12.1 support for GPU acceleration
  - Tensor operations
  - Automatic differentiation
  - Model inference

### **Diffusion Models**
- **Diffusers 0.27.2** - Hugging Face diffusion models library
  - `StableDiffusionPipeline` - Text-to-image generation
  - `StableDiffusionImg2ImgPipeline` - Image-to-image transformation
  - `AutoencoderKL` - VAE for latent space encoding/decoding
  - `UNet2DConditionModel` - Core diffusion model architecture
  - Custom attention processors (`AttnProcessor2_0`)

### **Schedulers/Samplers**
- **DDIMScheduler** - Denoising Diffusion Implicit Models
- **EulerAncestralDiscreteScheduler** - Euler A sampler
- **DPMSolverMultistepScheduler** - DPM++ 2M SDE Karras (default)
  - Faster convergence
  - Better quality
  - Karras noise schedule

### **Text Encoding**
- **Transformers 4.36.2** - Hugging Face transformers library
  - `CLIPTextModel` - Text encoder for prompts
  - `CLIPTokenizer` - Text tokenization
  - Handles long prompts with chunking

### **Specialized Models**

#### **1. IC-Light Models**
Custom relighting models from lllyasviel:
- **iclight_sd15_fc.safetensors** - Text + foreground conditioned
  - Main model for text-based relighting
  - 8-channel input (4 RGB + 4 foreground condition)
  - Based on Stable Diffusion 1.5
- **iclight_sd15_fcon.safetensors** - With offset noise
- **iclight_sd15_fbc.safetensors** - Text + foreground + background conditioned

#### **2. Base Stable Diffusion Model**
- **Realistic Vision v5.1** (`stablediffusionapi/realistic-vision-v51`)
  - Fine-tuned SD 1.5 for photorealistic images
  - Better quality than base SD 1.5
  - Components:
    - VAE (Variational Autoencoder)
    - UNet (Denoising network)
    - Text Encoder (CLIP)
    - Tokenizer

#### **3. Background Removal**
- **BRIA RMBG 1.4** (`briaai/RMBG-1.4`)
  - State-of-the-art background removal
  - Produces alpha matte
  - GPU-accelerated
  - Note: Non-commercial license (replace with BiRefNet for commercial use)

#### **4. Image Upscaling**
- **Real-ESRGAN x4plus**
  - 4x super-resolution
  - Trained on real-world degradations
  - Upscales to minimum 1080p
  - Model file: `RealESRGAN_x4plus.pth`
  - Dependencies:
    - `realesrgan>=0.3.0`
    - `basicsr>=1.4.2` - Basic image processing
    - `facexlib>=0.3.0` - Face enhancement
    - `gfpgan>=1.3.8` - Face restoration

#### **5. AI Recommendations**
- **GPT-4o-mini** (OpenAI API)
  - Vision-language model
  - Analyzes uploaded images
  - Generates contextual lighting suggestions
  - Supports mood-based recommendations
  - Returns 3 suggestions (max 3 words each)

---

## 🛠️ Backend Technologies

### **Python Version**
- **Python 3.10** - Core language

### **Web Server**
- **Flask Development Server** - For development
- **Werkzeug** - WSGI utility library (included with Flask)
  - File upload handling
  - Secure filename generation

### **Image Processing**
- **Pillow 10.2.0** (PIL) - Python Imaging Library
  - Image loading/saving
  - Format conversion
  - Resizing and cropping
  - LANCZOS resampling
- **OpenCV (opencv-python)** - Computer vision library
  - Advanced image operations
  - Color space conversions
- **NumPy** - Numerical computing
  - Array operations
  - Image data manipulation
  - Gradient generation

### **Model Loading & Serialization**
- **SafeTensors** - Secure tensor serialization
  - Fast loading
  - Memory-efficient
  - Safer than pickle
- **Protobuf 3.20** - Protocol buffers for model serialization

### **Additional Libraries**
- **einops** - Tensor operations with readable notation
- **peft** - Parameter-Efficient Fine-Tuning
- **certifi** - SSL certificate bundle
  - Fixes SSL verification issues
  - Ensures secure HTTPS connections

---

## 🔧 Development Tools

### **Environment Management**
- **Virtual Environment** (`venv_py310`)
  - Isolated Python environment
  - Dependency management
  - Located at: `D:\new1\venv_py310`

### **Package Management**
- **pip** - Python package installer
- **requirements.txt** - Dependency specification

### **Version Control**
- **Git** - Source control
- **.gitignore** - Excludes models, outputs, venv, cache

---

## 🌐 API Integration

### **OpenAI API**
- **Endpoint**: GPT-4o-mini vision model
- **Authentication**: API key via environment variable
- **Features**:
  - Image analysis
  - Natural language generation
  - Mood-based recommendations
- **Configuration**: `.env` file
  ```
  OPENAI_API_KEY=your-api-key-here
  ```

---

## 💾 Data Storage

### **File System**
- **Uploads Folder** (`./uploads/`) - Temporary uploaded images
- **Outputs Folder** (`./outputs/`) - Generated relit images
- **Models Folder** (`./models/`) - AI model weights
  - IC-Light models (auto-downloaded)
  - Real-ESRGAN model (manual placement required)
  - BRIA RMBG (auto-downloaded)

### **File Formats**
- **Input**: JPG, PNG (max 50MB)
- **Output**: PNG (high quality, lossless)
- **Models**: `.safetensors`, `.pth`

---

## ⚙️ Hardware Requirements

### **GPU**
- **CUDA 12.1** - NVIDIA GPU acceleration
- **Minimum VRAM**: 8GB
- **Recommended VRAM**: 12GB+
- **Device**: `torch.device('cuda')`

### **Memory Usage**
- IC-Light model: ~4GB VRAM
- Real-ESRGAN: ~2GB VRAM
- Total pipeline: ~6-8GB VRAM
- CPU fallback available (slower)

### **Precision**
- **Text Encoder**: float16 (half precision)
- **VAE**: bfloat16 (brain float)
- **UNet**: float16
- **RMBG**: float32 (full precision)

---

## 🏗️ Architecture Patterns

### **Design Patterns**
- **MVC Pattern** (Model-View-Controller)
  - Model: AI models, data processing
  - View: HTML templates
  - Controller: Flask routes
- **Client-Server Architecture**
  - Frontend: HTML/CSS/JS
  - Backend: Flask API
  - Communication: RESTful API
- **Pipeline Pattern**
  - Background removal → Relighting → Upscaling

### **Code Organization**
```
light/
├── app.py                    # Flask application (main)
├── gradio_demo.py            # Gradio interface (legacy)
├── gradio_demo_bg.py         # Background-conditioned demo
├── upscaler.py               # ImageUpscaler class
├── gpt_recommendations.py    # GPTRecommendationClient class
├── briarmbg.py               # Background removal wrapper
├── db_examples.py            # Example database
├── templates/
│   └── index.html            # Flask UI template
├── static/
│   └── light_pointer.js      # Draggable light control (if exists)
├── models/                   # Model weights
├── uploads/                  # Temporary uploads
├── outputs/                  # Generated images
├── requirements.txt          # Dependencies
├── .env                      # Environment variables
└── .env.example              # Environment template
```

---

## 🔐 Security

### **SSL/TLS**
- **certifi** - Trusted CA bundle
- Environment variables:
  - `SSL_CERT_FILE`
  - `REQUESTS_CA_BUNDLE`
  - `CURL_CA_BUNDLE`

### **File Upload Security**
- **Werkzeug secure_filename()** - Sanitizes filenames
- **File size limit**: 50MB max
- **File type validation**: Image files only
- **Temporary storage**: Uploads folder

### **API Security**
- **Environment variables** - API keys not in code
- **.env file** - Gitignored, not committed
- **API key validation** - Graceful error handling

---

## 📊 Performance Optimizations

### **Model Optimizations**
- **Attention Processor 2.0** - Faster attention computation
- **Mixed Precision** - float16/bfloat16 for speed
- **Latent Space Processing** - Smaller tensors
- **Batch Processing** - Multiple images at once

### **Image Processing**
- **LANCZOS Resampling** - High-quality resizing
- **Center Cropping** - Maintains aspect ratio
- **Progressive Upscaling** - Lowres → Highres
- **Browser-Compatible Sizing** - Max 1280p for display

### **Web Performance**
- **Async Processing** - Non-blocking operations
- **Progress Indicators** - User feedback
- **File Paths vs Arrays** - Reduces memory transfer
- **Threaded Server** - Multiple concurrent requests

---

## 🌍 Deployment

### **Current Setup**
- **Development Server**: Flask built-in
- **Host**: `0.0.0.0` (all interfaces)
- **Port**: 5000
- **Access**: http://localhost:5000

### **Production Recommendations**
- **WSGI Server**: Gunicorn or uWSGI
- **Reverse Proxy**: Nginx
- **Process Manager**: Supervisor or systemd
- **Containerization**: Docker (optional)

---

## 📦 Dependencies Summary

### **Core ML/AI**
```
torch (CUDA 12.1)
diffusers==0.27.2
transformers==4.36.2
safetensors
```

### **Image Processing**
```
pillow==10.2.0
opencv-python
numpy
realesrgan>=0.3.0
basicsr>=1.4.2
facexlib>=0.3.0
gfpgan>=1.3.8
```

### **Web Framework**
```
flask>=2.0.0
werkzeug
```

### **AI Services**
```
openai>=1.0.0
```

### **Utilities**
```
einops
peft
protobuf==3.20
certifi
```

### **UI (Legacy)**
```
gradio==3.41.2
```

---

## 🎯 Key Features Enabled by Tech Stack

### **1. Draggable Light Source**
- **Technologies**: HTML5, CSS3, JavaScript
- **Features**:
  - Mouse/touch drag events
  - Real-time position tracking
  - Direction detection algorithm
  - Visual feedback (glow, pulse)
  - Sync with dropdown

### **2. AI Recommendations**
- **Technologies**: OpenAI GPT-4o-mini, Flask API
- **Features**:
  - Image analysis
  - Mood-based suggestions
  - One-click apply
  - Graceful fallback

### **3. High-Quality Relighting**
- **Technologies**: IC-Light, Stable Diffusion, PyTorch
- **Features**:
  - Text-conditioned lighting
  - Directional lighting control
  - Background removal
  - Highres fix

### **4. Automatic Upscaling**
- **Technologies**: Real-ESRGAN, basicsr
- **Features**:
  - 4x super-resolution
  - Minimum 1080p output
  - Face enhancement
  - Real-world quality

### **5. Modern UI/UX**
- **Technologies**: Flask, HTML5, CSS3, JavaScript
- **Features**:
  - Dark theme
  - Gradient backgrounds
  - Smooth animations
  - Progress indicators
  - Responsive design
  - No timeout issues

---

## 🔄 Processing Pipeline

```
1. User uploads image (HTML5 drag-drop)
   ↓
2. Flask receives file (Werkzeug)
   ↓
3. Background removal (BRIA RMBG 1.4)
   ↓
4. User sets lighting (Draggable pointer / Dropdown)
   ↓
5. Optional: AI recommendations (GPT-4o-mini)
   ↓
6. Relighting process:
   - Text encoding (CLIP)
   - Latent encoding (VAE)
   - Denoising (UNet + IC-Light)
   - Lowres generation (512x640)
   - Highres refinement (768x960)
   - Latent decoding (VAE)
   ↓
7. Upscaling (Real-ESRGAN 4x)
   ↓
8. Browser-compatible resize (max 1280p)
   ↓
9. Save to outputs folder (PNG)
   ↓
10. Display result (Flask serves file)
```

---

## 📈 Performance Metrics

### **Processing Time**
- Background removal: ~2-3 seconds
- Relighting (lowres): ~10-15 seconds
- Relighting (highres): ~5-10 seconds
- Upscaling: ~5-10 seconds
- **Total**: ~20-30 seconds per image

### **Quality Metrics**
- Input: Any resolution
- Processing: 512x640 → 768x960
- Output: Minimum 1080p (upscaled)
- Display: Max 1280p (browser-compatible)

---

## 🎓 Learning Resources

### **PyTorch**
- https://pytorch.org/docs/

### **Diffusers**
- https://huggingface.co/docs/diffusers/

### **Stable Diffusion**
- https://stability.ai/

### **IC-Light**
- https://github.com/lllyasviel/IC-Light

### **Real-ESRGAN**
- https://github.com/xinntao/Real-ESRGAN

### **Flask**
- https://flask.palletsprojects.com/

### **OpenAI API**
- https://platform.openai.com/docs/

---

## 🚀 Quick Start Commands

### **Activate Environment**
```powershell
D:\new1\venv_py310\Scripts\Activate.ps1
```

### **Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Set API Key**
```bash
# Windows CMD
set OPENAI_API_KEY=your-api-key-here

# PowerShell
$env:OPENAI_API_KEY="your-api-key-here"
```

### **Run Application**
```bash
# Flask (recommended)
python app.py

# Gradio (legacy)
python gradio_demo.py
```

### **Access Application**
```
http://localhost:5000
```

---

## 📝 Notes

- **GPU Required**: CUDA-capable NVIDIA GPU recommended
- **Model Downloads**: IC-Light and RMBG auto-download on first run
- **Manual Model**: Real-ESRGAN must be placed in `./models/` manually
- **API Key**: OpenAI API key required for AI recommendations
- **Commercial Use**: Replace BRIA RMBG with BiRefNet for commercial projects

---

**Your IC-Light application is powered by cutting-edge AI models and modern web technologies!** 🎨✨
