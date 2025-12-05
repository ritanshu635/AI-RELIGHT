🚀 RELIGHT — AI Cinematic Lighting Intelligence Assistant
Reimagining Light for Companies, Creators & Everyday Users

RELIGHT is an advanced AI-powered lighting engine that transforms any image into a studio-quality visual using intelligent relighting, background-aware lighting matching, and HD upscaling — all without the need for professional equipment or editing skills.

Built using IC-Light, GPT-4o-mini Vision, and Real-ESRGAN, RELIGHT brings the power of a full photography lighting setup directly to your laptop.

✨ Key Features
⭐ 1. AI-Recommended Lighting (GPT-4o-mini Vision)

A virtual lighting director that analyzes your image and suggests three optimized lighting moods (max 3 words each).

Upload → Click “AI Recommendations” → Apply lighting

Perfect for beginners, marketers, and fast creative workflows

GPT analyzes: shadows, highlights, mood, exposure, color ambiance

Example Suggestions:
Soft Rimlight, Moody Sideglow, Warm Spotlight

⭐ 2. Background-Aware Relighting (IC-Light FBC)

Seamless composites with realistic lighting that matches the new background.

Upload subject + background → Click “Relight”

IC-Light matches:

Ambient direction

Color temperature

Shadow falloff

Contrast and mood

Ideal for:
Product advertising, e-commerce, design mockups, thumbnails, posters

⭐ 3. Manual Text-Based Relighting (IC-Light FC)

Full creative freedom using text prompts.

Examples:
cold sidelight, golden rimlight, soft cinematic glow

IC-Light applies physically realistic lighting by injecting lighting embeddings in diffusion space.

⭐ 4. Interactive Light Gizmo

Drag a light point → RELIGHT converts position → optimized lighting parameters
Perfect for visual thinkers and designers.

⭐ 5. High-Resolution Upscaling (Real-ESRGAN x4plus)

Every final output is processed through ESRGAN:

Minimum 1080p

Crisp edges

High detail

Printable and advertisement-ready quality

🧠 System Architecture
UPLOAD
   ↓
RMBG (Background Removal)
   ↓
RELIGHT ENGINE
   ├── IC-Light (FC) – Text-based relighting
   └── IC-Light (FBC) – Background-aware relighting
   ↓
GPT Lighting Advisor (Optional)
   ↓
Real-ESRGAN Upscaler
   ↓
OUTPUT (HD)

🔧 Tech Stack
Backend

Python 3.10

Flask

PyTorch

Diffusers

Transformers

SafeTensors

BRIA RMBG

Real-ESRGAN

AI Models

IC-Light FC (foreground relighting)

IC-Light FBC (background conditioned)

GPT-4o-mini Vision

Real-ESRGAN x4plus

Frontend

HTML

CSS

JavaScript

Drag-and-drop light gizmo

Live preview panel

Clean, modern UI

🛠 How It Works (Short Explanation)
1. Background Removal

RMBG extracts a high-quality alpha matte so lighting applies correctly.

2. Lighting Processing

Depending on the mode:

Manual → IC-Light FC

AI Recommendation → GPT → IC-Light FC

Background Mode → IC-Light FBC

3. GPT Lighting Advisor

Image → Base64 → GPT → 3 lighting prompts

4. Super Resolution

Real-ESRGAN ensures HD clarity every time.

5. Output

Displayed instantly in the UI, ready for download.

⚠️ Challenges Tackled

GPU VRAM limits on 8GB cards

Multi-model memory optimization

Achieving perfect background–subject lighting match

Designing intuitive light-drag UI

Keeping inference under 12 seconds

🏆 Achievements

Fully working end-to-end AI lighting pipeline

Background-aware relighting with strong realism

GPT-powered lighting intelligence

Smooth 1080p+ HD outputs

Modern, intuitive UI

Works on consumer hardware

📚 Key Learnings

Lighting in diffusion is extremely sensitive

Multi-model pipelines need smart VRAM management

GPT massively improves usability & creativity

Background lighting consistency is crucial

UI/UX impacts adoption more than model accuracy

🔭 Roadmap
Immediate

More lighting presets

Live preview mode

Faster background blending

Advanced

3D relighting with depth maps

Video relighting

Mobile version

Public API for brands

📦 Installation
git clone https://github.com/yourusername/relight
cd relight
pip install -r requirements.txt
python app.py


Make sure to download:

IC-Light FC model

IC-Light FBC model

Real-ESRGAN x4plus

RMBG 1.4

Place them in the /models directory.

▶️ Usage
Run Backend
python app.py

Access Frontend

Open:

http://localhost:5000


Upload → Choose mode → Relight → Download HD output.

🌟 Why RELIGHT Matters

RELIGHT democratizes lighting for:

Companies & Advertising Teams

No studio | No retouching | Ultra-fast campaign variations

E-commerce Sellers

High-quality product photos instantly

Designers & Filmmakers

Creative control without equipment

Everyday Users

Professional lighting with zero editing skills

Lighting becomes:
Simple. Intelligent. Beautiful. Accessible.

🤝 Contributors

Ritanshu — Creator & Developer
AI Integration • Backend • Frontend • UX Engineering
