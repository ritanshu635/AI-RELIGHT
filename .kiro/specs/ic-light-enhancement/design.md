# Design Document

## Overview

This design document outlines the technical architecture for enhancing the IC-Light image relighting application. The enhancement adds five major features: a modernized UI, AI-powered lighting recommendations via GPT-4o-mini, background image blending, automatic 1080p upscaling with Real-ESRGAN, and an optional interactive light direction control. The design maintains backward compatibility with existing IC-Light functionality while extending capabilities through modular additions.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Gradio Web Interface                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Custom CSS   │  │ Image Upload │  │ AI Recommend │      │
│  │ Styling      │  │ Components   │  │ Button       │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Backend Processing Layer                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ GPT Vision   │  │ IC-Light     │  │ Real-ESRGAN  │      │
│  │ API Client   │  │ Pipeline     │  │ Upscaler     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┐
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      Model Layer (GPU)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ IC-Light FC  │  │ IC-Light FBC │  │ RealESRGAN   │      │
│  │ Model        │  │ Model        │  │ x4plus       │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow

1. **User uploads image** → Gradio Interface → Backend
2. **AI Recommendations requested** → GPT Vision API → Returns 3 suggestions
3. **User selects suggestion** → Populates prompt field
4. **Relight/Blend triggered** → IC-Light Model → Raw output
5. **Raw output** → Real-ESRGAN Upscaler → 1080p output
6. **Final image** → Gradio Interface → Display to user

## Components and Interfaces

### 1. UI Styling Module

**File:** `light/gradio_demo.py` and `light/gradio_demo_bg.py`

**Purpose:** Apply modern visual design to the Gradio interface

**Implementation:**
```python
# Custom CSS block to be added
custom_css = """
/* Dark theme base */
.gradio-container {
    background: linear-gradient(135deg, #0B0F19 0%, #1a1f2e 100%) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Typography */
h1, h2, h3 {
    color: #E6E9EF !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
}

label {
    color: #B8BCC8 !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
}

/* Buttons */
.gr-button {
    background: linear-gradient(135deg, #6C5CE7 0%, #5B4BC4 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 0.75rem 1.5rem !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 12px rgba(108, 92, 231, 0.3) !important;
}

.gr-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(108, 92, 231, 0.4) !important;
}

/* Input fields and sliders */
.gr-input, .gr-slider {
    background-color: #1E2330 !important;
    border: 1px solid #2D3548 !important;
    color: #E6E9EF !important;
    border-radius: 6px !important;
}

/* Image containers */
.gr-image {
    border-radius: 12px !important;
    border: 2px solid #2D3548 !important;
    overflow: hidden !important;
}

/* Gallery */
.gr-gallery {
    background-color: #1E2330 !important;
    border-radius: 12px !important;
    border: 1px solid #2D3548 !important;
}

/* Accordion */
.gr-accordion {
    background-color: #1E2330 !important;
    border: 1px solid #2D3548 !important;
    border-radius: 8px !important;
}

/* Special button for AI Recommendations */
.ai-recommend-btn {
    background: linear-gradient(135deg, #00D9FF 0%, #0099CC 100%) !important;
    box-shadow: 0 4px 12px rgba(0, 217, 255, 0.3) !important;
}

.ai-recommend-btn:hover {
    box-shadow: 0 6px 20px rgba(0, 217, 255, 0.4) !important;
}
"""
```

**Interface:**
- Input: Gradio Blocks object
- Output: Styled Gradio interface
- Method: Apply CSS via `gr.Blocks(css=custom_css)`

### 2. GPT Vision API Client

**File:** `light/gpt_recommendations.py` (new file)

**Purpose:** Interface with OpenAI GPT-4o-mini for lighting recommendations

**Implementation:**
```python
import base64
import os
from openai import OpenAI
from PIL import Image
import io

class GPTRecommendationClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=self.api_key)
    
    def image_to_base64(self, image_array):
        """Convert numpy array to base64 string"""
        img = Image.fromarray(image_array)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def get_lighting_recommendations(self, image_array):
        """
        Get 3 lighting recommendations from GPT-4o-mini
        Returns: List of 3 strings, each max 3 words
        """
        base64_image = self.image_to_base64(image_array)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            },
                            {
                                "type": "text",
                                "text": "Analyze this image and suggest 3 lighting styles that would enhance it. Each suggestion must be EXACTLY 3 words or less. Format: one suggestion per line, no numbering. Examples: 'warm golden hour', 'dramatic side lighting', 'soft studio light'"
                            }
                        ]
                    }
                ],
                max_tokens=100
            )
            
            # Parse response
            content = response.choices[0].message.content
            suggestions = [s.strip() for s in content.split('\n') if s.strip()]
            
            # Ensure we have exactly 3 suggestions, each max 3 words
            suggestions = suggestions[:3]
            suggestions = [' '.join(s.split()[:3]) for s in suggestions]
            
            # Pad if less than 3
            while len(suggestions) < 3:
                suggestions.append("natural lighting")
            
            return suggestions
            
        except Exception as e:
            print(f"GPT API Error: {e}")
            # Return default suggestions on error
            return ["warm golden light", "dramatic side lighting", "soft studio light"]
```

**Interface:**
- Input: numpy array (image)
- Output: List of 3 strings (lighting suggestions)
- Dependencies: openai, PIL, base64

### 3. Real-ESRGAN Upscaling Module

**File:** `light/upscaler.py` (new file)

**Purpose:** Upscale all output images to minimum 1080p resolution

**Implementation:**
```python
import torch
import numpy as np
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

class ImageUpscaler:
    def __init__(self, model_path='./models/RealESRGAN_x4plus.pth', device='cuda'):
        self.device = device
        self.model = None
        self.model_path = model_path
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize Real-ESRGAN model"""
        # Define RRDBNet architecture for RealESRGAN_x4plus
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4
        )
        
        # Determine half precision based on device
        use_half = True if self.device == 'cuda' else False
        
        # Create RealESRGANer upsampler
        self.upsampler = RealESRGANer(
            scale=4,
            model_path=self.model_path,
            model=model,
            tile=0,  # 0 for no tile, or set to 400-800 for large images
            tile_pad=10,
            pre_pad=0,
            half=use_half,
            device=self.device
        )
    
    def upscale_to_1080p(self, image_array):
        """
        Upscale image to minimum 1080p height
        Input: numpy array (H, W, C) in uint8 format
        Output: numpy array (H', W', C) where H' >= 1080
        """
        # Ensure input is uint8
        if isinstance(image_array, np.ndarray):
            if image_array.dtype == np.float32 or image_array.dtype == np.float64:
                image_array = (image_array * 255).clip(0, 255).astype(np.uint8)
        
        # Get original dimensions
        h_orig, w_orig = image_array.shape[:2]
        
        # Calculate target scale to reach at least 1080p
        # First apply 4x upscaling, then check if additional scaling needed
        target_scale = 4.0
        if (h_orig * 4) < 1080:
            # Need more than 4x to reach 1080p
            target_scale = 1080 / h_orig
        
        try:
            # Apply Real-ESRGAN upscaling with custom outscale
            output, _ = self.upsampler.enhance(image_array, outscale=target_scale)
            
            # Verify output meets 1080p requirement
            h_out, w_out = output.shape[:2]
            if h_out < 1080:
                # Additional resize if still below 1080p (fallback)
                scale_factor = 1080 / h_out
                new_h = 1080
                new_w = int(w_out * scale_factor)
                output_img = Image.fromarray(output)
                output_img = output_img.resize((new_w, new_h), Image.LANCZOS)
                output = np.array(output_img)
            
            return output
            
        except Exception as e:
            print(f"Upscaling error: {e}")
            # Fallback: simple resize if Real-ESRGAN fails
            scale_factor = max(1080 / h_orig, 1.0)
            new_h = int(h_orig * scale_factor)
            new_w = int(w_orig * scale_factor)
            img = Image.fromarray(image_array)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            return np.array(img)
```

**Interface:**
- Input: numpy array (image, any resolution)
- Output: numpy array (image, min 1080p height)
- Dependencies: basicsr, realesrgan, torch

**Model Location:** `D:\new1\light\models\RealESRGAN_x4plus.pth`

### 4. Background Blending Integration

**File:** Modifications to `light/gradio_demo_bg.py`

**Purpose:** Enable user-uploaded background images with automatic AI prompt application

**Key Changes:**
1. Add "Add Background Image" button in the foreground upload section
2. Modify the interface to show/hide background upload based on user action
3. Create new "Blend" button that triggers background-conditioned relighting
4. Integrate AI recommendations into the blend workflow

**Implementation Pattern:**
```python
# In gradio_demo_bg.py
with gr.Column():
    input_fg = gr.Image(source='upload', type="numpy", label="Foreground Image", height=480)
    
    # New: Add background button
    add_bg_button = gr.Button(value="Add Background Image", elem_classes=["gr-button"])
    
    # New: Background upload (initially hidden)
    input_bg = gr.Image(
        source='upload',
        type="numpy",
        label="Background Image",
        height=480,
        visible=False
    )
    
    # New: Blend button
    blend_button = gr.Button(value="Blend with Background", visible=False)
    
    # AI Recommendations section
    ai_recommend_button = gr.Button(value="Get AI Recommendations", elem_classes=["ai-recommend-btn"])
    ai_suggestions = gr.Radio(label="AI Lighting Suggestions", choices=[], visible=False)
    apply_ai_button = gr.Button(value="Apply Selected Suggestion", visible=False)

# Event handlers
def show_background_upload():
    return gr.update(visible=True), gr.update(visible=True)

add_bg_button.click(
    fn=show_background_upload,
    outputs=[input_bg, blend_button]
)
```

### 5. Optional: Interactive Light Direction Control

**File:** `light/static/light_pointer.js` (new file) + modifications to Gradio demos

**Purpose:** Allow users to drag a light pointer on the image to set direction

**Implementation:**

**JavaScript Component:**
```javascript
// light_pointer.js
class LightPointerControl {
    constructor(imageElement, callbackFn) {
        this.image = imageElement;
        this.callback = callbackFn;
        this.pointer = null;
        this.isDragging = false;
        this.init();
    }
    
    init() {
        // Create pointer overlay
        this.pointer = document.createElement('div');
        this.pointer.className = 'light-pointer';
        this.pointer.style.cssText = `
            position: absolute;
            width: 40px;
            height: 40px;
            background: radial-gradient(circle, #FFD700 0%, #FFA500 100%);
            border-radius: 50%;
            cursor: move;
            box-shadow: 0 0 20px rgba(255, 215, 0, 0.8);
            z-index: 1000;
        `;
        
        // Position at center initially
        this.updatePosition(0.5, 0.5);
        
        // Add to image container
        const container = this.image.parentElement;
        container.style.position = 'relative';
        container.appendChild(this.pointer);
        
        // Event listeners
        this.pointer.addEventListener('mousedown', this.startDrag.bind(this));
        document.addEventListener('mousemove', this.drag.bind(this));
        document.addEventListener('mouseup', this.endDrag.bind(this));
    }
    
    startDrag(e) {
        this.isDragging = true;
        e.preventDefault();
    }
    
    drag(e) {
        if (!this.isDragging) return;
        
        const rect = this.image.getBoundingClientRect();
        const x = (e.clientX - rect.left) / rect.width;
        const y = (e.clientY - rect.top) / rect.height;
        
        this.updatePosition(x, y);
        this.updateDirection(x, y);
    }
    
    endDrag() {
        this.isDragging = false;
    }
    
    updatePosition(x, y) {
        const rect = this.image.getBoundingClientRect();
        this.pointer.style.left = `${x * rect.width - 20}px`;
        this.pointer.style.top = `${y * rect.height - 20}px`;
    }
    
    updateDirection(x, y) {
        // Map coordinates to direction
        let direction = "None";
        
        // Define regions (with some overlap for corners)
        if (x < 0.33) {
            if (y < 0.33) direction = "Top Left";
            else if (y > 0.67) direction = "Bottom Left";
            else direction = "Left";
        } else if (x > 0.67) {
            if (y < 0.33) direction = "Top Right";
            else if (y > 0.67) direction = "Bottom Right";
            else direction = "Right";
        } else {
            if (y < 0.33) direction = "Top";
            else if (y > 0.67) direction = "Bottom";
            else direction = "None";
        }
        
        // Call callback with direction
        if (this.callback) {
            this.callback(direction);
        }
    }
}
```

**Python Integration:**
```python
# In gradio_demo.py
# Add custom JavaScript
custom_js = """
<script src="file=light/static/light_pointer.js"></script>
<script>
function initLightPointer() {
    const imageEl = document.querySelector('#input_fg img');
    if (imageEl) {
        new LightPointerControl(imageEl, function(direction) {
            // Update hidden direction field
            document.querySelector('#light_direction_hidden').value = direction;
        });
    }
}
setTimeout(initLightPointer, 1000);
</script>
"""

# Add hidden field for direction
light_direction_hidden = gr.Textbox(elem_id="light_direction_hidden", visible=False)
```

**Note:** This feature is marked as optional and may require additional Gradio customization or a custom HTML component.

## Data Models

### GPT Recommendation Response
```python
{
    "suggestions": [
        "warm golden hour",
        "dramatic side lighting",
        "soft studio light"
    ]
}
```

### Upscaler Configuration
```python
{
    "model_path": "D:\\new1\\light\\models\\RealESRGAN_x4plus.pth",
    "scale": 4,
    "target_min_height": 1080,
    "device": "cuda"
}
```

### UI State
```python
{
    "foreground_image": numpy.ndarray,
    "background_image": numpy.ndarray | None,
    "selected_ai_suggestion": str | None,
    "light_direction": str,  # "Left", "Right", "Top", "Bottom", etc.
    "current_mode": str  # "text_conditioned" or "background_conditioned"
}
```

## Error Handling

### GPT API Errors
- **Timeout:** Return default suggestions after 10 seconds
- **Invalid API Key:** Display error message, disable AI recommendations
- **Rate Limit:** Show user-friendly message, suggest retry
- **Network Error:** Fall back to default suggestions

### Upscaling Errors
- **Model Not Found:** Check `D:\new1\light\models\RealESRGAN_x4plus.pth`, display error if missing
- **GPU Memory Error:** Fall back to CPU processing with warning
- **Invalid Image:** Skip upscaling, return original image

### Background Blending Errors
- **No Background Uploaded:** Display validation message
- **Image Size Mismatch:** Automatically resize to match
- **Model Loading Error:** Fall back to text-conditioned mode

## Testing Strategy

### Unit Tests

1. **GPT Client Tests**
   - Test base64 encoding of various image formats
   - Mock API responses and verify parsing
   - Test error handling for API failures

2. **Upscaler Tests**
   - Test upscaling of images below 1080p
   - Test upscaling of images already above 1080p
   - Verify aspect ratio preservation
   - Test GPU and CPU modes

3. **UI Component Tests**
   - Verify CSS application
   - Test button visibility toggling
   - Test prompt field population

### Integration Tests

1. **End-to-End Workflow**
   - Upload image → Get AI recommendations → Apply → Relight → Verify 1080p output
   - Upload foreground + background → Blend → Verify output
   - Test all features in sequence

2. **Model Integration**
   - Verify IC-Light models load correctly
   - Verify Real-ESRGAN model loads from specified path
   - Test GPU memory management with all models loaded

### Performance Tests

1. **Latency Measurements**
   - Measure GPT API response time
   - Measure IC-Light processing time
   - Measure Real-ESRGAN upscaling time
   - Verify total pipeline < 30 seconds

2. **Memory Tests**
   - Monitor GPU memory usage
   - Test with various image sizes
   - Verify no memory leaks

### User Acceptance Tests

1. **UI/UX Validation**
   - Verify visual design matches requirements
   - Test responsiveness of interface
   - Validate button interactions

2. **Feature Validation**
   - Verify AI suggestions are relevant
   - Verify upscaled images are high quality
   - Verify background blending produces natural results

## Dependencies

### New Python Packages
```
openai>=1.0.0
basicsr>=1.4.2
realesrgan>=0.3.0
facexlib>=0.3.0
gfpgan>=1.3.8
```

### Existing Dependencies (from requirements.txt)
```
diffusers==0.27.2
transformers==4.36.2
opencv-python
safetensors
pillow==10.2.0
einops
torch
peft
gradio==3.41.2
protobuf==3.20
```

### Environment Variables
```
OPENAI_API_KEY=<user-provided-key>
```

## Deployment Considerations

1. **Model Files:**
   - IC-Light models: Auto-download from HuggingFace (existing)
   - Real-ESRGAN model: Already present at `D:\new1\light\models\RealESRGAN_x4plus.pth`

2. **API Keys:**
   - GPT API key will be provided by user
   - Store in environment variable or config file

3. **GPU Requirements:**
   - CUDA 12.1 compatible GPU (existing requirement)
   - Minimum 8GB VRAM recommended for all models
   - 12GB+ VRAM recommended for optimal performance

4. **Virtual Environment:**
   - Use existing `D:\new1\venv_py310`
   - Install new dependencies via pip

## File Structure

```
light/
├── gradio_demo.py (modified)
├── gradio_demo_bg.py (modified)
├── gpt_recommendations.py (new)
├── upscaler.py (new)
├── static/ (new)
│   └── light_pointer.js (new, optional)
├── models/
│   ├── iclight_sd15_fc.safetensors
│   ├── iclight_sd15_fbc.safetensors
│   └── RealESRGAN_x4plus.pth (existing)
├── requirements.txt (modified)
└── ... (existing files)
```
