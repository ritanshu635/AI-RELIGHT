# Project Structure

## Directory Layout

```
.
├── .kiro/                      # Kiro configuration
│   ├── specs/                  # Feature specifications
│   └── steering/               # Steering rules
├── docs/                       # Documentation
│   ├── model_zoo.md
│   ├── Training.md
│   ├── FAQ.md
│   └── ...
├── light/                      # Main application directory
│   ├── gradio_demo.py          # Text-conditioned demo (main entry)
│   ├── gradio_demo_bg.py       # Background-conditioned demo
│   ├── upscaler.py             # Real-ESRGAN upscaling module
│   ├── gpt_recommendations.py  # GPT-4o-mini API client
│   ├── briarmbg.py             # Background removal
│   ├── db_examples.py          # Example database
│   ├── models/                 # Model weights directory
│   ├── static/                 # Static assets (JS, CSS)
│   │   └── light_pointer.js    # Interactive light control
│   ├── outputs/                # Generated images (auto-created)
│   ├── test_*.py               # Unit tests
│   ├── validate_integration.py # Integration tests
│   ├── requirements.txt        # Python dependencies
│   ├── .env                    # Environment variables (gitignored)
│   └── .env.example            # Environment template
└── venv_py310/                 # Python virtual environment
```

## Code Organization

### Main Application Files

- **gradio_demo.py**: Primary demo with text-conditioned relighting
  - Initializes all models (SD, VAE, UNet, RMBG)
  - Defines UI with Gradio blocks
  - Implements processing pipeline
  - Integrates AI recommendations and upscaling

- **gradio_demo_bg.py**: Background-conditioned variant
  - Similar structure to gradio_demo.py
  - Adds background image upload and blending
  - Uses `iclight_sd15_fbc.safetensors` model

### Feature Modules

- **upscaler.py**: `ImageUpscaler` class
  - Wraps Real-ESRGAN for 1080p upscaling
  - Handles GPU/CPU fallback
  - Preserves aspect ratio

- **gpt_recommendations.py**: `GPTRecommendationClient` class
  - OpenAI API integration
  - Image to base64 conversion
  - Returns 3 lighting suggestions (max 3 words each)
  - Graceful error handling with defaults

- **briarmbg.py**: Background removal model wrapper

### Static Assets

- **static/light_pointer.js**: `LightPointerControl` JavaScript class
  - Draggable light direction pointer
  - 9-region direction mapping
  - Touch support

## Naming Conventions

### Python Files
- Main demos: `gradio_demo*.py`
- Feature modules: descriptive names (`upscaler.py`, `gpt_recommendations.py`)
- Tests: `test_*.py` prefix
- Validation: `validate_*.py` prefix

### Python Code Style
- Classes: PascalCase (`ImageUpscaler`, `GPTRecommendationClient`)
- Functions: snake_case (`process_relight`, `get_ai_recommendations`)
- Constants: UPPER_SNAKE_CASE or Enum classes (`BGSource`)
- Private methods: `_method_name` prefix

### Variables
- Descriptive names: `input_fg`, `output_bg`, `upscaled_results`
- Model instances: `upscaler`, `gpt_client`, `rmbg`
- Pipelines: `t2i_pipe`, `i2i_pipe`

## Testing Structure

### Test Files
- One test file per module: `test_upscaler.py`, `test_gpt_recommendations.py`
- Integration tests: `test_integration.py`, `validate_integration.py`
- Performance tests: `test_performance.py`, `test_gpu_limits.py`

### Test Classes
- Inherit from `unittest.TestCase`
- Class name: `Test{ModuleName}` (e.g., `TestImageUpscaler`)
- Setup: `setUpClass()` for expensive initialization
- Test methods: `test_*` prefix with descriptive names

## Configuration Files

- **requirements.txt**: Pinned versions for stability
- **.env**: Local environment variables (not committed)
- **.env.example**: Template for required variables
- **.gitignore**: Excludes models, outputs, venv, cache

## Model Storage

All models stored in `light/models/`:
- IC-Light models: Auto-downloaded on first run
- Real-ESRGAN: Must be manually placed
- BRIA RMBG: Auto-downloaded

## Output Storage

Generated images saved to `light/outputs/`:
- Naming: `relit_{timestamp}_{index}.png`
- Auto-created if doesn't exist
- Not tracked in git

## Import Patterns

```python
# Standard library
import os
import math
from enum import Enum

# Third-party
import torch
import numpy as np
import gradio as gr
from PIL import Image

# Local modules
from upscaler import ImageUpscaler
from gpt_recommendations import GPTRecommendationClient
```

## UI Component Organization

Gradio blocks structured as:
1. Header/title
2. Input column (left)
   - Image upload
   - AI recommendations
   - Prompt and controls
   - Advanced options (accordion)
3. Output column (right)
   - Result gallery
4. Examples section (bottom)
