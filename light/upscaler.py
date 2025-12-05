import os
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
        self.upsampler = None
        self.model_available = False
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize Real-ESRGAN model"""
        # Check if model file exists
        if not os.path.exists(self.model_path):
            print(f"WARNING: Real-ESRGAN model not found at {self.model_path}")
            print("Upscaling will be disabled. Download model from:")
            print("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth")
            self.model_available = False
            return
        
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
        try:
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
            self.model_available = True
            print(f"Real-ESRGAN model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error initializing Real-ESRGAN model: {e}")
            # Fallback to CPU if GPU initialization fails
            if self.device == 'cuda':
                print("Falling back to CPU...")
                self.device = 'cpu'
                try:
                    self.upsampler = RealESRGANer(
                        scale=4,
                        model_path=self.model_path,
                        model=model,
                        tile=0,
                        tile_pad=10,
                        pre_pad=0,
                        half=False,
                        device=self.device
                    )
                    self.model_available = True
                    print("Real-ESRGAN model loaded successfully on CPU")
                except Exception as e2:
                    print(f"Failed to initialize on CPU: {e2}")
                    self.model_available = False
    
    def upscale_to_1080p(self, image_array):
        """
        Upscale image to minimum 1080p height if below 1080p, otherwise return as-is
        Input: numpy array (H, W, C) in uint8 format
        Output: numpy array (H', W', C) where H' >= 1080 or original if already >= 1080
        """
        # If model not available, use simple PIL resize
        if not self.model_available or self.upsampler is None:
            return self._simple_upscale(image_array)
        
        # Ensure input is uint8
        if isinstance(image_array, np.ndarray):
            if image_array.dtype == np.float32 or image_array.dtype == np.float64:
                image_array = (image_array * 255).clip(0, 255).astype(np.uint8)
        
        # Get original dimensions
        h_orig, w_orig = image_array.shape[:2]
        
        # If image is already 1080p or higher, return as-is
        if h_orig >= 1080:
            print(f"Image already at {h_orig}p, no upscaling needed")
            return image_array
        
        # Calculate target scale to reach at least 1080p
        target_scale = 4.0
        if (h_orig * 4) < 1080:
            # Need more than 4x to reach 1080p
            target_scale = 1080 / h_orig
        
        try:
            # Apply Real-ESRGAN upscaling
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
            # If upscaling fails, use simple upscale
            print("Falling back to simple upscaling")
            return self._simple_upscale(image_array)
    
    def _simple_upscale(self, image_array):
        """Simple upscaling using PIL when Real-ESRGAN is not available"""
        # Ensure input is uint8
        if isinstance(image_array, np.ndarray):
            if image_array.dtype == np.float32 or image_array.dtype == np.float64:
                image_array = (image_array * 255).clip(0, 255).astype(np.uint8)
        
        h_orig, w_orig = image_array.shape[:2]
        
        # If already 1080p or higher, return as-is
        if h_orig >= 1080:
            return image_array
        
        # Calculate scale to reach 1080p
        scale = 1080 / h_orig
        new_h = 1080
        new_w = int(w_orig * scale)
        
        print(f"Simple upscaling from {h_orig}x{w_orig} to {new_h}x{new_w}")
        
        pil_img = Image.fromarray(image_array)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        return np.array(pil_img)
