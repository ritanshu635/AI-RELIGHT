import torch
import numpy as np
from upscaler import ImageUpscaler

def test_large_image():
    """Test upscaling a larger image with automatic tiling"""
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")
    
    upscaler = ImageUpscaler(device='cuda')
    
    # Test 900p image (should trigger tiling)
    height, width = 900, 1600
    print(f"Testing {height}x{width} image (should use tiling)")
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    try:
        result = upscaler.upscale_to_1080p(test_image)
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        
        print(f"✓ Success!")
        print(f"  Input: {height}x{width}")
        print(f"  Output: {result.shape[0]}x{result.shape[1]}")
        print(f"  Peak VRAM: {peak_memory:.2f} GB")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    finally:
        torch.cuda.empty_cache()

if __name__ == '__main__':
    test_large_image()
