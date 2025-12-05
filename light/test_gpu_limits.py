import torch
import numpy as np
from upscaler import ImageUpscaler

def test_gpu_capacity():
    """Test to find practical upscaling limits on RTX 4060 8GB"""
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    upscaler = ImageUpscaler(device='cuda')
    
    # Test different resolutions
    test_cases = [
        (480, 854, "480p (SD)"),
        (720, 1280, "720p (HD)"),
        (1080, 1920, "1080p (FHD)"),
        (1440, 2560, "1440p (2K)"),
        (2160, 3840, "2160p (4K)"),
    ]
    
    for height, width, label in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing {label}: {height}x{width}")
        
        # Clear cache before test
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Create test image
        test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        try:
            result = upscaler.upscale_to_1080p(test_image)
            
            # Get memory stats
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
            print(f"✓ Success! Output: {result.shape[0]}x{result.shape[1]}")
            print(f"  Peak VRAM used: {peak_memory:.2f} GB")
            
        except Exception as e:
            print(f"✗ Failed: {str(e)[:100]}")
        
        finally:
            torch.cuda.empty_cache()

if __name__ == '__main__':
    test_gpu_capacity()
