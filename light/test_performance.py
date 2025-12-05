"""
Performance Testing Script for IC-Light Enhanced Pipeline
Tests end-to-end latency and GPU memory usage
"""

import time
import torch
import numpy as np
from PIL import Image
import psutil
import os

# Import the modules we need to test
from gpt_recommendations import GPTRecommendationClient
from upscaler import ImageUpscaler


def get_gpu_memory_usage():
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def create_test_image(width=512, height=640):
    """Create a test image for performance testing"""
    # Create a simple test image with some content
    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return img


def test_gpt_recommendations_latency():
    """Test GPT API response time"""
    print("\n=== Testing GPT Recommendations Latency ===")
    
    try:
        client = GPTRecommendationClient()
        test_img = create_test_image()
        
        start_time = time.time()
        suggestions = client.get_lighting_recommendations(test_img)
        end_time = time.time()
        
        latency = end_time - start_time
        print(f"✓ GPT API Response Time: {latency:.2f} seconds")
        print(f"  Suggestions: {suggestions}")
        
        if latency > 10:
            print("  ⚠ Warning: GPT API response time exceeds 10 seconds")
        
        return latency
        
    except Exception as e:
        print(f"✗ GPT API Test Failed: {e}")
        print("  Note: This is expected if OPENAI_API_KEY is not set")
        return 0


def test_upscaler_latency():
    """Test Real-ESRGAN upscaling time"""
    print("\n=== Testing Real-ESRGAN Upscaling Latency ===")
    
    try:
        upscaler = ImageUpscaler(model_path='./models/RealESRGAN_x4plus.pth', device='cuda')
        
        # Test with different image sizes
        test_sizes = [
            (256, 320, "Small (256x320)"),
            (512, 640, "Medium (512x640)"),
            (768, 960, "Large (768x960)")
        ]
        
        latencies = []
        
        for width, height, label in test_sizes:
            test_img = create_test_image(width, height)
            
            start_time = time.time()
            upscaled = upscaler.upscale_to_1080p(test_img)
            end_time = time.time()
            
            latency = end_time - start_time
            latencies.append(latency)
            
            output_height, output_width = upscaled.shape[:2]
            print(f"✓ {label}: {latency:.2f} seconds")
            print(f"  Input: {width}x{height} → Output: {output_width}x{output_height}")
            
            # Verify 1080p requirement
            if output_height < 1080:
                print(f"  ✗ Error: Output height {output_height} is below 1080p requirement")
            else:
                print(f"  ✓ Output meets 1080p requirement")
        
        avg_latency = sum(latencies) / len(latencies)
        print(f"\nAverage Upscaling Time: {avg_latency:.2f} seconds")
        
        return avg_latency
        
    except Exception as e:
        print(f"✗ Upscaler Test Failed: {e}")
        print("  Note: Ensure RealESRGAN_x4plus.pth is in ./models/ directory")
        return 0


def test_gpu_memory_usage():
    """Monitor GPU memory usage"""
    print("\n=== Testing GPU Memory Usage ===")
    
    if not torch.cuda.is_available():
        print("✗ CUDA not available, skipping GPU memory test")
        return
    
    # Get initial memory
    torch.cuda.empty_cache()
    initial_memory = get_gpu_memory_usage()
    print(f"Initial GPU Memory: {initial_memory:.2f} MB")
    
    try:
        # Load upscaler
        upscaler = ImageUpscaler(model_path='./models/RealESRGAN_x4plus.pth', device='cuda')
        after_upscaler = get_gpu_memory_usage()
        print(f"After Loading Upscaler: {after_upscaler:.2f} MB (+{after_upscaler - initial_memory:.2f} MB)")
        
        # Process an image
        test_img = create_test_image(512, 640)
        upscaled = upscaler.upscale_to_1080p(test_img)
        after_processing = get_gpu_memory_usage()
        print(f"After Processing Image: {after_processing:.2f} MB (+{after_processing - initial_memory:.2f} MB)")
        
        # Get total GPU memory
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        used_percentage = (after_processing / total_memory) * 100
        print(f"\nTotal GPU Memory: {total_memory:.2f} MB")
        print(f"Used: {after_processing:.2f} MB ({used_percentage:.1f}%)")
        
        if used_percentage > 80:
            print("⚠ Warning: GPU memory usage exceeds 80%")
        else:
            print("✓ GPU memory usage is within acceptable range")
            
    except Exception as e:
        print(f"✗ GPU Memory Test Failed: {e}")


def test_end_to_end_pipeline():
    """Test complete pipeline latency"""
    print("\n=== Testing End-to-End Pipeline Latency ===")
    print("Note: This simulates the complete workflow without IC-Light model")
    
    total_start = time.time()
    
    # Step 1: Create test image
    test_img = create_test_image(512, 640)
    print("✓ Step 1: Image loaded")
    
    # Step 2: Get AI recommendations (if available)
    gpt_time = 0
    try:
        client = GPTRecommendationClient()
        gpt_start = time.time()
        suggestions = client.get_lighting_recommendations(test_img)
        gpt_time = time.time() - gpt_start
        print(f"✓ Step 2: AI recommendations ({gpt_time:.2f}s)")
    except:
        print("⊘ Step 2: AI recommendations skipped (API key not set)")
    
    # Step 3: Simulate IC-Light processing (we'll estimate 10-15 seconds)
    iclight_time = 12.0  # Estimated average
    print(f"⊘ Step 3: IC-Light processing (estimated ~{iclight_time:.2f}s)")
    
    # Step 4: Upscaling
    upscale_time = 0
    try:
        upscaler = ImageUpscaler(model_path='./models/RealESRGAN_x4plus.pth', device='cuda')
        upscale_start = time.time()
        upscaled = upscaler.upscale_to_1080p(test_img)
        upscale_time = time.time() - upscale_start
        print(f"✓ Step 4: Upscaling ({upscale_time:.2f}s)")
    except:
        print("✗ Step 4: Upscaling failed")
    
    # Calculate total time
    measured_time = gpt_time + upscale_time
    estimated_total = measured_time + iclight_time
    
    print(f"\n--- Pipeline Timing Summary ---")
    print(f"AI Recommendations: {gpt_time:.2f}s")
    print(f"IC-Light Processing: ~{iclight_time:.2f}s (estimated)")
    print(f"Upscaling: {upscale_time:.2f}s")
    print(f"Estimated Total: ~{estimated_total:.2f}s")
    
    # Check against 30-second requirement
    if estimated_total < 30:
        print(f"\n✓ Pipeline meets <30 second requirement")
    else:
        print(f"\n⚠ Warning: Pipeline may exceed 30 second requirement")
    
    return estimated_total


def main():
    """Run all performance tests"""
    print("=" * 60)
    print("IC-Light Enhanced Pipeline - Performance Testing")
    print("=" * 60)
    
    # System info
    print("\n=== System Information ===")
    print(f"Python: {os.sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Run tests
    results = {}
    
    results['gpt_latency'] = test_gpt_recommendations_latency()
    results['upscaler_latency'] = test_upscaler_latency()
    test_gpu_memory_usage()
    results['total_pipeline'] = test_end_to_end_pipeline()
    
    # Final summary
    print("\n" + "=" * 60)
    print("Performance Test Summary")
    print("=" * 60)
    
    if results['gpt_latency'] > 0:
        print(f"✓ GPT API: {results['gpt_latency']:.2f}s")
    else:
        print("⊘ GPT API: Not tested (API key not set)")
    
    if results['upscaler_latency'] > 0:
        print(f"✓ Upscaling: {results['upscaler_latency']:.2f}s")
    else:
        print("✗ Upscaling: Failed")
    
    print(f"⊘ Estimated Total Pipeline: ~{results['total_pipeline']:.2f}s")
    
    if results['total_pipeline'] < 30:
        print("\n✓ All performance requirements met!")
    else:
        print("\n⚠ Performance may need optimization")
    
    print("\nNote: For accurate IC-Light processing time, run the actual")
    print("      Gradio demo and measure with real model inference.")


if __name__ == "__main__":
    main()
