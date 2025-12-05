"""
Integration tests for IC-Light enhanced features
Tests the complete workflow including AI recommendations, upscaling, and background blending
"""

import os
import sys
import time
import numpy as np
from PIL import Image
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    try:
        from gpt_recommendations import GPTRecommendationClient
        from upscaler import ImageUpscaler
        print("✓ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_gpt_client():
    """Test GPT recommendations client"""
    print("\nTesting GPT recommendations client...")
    try:
        from gpt_recommendations import GPTRecommendationClient
        
        # Check if API key is set
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("⚠ OPENAI_API_KEY not set, skipping GPT tests")
            return True
        
        client = GPTRecommendationClient()
        
        # Create a test image
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        # Test image to base64 conversion
        base64_str = client.image_to_base64(test_image)
        assert isinstance(base64_str, str), "Base64 conversion failed"
        print("✓ Image to base64 conversion works")
        
        # Test getting recommendations (with timeout)
        print("  Testing API call (this may take a few seconds)...")
        start_time = time.time()
        suggestions = client.get_lighting_recommendations(test_image, timeout=15)
        elapsed = time.time() - start_time
        
        assert isinstance(suggestions, list), "Suggestions should be a list"
        assert len(suggestions) == 3, f"Expected 3 suggestions, got {len(suggestions)}"
        
        for i, suggestion in enumerate(suggestions):
            word_count = len(suggestion.split())
            print(f"  Suggestion {i+1}: '{suggestion}' ({word_count} words)")
            assert word_count <= 3, f"Suggestion '{suggestion}' exceeds 3 words"
        
        print(f"✓ GPT recommendations work (took {elapsed:.2f}s)")
        return True
        
    except Exception as e:
        print(f"✗ GPT client test failed: {e}")
        return False


def test_upscaler():
    """Test Real-ESRGAN upscaler"""
    print("\nTesting Real-ESRGAN upscaler...")
    try:
        from upscaler import ImageUpscaler
        
        # Check if model exists
        model_path = './models/RealESRGAN_x4plus.pth'
        if not os.path.exists(model_path):
            print(f"⚠ Model not found at {model_path}, skipping upscaler tests")
            return True
        
        # Initialize upscaler
        print("  Initializing upscaler...")
        upscaler = ImageUpscaler(model_path=model_path, device='cuda' if torch.cuda.is_available() else 'cpu')
        print("✓ Upscaler initialized")
        
        # Test with small image (should upscale)
        print("  Testing upscaling of 256x256 image...")
        small_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        start_time = time.time()
        upscaled = upscaler.upscale_to_1080p(small_image)
        elapsed = time.time() - start_time
        
        h, w = upscaled.shape[:2]
        print(f"  Input: 256x256 → Output: {w}x{h} (took {elapsed:.2f}s)")
        assert h >= 1080, f"Output height {h} is less than 1080p"
        print("✓ Small image upscaled to 1080p")
        
        # Test with already large image (should not upscale)
        print("  Testing with 1920x1080 image...")
        large_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        start_time = time.time()
        result = upscaler.upscale_to_1080p(large_image)
        elapsed = time.time() - start_time
        
        h, w = result.shape[:2]
        print(f"  Input: 1920x1080 → Output: {w}x{h} (took {elapsed:.2f}s)")
        assert h >= 1080, f"Output height {h} is less than 1080p"
        print("✓ Large image handled correctly")
        
        # Test aspect ratio preservation
        print("  Testing aspect ratio preservation...")
        test_image = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
        upscaled = upscaler.upscale_to_1080p(test_image)
        
        original_ratio = 600 / 400
        upscaled_ratio = upscaled.shape[1] / upscaled.shape[0]
        ratio_diff = abs(original_ratio - upscaled_ratio)
        
        print(f"  Original ratio: {original_ratio:.3f}, Upscaled ratio: {upscaled_ratio:.3f}, Diff: {ratio_diff:.3f}")
        assert ratio_diff < 0.01, f"Aspect ratio not preserved (diff: {ratio_diff})"
        print("✓ Aspect ratio preserved")
        
        return True
        
    except Exception as e:
        print(f"✗ Upscaler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling scenarios"""
    print("\nTesting error handling...")
    try:
        from gpt_recommendations import GPTRecommendationClient
        from upscaler import ImageUpscaler
        
        # Test GPT client with invalid API key
        print("  Testing GPT client with invalid API key...")
        os.environ['OPENAI_API_KEY'] = 'invalid_key_12345'
        client = GPTRecommendationClient()
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        try:
            suggestions = client.get_lighting_recommendations(test_image, timeout=5)
            # Should return default suggestions on error
            assert len(suggestions) == 3, "Should return 3 default suggestions on error"
            print("✓ GPT client handles invalid API key gracefully")
        except ValueError as e:
            if "Invalid OpenAI API key" in str(e):
                print("✓ GPT client raises appropriate error for invalid API key")
            else:
                raise
        
        # Test upscaler with missing model
        print("  Testing upscaler with missing model...")
        try:
            upscaler = ImageUpscaler(model_path='./models/nonexistent_model.pth', device='cpu')
            print("⚠ Upscaler should fail with missing model")
        except Exception:
            print("✓ Upscaler handles missing model appropriately")
        
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False


def test_performance():
    """Test performance metrics"""
    print("\nTesting performance...")
    try:
        from upscaler import ImageUpscaler
        
        model_path = './models/RealESRGAN_x4plus.pth'
        if not os.path.exists(model_path):
            print("⚠ Model not found, skipping performance tests")
            return True
        
        upscaler = ImageUpscaler(model_path=model_path, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Test upscaling performance
        test_image = np.random.randint(0, 255, (512, 640, 3), dtype=np.uint8)
        
        print("  Running 3 upscaling iterations...")
        times = []
        for i in range(3):
            start_time = time.time()
            _ = upscaler.upscale_to_1080p(test_image)
            elapsed = time.time() - start_time
            times.append(elapsed)
            print(f"  Iteration {i+1}: {elapsed:.2f}s")
        
        avg_time = sum(times) / len(times)
        print(f"  Average upscaling time: {avg_time:.2f}s")
        
        # Check if within reasonable time (30 seconds for full pipeline)
        # Upscaling should be < 10 seconds
        if avg_time < 10:
            print("✓ Upscaling performance is good")
        else:
            print(f"⚠ Upscaling is slow ({avg_time:.2f}s), may need optimization")
        
        # Check GPU memory if available
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"  GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
            print("✓ GPU memory usage checked")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        return False


def run_all_tests():
    """Run all integration tests"""
    print("=" * 60)
    print("IC-Light Enhanced Features - Integration Tests")
    print("=" * 60)
    
    results = {
        "Imports": test_imports(),
        "GPT Client": test_gpt_client(),
        "Upscaler": test_upscaler(),
        "Error Handling": test_error_handling(),
        "Performance": test_performance()
    }
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠ {total_tests - passed_tests} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
