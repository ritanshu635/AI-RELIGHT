import unittest
import numpy as np
from PIL import Image
import os
from upscaler import ImageUpscaler


class TestImageUpscaler(unittest.TestCase):
    """Unit tests for ImageUpscaler class"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        # Check if model exists
        cls.model_path = './models/RealESRGAN_x4plus.pth'
        cls.model_exists = os.path.exists(cls.model_path)
        
        if cls.model_exists:
            # Initialize upscaler with actual model
            try:
                cls.upscaler = ImageUpscaler(model_path=cls.model_path, device='cuda')
            except Exception as e:
                print(f"Failed to initialize with CUDA, trying CPU: {e}")
                cls.upscaler = ImageUpscaler(model_path=cls.model_path, device='cpu')
        else:
            print(f"Warning: Model not found at {cls.model_path}")
            cls.upscaler = None
    
    def create_test_image(self, height, width):
        """Helper method to create a test image"""
        # Create a simple gradient image
        image = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                image[i, j] = [i % 256, j % 256, (i + j) % 256]
        return image
    
    def test_upscale_image_below_1080p(self):
        """Test upscaling images below 1080p"""
        if not self.model_exists or self.upscaler is None:
            self.skipTest("Model not available")
        
        # Create a 480p image (854x480)
        test_image = self.create_test_image(480, 854)
        
        # Upscale
        result = self.upscaler.upscale_to_1080p(test_image)
        
        # Verify output is at least 1080p
        self.assertIsInstance(result, np.ndarray)
        self.assertGreaterEqual(result.shape[0], 1080, 
                                f"Output height {result.shape[0]} is less than 1080")
        self.assertEqual(result.shape[2], 3, "Output should have 3 channels")
        
        print(f"480p upscaled to: {result.shape[0]}x{result.shape[1]}")
    
    def test_upscale_image_already_above_1080p(self):
        """Test that images already above 1080p are not upscaled"""
        if not self.model_exists or self.upscaler is None:
            self.skipTest("Model not available")
        
        # Create an image that's already above 1080p (1200x1920)
        test_image = self.create_test_image(1200, 1920)
        
        # Call upscale (should return original without processing)
        result = self.upscaler.upscale_to_1080p(test_image)
        
        # Verify output is unchanged (same dimensions)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[0], 1200, "Height should remain unchanged")
        self.assertEqual(result.shape[1], 1920, "Width should remain unchanged")
        self.assertEqual(result.shape[2], 3)
        
        # Verify it's the same array (no processing occurred)
        np.testing.assert_array_equal(result, test_image, "Image should be unchanged")
        
        print(f"1200p image returned unchanged: {result.shape[0]}x{result.shape[1]}")
    
    def test_aspect_ratio_preservation(self):
        """Test that aspect ratio is preserved during upscaling"""
        if not self.model_exists or self.upscaler is None:
            self.skipTest("Model not available")
        
        # Create a 16:9 image at 720p (1280x720)
        test_image = self.create_test_image(720, 1280)
        original_aspect_ratio = 1280 / 720
        
        # Upscale
        result = self.upscaler.upscale_to_1080p(test_image)
        
        # Calculate result aspect ratio
        result_aspect_ratio = result.shape[1] / result.shape[0]
        
        # Verify aspect ratio is preserved (within 1% tolerance)
        aspect_ratio_diff = abs(result_aspect_ratio - original_aspect_ratio)
        self.assertLess(aspect_ratio_diff, 0.01 * original_aspect_ratio,
                       f"Aspect ratio changed from {original_aspect_ratio:.4f} to {result_aspect_ratio:.4f}")
        
        print(f"Original aspect ratio: {original_aspect_ratio:.4f}")
        print(f"Result aspect ratio: {result_aspect_ratio:.4f}")
    
    def test_input_format_float32(self):
        """Test handling of float32 input images"""
        if not self.model_exists or self.upscaler is None:
            self.skipTest("Model not available")
        
        # Create a float32 image (normalized to 0-1)
        test_image = self.create_test_image(480, 640).astype(np.float32) / 255.0
        
        # Upscale
        result = self.upscaler.upscale_to_1080p(test_image)
        
        # Verify output
        self.assertIsInstance(result, np.ndarray)
        self.assertGreaterEqual(result.shape[0], 1080)
        self.assertEqual(result.dtype, np.uint8, "Output should be uint8")
    
    def test_very_small_image(self):
        """Test upscaling very small images that need more than 4x scaling"""
        if not self.model_exists or self.upscaler is None:
            self.skipTest("Model not available")
        
        # Create a very small image (200x300)
        test_image = self.create_test_image(200, 300)
        
        # Upscale
        result = self.upscaler.upscale_to_1080p(test_image)
        
        # Verify output meets 1080p requirement
        self.assertGreaterEqual(result.shape[0], 1080,
                               f"Output height {result.shape[0]} is less than 1080")
        
        print(f"200x300 upscaled to: {result.shape[0]}x{result.shape[1]}")


if __name__ == '__main__':
    unittest.main()
