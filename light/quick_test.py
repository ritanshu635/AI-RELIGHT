"""Quick test to verify image processing and display"""
import numpy as np
from PIL import Image
import os

print("Testing image format for Gradio Gallery...")

# Create a test image
test_img = np.random.randint(0, 255, (1280, 1024, 3), dtype=np.uint8)
print(f"Test image shape: {test_img.shape}")
print(f"Test image dtype: {test_img.dtype}")
print(f"Test image min/max: {test_img.min()}/{test_img.max()}")

# Save it
output_dir = "./outputs"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "test_image.png")
Image.fromarray(test_img).save(output_path)
print(f"Saved test image to: {output_path}")

# Verify it can be loaded
loaded = np.array(Image.open(output_path))
print(f"Loaded image shape: {loaded.shape}")
print(f"Loaded image dtype: {loaded.dtype}")

print("\n✓ Image format test passed!")
print("\nNow restart the Gradio app and try uploading an image.")
