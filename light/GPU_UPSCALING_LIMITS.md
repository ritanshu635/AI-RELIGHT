# GPU Upscaling Limits - RTX 4060 8GB

## Tested Configurations

Based on testing with NVIDIA GeForce RTX 4060 (8GB VRAM):

### Upscaling (Below 1080p)
- **480p (480x854)**: ✅ 3.41 GB VRAM → Upscales to 1920x3416
- **720p (720x1280)**: ✅ 7.65 GB VRAM → Upscales to 2880x5120 (close to 8GB limit)

### No Upscaling (Already ≥1080p)
- **1080p+**: ✅ Returns original image unchanged (0.12 GB VRAM - minimal usage)

## Current Behavior

The `upscale_to_1080p()` method implements simple logic:

1. **Check height**: If image height ≥ 1080px → return original (no processing)
2. **Upscale**: If image height < 1080px → upscale to at least 1080p using Real-ESRGAN 4x
3. **Error handling**: If upscaling fails (OOM or other error) → return original image

## Recommendations

For your 8GB RTX 4060:
- ✅ **Safe**: Images up to 720p upscale successfully
- ⚠️ **Risk**: Images between 720p-1080p may cause OOM errors (will return original if failed)
- ✅ **Optimal**: Images already ≥1080p are returned unchanged instantly

## Error Handling

If GPU runs out of memory during upscaling:
- The original image is returned unchanged
- An error message is printed to console
- No crash or exception propagates to caller

## Testing

Run tests:
```bash
python -m unittest test_upscaler.py -v
python test_gpu_limits.py
```

All tests pass successfully!
