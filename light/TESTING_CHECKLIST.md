# IC-Light Enhanced Features - Testing Checklist

This document provides a comprehensive checklist for manually testing all enhanced features.

## Prerequisites

- [ ] Python environment activated
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] OPENAI_API_KEY environment variable set
- [ ] Real-ESRGAN model at `./models/RealESRGAN_x4plus.pth`
- [ ] CUDA-compatible GPU available (recommended)

## 8.1 Complete Workflow Testing

### Test Case 1: Basic Workflow with AI Recommendations

**Steps:**
1. [ ] Start gradio_demo.py: `python gradio_demo.py`
2. [ ] Upload a test image (foreground)
3. [ ] Click "✨ Get AI Recommendations" button
4. [ ] Verify 3 suggestions appear (each ≤ 3 words)
5. [ ] Select a suggestion from the radio buttons
6. [ ] Click "Apply Selected Suggestion"
7. [ ] Verify prompt field is populated with the suggestion
8. [ ] Click "Relight" button
9. [ ] Wait for processing to complete
10. [ ] Verify output image appears in gallery
11. [ ] Download output image and check dimensions (height ≥ 1080px)

**Expected Results:**
- AI suggestions are contextually relevant to the image
- Prompt field updates correctly
- Output image is generated successfully
- Output image height is at least 1080 pixels
- Processing completes in < 30 seconds

**Test with various image sizes:**
- [ ] Small image (256x256)
- [ ] Medium image (512x512)
- [ ] Large image (1024x1024)
- [ ] Already 1080p image (1920x1080)

**Test with various formats:**
- [ ] JPEG
- [ ] PNG
- [ ] WebP (if supported)

### Test Case 2: Manual Prompt Entry

**Steps:**
1. [ ] Upload an image
2. [ ] Manually type a prompt (e.g., "warm golden hour")
3. [ ] Select a lighting preference (e.g., "Left Light")
4. [ ] Click "Relight"
5. [ ] Verify output is generated and upscaled

**Expected Results:**
- Manual prompts work without AI recommendations
- Lighting preference affects the output
- Upscaling still applies

### Test Case 3: Quick Prompt Lists

**Steps:**
1. [ ] Upload an image
2. [ ] Click a prompt from "Subject Quick List"
3. [ ] Verify prompt field updates
4. [ ] Click a prompt from "Lighting Quick List"
5. [ ] Verify prompt field appends the lighting
6. [ ] Click "Relight"

**Expected Results:**
- Quick prompts populate correctly
- Multiple quick prompts combine properly

## 8.2 Background Blending Workflow

### Test Case 4: Background Blending with AI Recommendations

**Steps:**
1. [ ] Start gradio_demo_bg.py: `python gradio_demo_bg.py`
2. [ ] Upload a foreground image
3. [ ] Click "Add Background Image" button
4. [ ] Verify background upload component appears
5. [ ] Verify "Blend with Background" button appears
6. [ ] Upload a background image
7. [ ] Click "✨ Get AI Recommendations"
8. [ ] Select and apply a suggestion
9. [ ] Click "Blend with Background"
10. [ ] Verify blended output appears
11. [ ] Check output dimensions (height ≥ 1080px)

**Expected Results:**
- Background upload UI shows/hides correctly
- AI recommendations work in background mode
- Blending produces natural-looking composite
- Lighting is harmonized between foreground and background
- Output is upscaled to 1080p

### Test Case 5: Background Source Options

**Steps:**
1. [ ] Upload foreground image
2. [ ] Test each background source option:
   - [ ] "Use Background Image" (with uploaded background)
   - [ ] "Use Flipped Background Image"
   - [ ] "Left Light"
   - [ ] "Right Light"
   - [ ] "Top Light"
   - [ ] "Bottom Light"
   - [ ] "Ambient"
3. [ ] Verify each produces different lighting effects

**Expected Results:**
- All background source options work
- Each option produces distinct lighting

### Test Case 6: Compute Normal

**Steps:**
1. [ ] Upload foreground and background
2. [ ] Enter a prompt
3. [ ] Click "Compute Normal (4x Slower)"
4. [ ] Wait for processing (will take longer)
5. [ ] Verify normal map outputs appear

**Expected Results:**
- Normal computation completes successfully
- Multiple output images show different lighting angles
- All outputs are upscaled

## 8.3 Error Handling Testing

### Test Case 7: Invalid API Key

**Steps:**
1. [ ] Set invalid OPENAI_API_KEY: `set OPENAI_API_KEY=invalid_key`
2. [ ] Start gradio_demo.py
3. [ ] Upload an image
4. [ ] Click "✨ Get AI Recommendations"
5. [ ] Observe behavior

**Expected Results:**
- Error is caught gracefully
- Default suggestions are provided OR
- User-friendly error message is displayed
- Application does not crash

### Test Case 8: Missing API Key

**Steps:**
1. [ ] Unset OPENAI_API_KEY: `set OPENAI_API_KEY=`
2. [ ] Start gradio_demo.py
3. [ ] Upload an image
4. [ ] Click "✨ Get AI Recommendations"

**Expected Results:**
- Error is handled gracefully
- Default suggestions provided or clear error message
- Application continues to work for other features

### Test Case 9: Missing Upscaler Model

**Steps:**
1. [ ] Temporarily rename `./models/RealESRGAN_x4plus.pth`
2. [ ] Start gradio_demo.py
3. [ ] Observe startup behavior
4. [ ] Try to relight an image

**Expected Results:**
- Clear error message about missing model OR
- Fallback to non-upscaled output
- Application doesn't crash

### Test Case 10: Network Errors

**Steps:**
1. [ ] Disconnect from internet
2. [ ] Try to get AI recommendations

**Expected Results:**
- Network error is caught
- Default suggestions provided
- User-friendly error message

### Test Case 11: Invalid Image Upload

**Steps:**
1. [ ] Try uploading a non-image file
2. [ ] Try uploading a corrupted image

**Expected Results:**
- Appropriate error message
- Application remains stable

## 8.4 Performance Testing

### Test Case 12: End-to-End Latency

**Steps:**
1. [ ] Upload a 512x512 image
2. [ ] Get AI recommendations (measure time)
3. [ ] Apply suggestion
4. [ ] Click "Relight" and measure total time from click to output

**Expected Results:**
- AI recommendations: < 10 seconds
- Relighting + upscaling: < 30 seconds total
- Total workflow: < 40 seconds

**Measure times for:**
- [ ] Small image (256x256): _____ seconds
- [ ] Medium image (512x512): _____ seconds
- [ ] Large image (1024x1024): _____ seconds

### Test Case 13: GPU Memory Usage

**Steps:**
1. [ ] Monitor GPU memory before starting
2. [ ] Start gradio_demo.py
3. [ ] Monitor GPU memory after model loading
4. [ ] Process an image
5. [ ] Monitor peak GPU memory usage

**Expected Results:**
- Models load successfully
- GPU memory usage is reasonable (< 12GB for optimal)
- No memory leaks after multiple operations

**Record measurements:**
- Idle GPU memory: _____ GB
- After model loading: _____ GB
- Peak during processing: _____ GB

### Test Case 14: Multiple Sequential Operations

**Steps:**
1. [ ] Process 5 images in sequence
2. [ ] Monitor memory usage
3. [ ] Verify no performance degradation

**Expected Results:**
- Each operation completes successfully
- No memory leaks
- Consistent performance across operations

## 8.5 UI/UX Testing

### Test Case 15: Modern UI Styling

**Steps:**
1. [ ] Start both gradio_demo.py and gradio_demo_bg.py
2. [ ] Verify dark theme is applied
3. [ ] Check button styling (gradients, hover effects)
4. [ ] Verify typography is consistent
5. [ ] Check image container styling
6. [ ] Verify accordion styling

**Expected Results:**
- Consistent dark theme across both demos
- Modern gradient buttons with hover effects
- Clean, professional appearance
- All UI elements properly styled

### Test Case 16: Component Visibility

**Steps:**
1. [ ] Verify AI suggestions radio is hidden initially
2. [ ] Click "Get AI Recommendations"
3. [ ] Verify radio appears
4. [ ] Verify "Apply" button appears
5. [ ] In background demo, verify background upload is hidden initially
6. [ ] Click "Add Background Image"
7. [ ] Verify background upload and blend button appear

**Expected Results:**
- All show/hide logic works correctly
- Smooth transitions
- No layout shifts

### Test Case 17: Interactive Light Pointer (Optional)

**Steps:**
1. [ ] Start gradio_demo.py
2. [ ] Upload an image
3. [ ] Look for golden light pointer overlay
4. [ ] Drag the pointer to different positions
5. [ ] Verify direction label updates
6. [ ] Verify lighting preference radio updates automatically

**Expected Results:**
- Light pointer appears on image
- Dragging is smooth
- Direction detection is accurate
- Lighting preference updates in real-time

## Cross-Feature Integration

### Test Case 18: Feature Combination

**Steps:**
1. [ ] Use AI recommendations + manual prompt editing
2. [ ] Use AI recommendations + quick prompts
3. [ ] Use light pointer + AI recommendations
4. [ ] Switch between text-conditioned and background-conditioned modes

**Expected Results:**
- All features work together seamlessly
- No conflicts or errors
- State is maintained correctly

## Test Results Summary

**Date:** _______________
**Tester:** _______________
**Environment:** _______________

**Overall Results:**
- Total test cases: 18
- Passed: _____
- Failed: _____
- Skipped: _____

**Critical Issues Found:**
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________

**Minor Issues Found:**
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________

**Performance Notes:**
_______________________________________________
_______________________________________________
_______________________________________________

**Recommendations:**
_______________________________________________
_______________________________________________
_______________________________________________
