# IC-Light Application - Quick Start Guide

## ✅ Your Application is Ready!

All issues have been fixed and the application is fully functional.

## 🚀 How to Start

### Option 1: Using the Launcher (Easiest)
```cmd
cd D:\new1\light
run_demo.bat
```
Then select:
- **1** for Text-Conditioned Relighting
- **2** for Background-Conditioned Relighting
- **3** to Run System Test

### Option 2: Direct Command
```cmd
cd D:\new1\light
D:\new1\venv_py310\Scripts\python.exe gradio_demo.py
```

## 🌐 Access the Application

Once started, open your browser to:
- **http://localhost:7860**
- Or: **http://0.0.0.0:7860**

## 📝 How to Use

### Text-Conditioned Relighting

1. **Upload Image** - Click the image upload area and select your image
2. **Get AI Recommendations** (Optional)
   - Click "✨ Get AI Recommendations"
   - Wait 2-5 seconds
   - Select one of the 3 suggestions
   - Click "Apply Selected Suggestion"
3. **Enter Prompt** - Type or modify your lighting description
4. **Select Lighting Preference** - Choose: None, Left, Right, Top, or Bottom
5. **Click "Relight"**
6. **Wait 20-30 seconds** - Progress bar will show:
   - Starting...
   - Removing background...
   - Processing relighting (10-20 seconds)...
   - Upscaling to high resolution...
   - Saving images...
   - Complete!
7. **View Results** - Images appear in the gallery on the right

### Background-Conditioned Relighting

1. **Upload Foreground Image**
2. **Click "Add Background Image"**
3. **Upload Background Image**
4. **Get AI Recommendations** (Optional)
5. **Click "Blend with Background"**
6. **Wait 20-30 seconds**
7. **View Results**

## 📁 Output Files

All generated images are automatically saved to:
```
D:\new1\light\outputs\
```

Files are named with timestamps:
- `relit_YYYYMMDD_HHMMSS_1.png` - Your final high-resolution image
- `temp_display_YYYYMMDD_HHMMSS_1.png` - Browser display version

## ⚡ What's Fixed

### ✅ No More Timeout Errors
- Frontend now waits indefinitely for processing
- Progress bar shows real-time updates
- No "Error" messages during processing

### ✅ Images Display Properly
- Images are saved as files and displayed via file paths
- Optimized for browser compatibility (max 1280p display)
- Full resolution saved to outputs folder

### ✅ SSL Certificate Fixed
- No more certificate errors
- Models download properly

### ✅ Environment Variables Loaded
- .env file is automatically loaded
- OpenAI API key works for AI recommendations

### ✅ Better Error Handling
- Clear error messages
- Graceful fallbacks
- Detailed logging

## 🎯 Expected Performance

### Processing Times
- Background removal: ~1-2 seconds
- IC-Light processing: ~10-15 seconds
- Upscaling: ~5-10 seconds
- **Total: 20-30 seconds per image**

### GPU Requirements
- Minimum: 8GB VRAM
- Recommended: 12GB+ VRAM
- Your GPU: NVIDIA GeForce RTX 4060 Laptop GPU ✅

### Image Sizes
- **Input**: Any size (will be resized)
- **Processing**: 512x640 (default, adjustable)
- **Output Display**: Max 1280p (for browser)
- **Output Saved**: Full resolution (1080p+)

## 🔧 Troubleshooting

### If Images Don't Display:
1. Check terminal output for errors
2. Check `D:\new1\light\outputs\` folder - images should be there
3. Refresh browser (Ctrl+F5)
4. Try with a smaller image first (512x512)

### If Processing is Slow:
1. Verify GPU is being used (check terminal for "CUDA available: True")
2. Reduce image dimensions in settings
3. Close other GPU-intensive applications

### If AI Recommendations Don't Work:
1. Check `.env` file has valid OPENAI_API_KEY
2. Verify internet connection
3. Default suggestions will be used automatically if API fails

### If Upscaling Fails:
1. Check `RealESRGAN_x4plus.pth` exists in `./models/` folder
2. System will automatically fall back to simple upscaling
3. Check terminal for specific error messages

## 📊 System Test

To verify everything is working:
```cmd
cd D:\new1\light
D:\new1\venv_py310\Scripts\python.exe test_system.py
```

This checks:
- ✅ Python environment
- ✅ All dependencies
- ✅ Model files
- ✅ Environment variables
- ✅ GPU availability
- ✅ Upscaler initialization
- ✅ GPT client initialization

## 📚 Documentation Files

- **FIXES_APPLIED.md** - Complete list of all fixes
- **TIMEOUT_FIXES.md** - Detailed timeout fix explanation
- **START_HERE.md** - This file
- **README.md** - Original IC-Light documentation

## 🎨 Features

### Modern UI
- Dark theme with gradient backgrounds
- Smooth animations and hover effects
- Professional typography
- Responsive design

### AI-Powered Recommendations
- GPT-4o-mini analyzes your image
- Suggests 3 contextual lighting styles
- One-click application to prompt

### Automatic Upscaling
- All outputs upscaled to minimum 1080p
- Uses Real-ESRGAN for high quality
- Maintains aspect ratio

### Background Blending
- Upload custom backgrounds
- Automatic lighting harmonization
- Professional compositing

## 💡 Tips for Best Results

### For Portraits:
- Use prompts like: "warm golden hour", "soft studio lighting"
- Try Left or Right lighting preference
- Use background blending for realistic scenes

### For Products:
- Use prompts like: "professional product lighting", "clean white background"
- Try Top or Bottom lighting
- Experiment with different lighting preferences

### For Creative Effects:
- Use prompts like: "neon cyberpunk", "dramatic side lighting"
- Try multiple lighting preferences
- Use AI recommendations for inspiration

## 🚨 Important Notes

1. **Processing Time**: Be patient! 20-30 seconds is normal
2. **Progress Bar**: Watch the progress bar for status updates
3. **Output Folder**: Check outputs folder if images don't display
4. **GPU Memory**: Close other applications if you get memory errors
5. **Image Size**: Larger input images take longer to process

## ✨ Success Indicators

When everything is working correctly, you should see:

1. ✅ Application starts without errors
2. ✅ Browser opens to http://localhost:7860
3. ✅ Progress bar appears when processing
4. ✅ Progress updates every few seconds
5. ✅ Images appear in gallery after 20-30 seconds
6. ✅ No "Error" messages
7. ✅ Images saved to outputs folder
8. ✅ Terminal shows detailed processing logs

## 🎉 You're All Set!

Your IC-Light application is fully functional with:
- ✅ No timeout errors
- ✅ Proper image display
- ✅ Real-time progress updates
- ✅ AI-powered recommendations
- ✅ Automatic upscaling
- ✅ Background blending
- ✅ Modern UI design

**Start the application and enjoy creating amazing relit images!**

---

**Need Help?**
- Check terminal output for detailed logs
- Review TIMEOUT_FIXES.md for technical details
- Run test_system.py to verify setup
- Check outputs folder for saved images
