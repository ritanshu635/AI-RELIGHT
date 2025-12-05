# 🖼️ Background Image Feature - Complete Guide

## 🎉 NEW FEATURE: Custom Background Support!

Your IC-Light app now supports **background-conditioned relighting**! You can upload a custom background image and harmonize your foreground subject with the background's lighting.

---

## 🚀 What's New?

### **Before (Text-Only Mode)**
- Upload main image
- Set lighting direction (Left, Right, Top, Bottom, None)
- Uses gradient backgrounds for lighting

### **After (Background Mode)**
- ✅ Upload main image (foreground)
- ✅ Upload background image (optional)
- ✅ Automatic background removal from main image
- ✅ Lighting harmonization between foreground and background
- ✅ Still supports gradient lighting if no background provided

---

## 📁 New Files Created

### **1. `app_with_background.py`**
- New Flask app with background support
- Uses `iclight_sd15_fbc.safetensors` model (12-channel)
- Handles both foreground and background conditioning

### **2. `templates/index_with_background.html`**
- New UI with background upload area
- Dual upload zones (main + background)
- Background preview
- All existing features preserved

---

## 🎯 How to Use

### **Step 1: Start the New App**

```powershell
cd light
python app_with_background.py
```

**Access:** http://localhost:5000

### **Step 2: Upload Images**

#### **Main Image (Required)**
1. Click or drag your main image to the first upload area
2. This will be your foreground subject
3. Background will be automatically removed

#### **Background Image (Optional)**
1. Click or drag a background image to the second upload area
2. This provides the lighting reference
3. Leave empty to use gradient lighting

### **Step 3: Configure & Relight**

1. **Select mood tags** (optional)
2. **Get AI recommendations** (optional)
3. **Edit prompt** manually if desired
4. **Drag light source** or use dropdown
5. **Click "Relight Image"**
6. Wait 20-30 seconds
7. **Download** your result!

---

## 🎨 Use Cases

### **1. Portrait with Custom Background**
```
Main Image: Portrait photo
Background: Sunset landscape
Result: Portrait lit as if in that sunset scene
```

### **2. Product Photography**
```
Main Image: Product on white background
Background: Lifestyle scene (kitchen, office, etc.)
Result: Product harmonized with scene lighting
```

### **3. Creative Compositing**
```
Main Image: Person
Background: Fantasy landscape
Result: Person lit to match fantasy environment
```

### **4. Gradient Lighting (No Background)**
```
Main Image: Any subject
Background: None (leave empty)
Lighting: Left/Right/Top/Bottom
Result: Gradient-based directional lighting
```

---

## 🔧 Technical Details

### **Model Differences**

| Feature | Old Model (fc) | New Model (fbc) |
|---------|---------------|-----------------|
| **File** | `iclight_sd15_fc.safetensors` | `iclight_sd15_fbc.safetensors` |
| **Channels** | 8 (4 RGB + 4 FG) | 12 (4 RGB + 4 FG + 4 BG) |
| **Background** | Gradient only | Custom image supported |
| **Use Case** | Text-conditioned | Text + Background conditioned |

### **Processing Pipeline**

```
1. Upload main image
   ↓
2. Remove background (BRIA RMBG)
   ↓
3. Upload background image (optional)
   ↓
4. Encode foreground to latent space
   ↓
5. Encode background to latent space
   ↓
6. Concatenate FG + BG conditioning (12 channels)
   ↓
7. Run diffusion with combined conditioning
   ↓
8. Decode to image
   ↓
9. Upscale to 1080p
   ↓
10. Display result
```

### **Memory Requirements**

- **Model Size**: ~2GB (iclight_sd15_fbc.safetensors)
- **VRAM Usage**: ~8-10GB (with background)
- **Processing Time**: 20-30 seconds per image

---

## 🎨 Examples

### **Example 1: Portrait with Sunset**

**Input:**
- Main: Portrait photo (any background)
- Background: Sunset landscape

**Settings:**
- Prompt: "warm golden hour lighting"
- Mood: Romantic, Cinematic

**Result:**
- Portrait lit with warm sunset tones
- Shadows match sunset direction
- Color temperature harmonized

### **Example 2: Product in Kitchen**

**Input:**
- Main: Coffee mug on white background
- Background: Modern kitchen scene

**Settings:**
- Prompt: "natural window light"
- Mood: Household

**Result:**
- Mug lit as if in that kitchen
- Reflections match environment
- Realistic integration

### **Example 3: Fantasy Character**

**Input:**
- Main: Person in costume
- Background: Fantasy forest

**Settings:**
- Prompt: "magical forest lighting, ethereal glow"
- Mood: Mysterious, Dramatic

**Result:**
- Character lit with forest ambiance
- Magical atmosphere
- Seamless integration

---

## 🆚 Comparison: With vs Without Background

### **Without Background Image**
```
Main Image: Portrait
Background: None
Lighting: Left Light
Result: Portrait with left-side gradient lighting
```

### **With Background Image**
```
Main Image: Portrait
Background: Beach sunset photo
Lighting: Ignored (uses background)
Result: Portrait lit as if at that beach sunset
```

---

## 🎯 Best Practices

### **Choosing Background Images**

✅ **Good Backgrounds:**
- Clear lighting direction
- Consistent lighting quality
- Similar perspective to foreground
- High resolution

❌ **Avoid:**
- Cluttered scenes
- Mixed lighting sources
- Very dark or very bright
- Low resolution

### **Foreground Preparation**

✅ **Best Results:**
- Clean subject separation
- Good lighting on subject
- High resolution
- Clear edges

### **Prompt Tips**

- Describe the lighting quality: "soft", "dramatic", "warm"
- Reference the background: "beach sunset lighting"
- Add atmosphere: "golden hour", "moody", "bright"
- Keep it concise: 3-10 words

---

## 🔄 Switching Between Modes

### **Use Background Mode When:**
- You have a specific background in mind
- You want realistic lighting harmonization
- You're doing compositing work
- You need environmental lighting

### **Use Text-Only Mode When:**
- You want simple directional lighting
- You don't have a background image
- You want gradient-based lighting
- You want faster processing

### **How to Switch:**

**To Background Mode:**
```powershell
python app_with_background.py
```

**To Text-Only Mode:**
```powershell
python app.py
```

---

## 📊 Feature Comparison

| Feature | Text-Only (`app.py`) | Background (`app_with_background.py`) |
|---------|---------------------|--------------------------------------|
| **Main Image Upload** | ✅ | ✅ |
| **Background Upload** | ❌ | ✅ |
| **Gradient Lighting** | ✅ | ✅ |
| **Custom Background** | ❌ | ✅ |
| **AI Recommendations** | ✅ | ✅ |
| **Mood Tags** | ✅ | ✅ |
| **Draggable Light** | ✅ | ✅ |
| **Manual Prompts** | ✅ | ✅ |
| **Upscaling** | ✅ | ✅ |
| **Processing Time** | 20-30s | 20-30s |
| **VRAM Usage** | 6-8GB | 8-10GB |

---

## 🐛 Troubleshooting

### **Background image not affecting result**
- Ensure you uploaded to the background area (second upload zone)
- Check that background image loaded (preview should show)
- Try a background with clearer lighting

### **Out of memory error**
- Background mode uses more VRAM
- Try smaller images
- Close other GPU applications
- Use text-only mode instead

### **Background looks wrong**
- Check background image quality
- Ensure lighting is clear in background
- Try different background images
- Adjust prompt to describe background lighting

### **Model not downloading**
- Check internet connection
- Model will auto-download on first run
- File: `iclight_sd15_fbc.safetensors` (~2GB)
- Location: `light/models/`

---

## 🎓 Advanced Tips

### **Lighting Harmonization**

The model learns to:
- Match color temperature
- Align shadow directions
- Harmonize light intensity
- Blend ambient occlusion

### **Best Background Types**

1. **Outdoor scenes** - Clear sun direction
2. **Studio setups** - Controlled lighting
3. **Interior spaces** - Window light
4. **Dramatic scenes** - Strong contrast

### **Combining with AI Recommendations**

1. Upload main + background images
2. Select mood tags matching background
3. Get AI recommendations
4. AI will suggest prompts that work with background
5. Apply and relight!

---

## 📝 Quick Reference

### **Keyboard Shortcuts**
- None currently (all mouse/touch based)

### **File Formats**
- **Input**: JPG, PNG (max 50MB each)
- **Output**: PNG (high quality)

### **Recommended Sizes**
- **Main Image**: Any size (will be resized)
- **Background**: Any size (will be resized)
- **Output**: Minimum 1080p (upscaled)

---

## 🚀 Getting Started Checklist

- [ ] Stop old Flask server
- [ ] Run `python app_with_background.py`
- [ ] Visit http://localhost:5000
- [ ] Upload main image
- [ ] Upload background image (optional)
- [ ] Configure settings
- [ ] Click "Relight Image"
- [ ] Download result!

---

## 🎉 Summary

**You now have TWO powerful modes:**

1. **Text-Only Mode** (`app.py`)
   - Simple, fast
   - Gradient lighting
   - Perfect for basic relighting

2. **Background Mode** (`app_with_background.py`)
   - Advanced, realistic
   - Custom backgrounds
   - Perfect for compositing

**Choose the mode that fits your needs!** 🎨✨

---

## 🌐 Access Your Apps

| Mode | Command | URL |
|------|---------|-----|
| **Text-Only** | `python app.py` | http://localhost:5000 |
| **Background** | `python app_with_background.py` | http://localhost:5000 |
| **Spooky Landing** | `npm run dev` | http://localhost:8080 |

---

**Enjoy your new background-conditioned relighting feature!** 🖼️💡✨
