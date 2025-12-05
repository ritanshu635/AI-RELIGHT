# New AI Recommendation Features! 🎨

## What's New

### Mood-Based AI Recommendations

Now you can guide the AI by selecting mood/style tags before getting recommendations!

## How It Works

### Step 1: Upload Your Image
- Drag & drop or click to upload

### Step 2: Select Mood Tags (Optional)
Choose from 9 different moods to guide the AI:

1. **👻 Haunted** - Spooky, eerie, ghostly atmosphere
2. **🗺️ Adventure** - Exciting, exploratory, journey vibes
3. **🏠 Household** - Cozy, domestic, home atmosphere
4. **🌲 Outdoor** - Natural, exterior, landscape lighting
5. **🎨 Playful** - Fun, colorful, whimsical mood
6. **🎭 Dramatic** - Intense, theatrical, bold lighting
7. **💕 Romantic** - Soft, warm, intimate atmosphere
8. **🔮 Mysterious** - Enigmatic, shadowy, intriguing
9. **🎬 Cinematic** - Movie-like, professional, epic

**You can select multiple moods!** For example:
- Haunted + Dramatic = Spooky dramatic lighting
- Romantic + Outdoor = Soft natural lighting
- Cinematic + Mysterious = Film noir style

### Step 3: Get AI Recommendations
- Click "✨ Get AI Recommendations"
- AI analyzes your image + selected moods
- Returns 3 lighting suggestions (2-3 words each)

### Step 4: Apply All & Relight
- Review the 3 suggestions
- Click "✨ Apply All & Relight"
- All 3 suggestions are combined (comma-separated)
- Relighting starts automatically!

## Example Workflow

### Example 1: Haunted Portrait
1. Upload a portrait photo
2. Select: **👻 Haunted** + **🎭 Dramatic**
3. Click "Get AI Recommendations"
4. AI might suggest:
   - window light
   - haunted atmosphere
   - shadow play
5. Click "Apply All & Relight"
6. Prompt becomes: "window light, haunted atmosphere, shadow play"
7. Image is relit with spooky dramatic lighting!

### Example 2: Outdoor Adventure
1. Upload a landscape photo
2. Select: **🗺️ Adventure** + **🌲 Outdoor**
3. AI might suggest:
   - golden hour
   - adventure glow
   - natural light
4. Apply all → "golden hour, adventure glow, natural light"

### Example 3: Romantic Indoor
1. Upload an indoor photo
2. Select: **💕 Romantic** + **🏠 Household**
3. AI might suggest:
   - warm candle
   - soft window
   - cozy glow
4. Apply all → "warm candle, soft window, cozy glow"

## Technical Details

### How Moods Affect AI
- Moods are sent to GPT-4o-mini along with the image
- GPT considers both the image content AND your selected moods
- Suggestions are tailored to match your vision

### Suggestion Format
- Each suggestion: 2-3 words maximum
- Examples: "window light", "magic lit", "haunted atmosphere"
- All 3 combined with commas when applied

### Combined Prompt
When you click "Apply All & Relight":
```
Suggestion 1, Suggestion 2, Suggestion 3
```
This combined prompt is used for relighting.

## Tips for Best Results

### Mood Selection
- **1-2 moods**: More focused, specific results
- **3+ moods**: More creative, mixed results
- **No moods**: AI decides based purely on image

### Mood Combinations
**Good Combinations:**
- Haunted + Mysterious = Dark, eerie
- Romantic + Cinematic = Movie romance
- Playful + Outdoor = Bright, fun
- Dramatic + Cinematic = Epic, bold

**Interesting Combinations:**
- Haunted + Playful = Spooky but fun (Halloween vibes)
- Romantic + Mysterious = Intriguing intimacy
- Adventure + Dramatic = Epic journey

### When to Use What

**Portraits:**
- Romantic, Dramatic, Cinematic, Mysterious

**Landscapes:**
- Outdoor, Adventure, Cinematic, Dramatic

**Indoor Scenes:**
- Household, Romantic, Playful, Mysterious

**Product Photos:**
- Cinematic, Dramatic, Playful

**Artistic Shots:**
- Mysterious, Haunted, Cinematic, Dramatic

## Advantages

### Before (Old Way):
1. Upload image
2. Get generic suggestions
3. Apply one at a time
4. Manually trigger relight each time

### Now (New Way):
1. Upload image
2. Select moods to guide AI
3. Get personalized suggestions
4. Apply all 3 at once
5. Automatic relighting!

## FAQ

**Q: Do I have to select moods?**
A: No! Moods are optional. AI will still work without them.

**Q: Can I select all 9 moods?**
A: Yes, but 1-3 moods usually give better focused results.

**Q: What if I don't like the suggestions?**
A: Click "Get AI Recommendations" again for new suggestions!

**Q: Can I edit the combined prompt?**
A: Currently no, but you can manually type in the prompt field if you prefer.

**Q: How long does it take?**
A: AI recommendations: 2-5 seconds
   Relighting: 20-30 seconds
   Total: ~25-35 seconds

## Access the App

**URL:** http://localhost:5000

**Start Server:**
```cmd
cd D:\new1\light
D:\new1\venv_py310\Scripts\python.exe app.py
```

Enjoy creating amazing relit images with mood-guided AI! 🎨✨
