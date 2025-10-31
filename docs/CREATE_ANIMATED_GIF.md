# Creating Animated GIF - Manual Instructions

## ⚠️ Important

**Automated terminal recording doesn't capture the full demo correctly.**

The `demo.cast` file created by automated recording is empty because:
- termtosvg needs an interactive terminal session
- The demo script completes before recording captures output
- Automated command execution doesn't preserve terminal formatting

## ✅ Current Working Solution

**Use the static preview image:**
- `assets/demo_preview.png` (442KB) - Shows all 8 demo scenes
- Already on GitHub and in README
- Looks professional and loads fast

## 📹 To Create True Animated GIF (Manual Process)

### Method 1: Manual termtosvg Recording

```bash
# 1. Start recording interactively
termtosvg record demo.cast

# 2. In the recording session, manually run:
source venv/bin/activate
python scripts/demo_recording.py

# 3. Wait for demo to complete (~65 seconds)

# 4. Stop recording: Ctrl+D

# 5. Convert to GIF
termtosvg render demo.cast demo.svg

# 6. Convert SVG to GIF using ImageMagick
convert demo.svg demo.gif

# 7. Optimize
gifsicle -O3 --colors 256 demo.gif -o demo_optimized.gif
```

### Method 2: Screen Recording (Easiest)

**macOS:**
```bash
# 1. Open QuickTime Player
# 2. File → New Screen Recording
# 3. Select terminal window
# 4. Run: python scripts/demo_recording.py
# 5. Save as demo.mov

# 6. Convert to GIF
ffmpeg -i demo.mov -vf "fps=10,scale=800:-1" demo.gif
gifsicle -O3 --colors 256 demo.gif -o demo_optimized.gif
```

**Linux:**
```bash
# Use peek, byzanz, or asciinema
peek  # GUI tool for GIF recording
```

### Method 3: Use Animated SVG

GitHub supports animated SVGs! Use the existing demo_output.svg:

```markdown
![Demo](demo_output.svg)
```

However, animated SVGs created by our demo script are static snapshots, not true animations.

## 💡 Recommendation

**For now, stick with `assets/demo_preview.png`** which:
- ✅ Works perfectly on GitHub
- ✅ Shows all 8 scenes clearly
- ✅ Loads fast (442KB)
- ✅ Professional appearance

**To create animated GIF later:**
- Do it manually when you have time
- Screen recording is easiest method
- Target: 2-5MB file size

## 📊 Current Demo Assets

```
✅ assets/demo_preview.png  - Static preview (on GitHub)
✅ demo_output.html        - Browser viewable
✅ demo_output.svg         - Static snapshot
⏳ Animated GIF            - Requires manual recording
```

**Bottom line:** The static preview is excellent and sufficient for most users. Create animated GIF only if you really need it.
