# Creating Animated GIF Demo

## ⚠️ Important Note

The `demo_output.svg` created by `demo_recording.py` is a **static snapshot**, not an animated recording. To create an **animated GIF**, you need to record a live terminal session.

## 📹 Method 1: Using termtosvg (Recommended)

**Install termtosvg:**
```bash
pip install termtosvg
```

**Record animated demo:**
```bash
# Start recording
termtosvg record demo_session.cast

# Run the demo
source venv/bin/activate
python scripts/demo_recording.py

# Stop recording (Ctrl+D)

# Render to GIF
termtosvg render demo_session.cast demo.gif

# Optimize GIF (if gifsicle installed)
gifsicle -O3 --colors 256 --lossy=80 demo.gif -o demo_optimized.gif
```

**Check size:**
```bash
ls -lh demo_optimized.gif
# Target: < 5MB for GitHub
```

---

## 📹 Method 2: Using asciinema + agg

**Install tools:**
```bash
pip install asciinema
# Install agg from: https://github.com/asciinema/agg
```

**Record and convert:**
```bash
# Record session
asciinema rec demo_session.cast

# Run demo in the recording session
source venv/bin/activate
python scripts/demo_recording.py

# Exit (Ctrl+D)

# Convert to GIF
agg demo_session.cast demo.gif

# Optimize
gifsicle -O3 --colors 256 demo.gif -o demo_optimized.gif
```

---

## 📹 Method 3: Screen Recording (macOS)

**Record with QuickTime:**
1. Open QuickTime Player
2. File → New Screen Recording
3. Start recording
4. Run: `python scripts/demo_recording.py`
5. Stop recording (Cmd+Ctrl+Esc)
6. Save as `demo.mov`

**Convert to GIF with ffmpeg:**
```bash
# Install ffmpeg (if needed)
brew install ffmpeg

# Convert MOV to GIF
ffmpeg -i demo.mov -vf "fps=10,scale=800:-1:flags=lanczos" -c:v gif demo.gif

# Optimize
gifsicle -O3 --colors 256 --lossy=80 demo.gif -o demo_optimized.gif
```

---

## 📹 Method 4: Python Script (Advanced)

Create multiple frames and combine into GIF:

```python
#!/usr/bin/env python3
"""
Create animated GIF by capturing multiple frames
"""

import asyncio
import subprocess
from pathlib import Path
from PIL import Image

async def capture_frames():
    """Capture frames during demo execution"""
    frames = []

    # TODO: Implement frame capture during demo execution
    # This requires modifying demo_recording.py to save
    # intermediate frames

    pass

async def create_gif():
    """Create GIF from frames"""
    frames = await capture_frames()

    # Save as GIF
    frames[0].save(
        'demo.gif',
        save_all=True,
        append_images=frames[1:],
        duration=100,  # ms per frame
        loop=0,
        optimize=True
    )

if __name__ == '__main__':
    asyncio.run(create_gif())
```

---

## 🎯 Current Demo Assets

**Available now:**
- `demo_output.html` - HTML preview (view in browser)
- `demo_output.svg` - Static SVG snapshot
- `demo_preview.png` - Full-size PNG preview (1200px)
- `demo_thumbnail.png` - Smaller thumbnail (800px)

**To create:**
- `demo.gif` - Animated GIF (requires recording, see methods above)
- `demo_optimized.gif` - Optimized animated GIF (< 5MB)

---

## 📊 GIF Optimization Tips

**Reduce file size:**
```bash
# Method 1: Reduce colors
gifsicle --colors 128 demo.gif -o demo_small.gif

# Method 2: Reduce frame rate
gifsicle --delay=15 demo.gif -o demo_small.gif  # ~6.7 fps

# Method 3: Reduce resolution
gifsicle --resize 640x_ demo.gif -o demo_small.gif

# Method 4: Combine all (best compression)
gifsicle -O3 --lossy=100 --colors 128 --resize 640x_ --delay=15 demo.gif -o demo_optimized.gif
```

**Check final size:**
```bash
ls -lh demo_optimized.gif
# GitHub limit: 10MB (recommended < 5MB)
```

---

## 🚀 Quick Start (Recommended)

**For static preview (GitHub README):**
```bash
# Already created!
# Use: demo_preview.png or demo_thumbnail.png
```

**For animated GIF:**
```bash
# Install termtosvg
pip install termtosvg

# Record and create GIF
termtosvg record demo.cast
# Run: python scripts/demo_recording.py
# Press Ctrl+D to stop

termtosvg render demo.cast demo.gif
gifsicle -O3 --colors 256 demo.gif -o demo_optimized.gif
```

---

## 📝 Add to README.md

**Static preview:**
```markdown
## Demo

![Unified Router Demo](demo_preview.png)
```

**Animated GIF (after creation):**
```markdown
## Demo

![Unified Router Demo](demo_optimized.gif)

*Showing quality routing, cost optimization, and cascade fallback*
```

**Both:**
```markdown
## Demo

### Quick Preview
![Static Demo Preview](demo_preview.png)

### Full Animated Demo
![Animated Demo](demo_optimized.gif)
```

---

## 🔗 Resources

- **termtosvg**: https://github.com/nbedos/termtosvg
- **asciinema**: https://asciinema.org/
- **agg**: https://github.com/asciinema/agg
- **gifsicle**: https://www.lcdf.org/gifsicle/
- **ffmpeg**: https://ffmpeg.org/

---

**Current Status:**
- ✅ Static preview images created (demo_preview.png, demo_thumbnail.png)
- ⏳ Animated GIF - requires terminal recording (use methods above)

**Recommendation:** Use `demo_preview.png` for README now, create animated GIF later if needed.
