# Unified Router v2.0 - Demo Guide

Complete guide for creating and running demonstrations of Unified Router.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Automated Demo Recording](#automated-demo-recording)
3. [Interactive Live Demo](#interactive-live-demo)
4. [Creating GIF/Video](#creating-gifvideo)
5. [Troubleshooting](#troubleshooting)
6. [Presentation Tips](#presentation-tips)

---

## Quick Start

### Prerequisites

```bash
# Install required dependencies
pip install rich

# Optional: For GIF creation
pip install termtosvg

# Optional: For video creation
# Install ffmpeg via your package manager
# macOS: brew install ffmpeg
# Linux: apt-get install ffmpeg
```

### Run Automated Demo

```bash
# Navigate to project root
cd /Users/home/projects/llm-router-v14

# Run automated recording
python scripts/demo_recording.py

# Output files will be created:
# - demo_output.html (view in browser)
# - demo_output.svg (vector graphics)
```

### Run Interactive Demo

```bash
# Run interactive demo
python scripts/demo_live.py

# Follow on-screen prompts to explore different routing strategies
```

---

## Automated Demo Recording

### Overview

The automated demo (`scripts/demo_recording.py`) creates a pre-scripted demonstration showcasing:

1. **Intro** (5s) - Title screen
2. **Initialization** (8s) - Component loading
3. **Quality Routing** (10s) - Eagle ELO demo
4. **Cost Routing** (9s) - CARROT demo
5. **Cascade Routing** (8s) - Multi-tier fallback
6. **Quality Check** (9s) - Self-check system
7. **Performance** (10s) - Metrics and stats
8. **Finale** (6s) - Call to action

**Total Duration**: ~65 seconds

### Running the Demo

```bash
# Basic usage
python scripts/demo_recording.py

# Specify output directory
python scripts/demo_recording.py -o ./demo_outputs

# Make executable
chmod +x scripts/demo_recording.py
./scripts/demo_recording.py
```

### Output Files

After running, you'll get:

- **demo_output.html** - HTML version (view in browser, send via email)
- **demo_output.svg** - SVG version (for conversion to GIF/video)

### Customization

Edit `scripts/demo_recording.py` to customize:

- **Scene durations**: Adjust `await asyncio.sleep()` values
- **Content**: Modify text in each scene method
- **Styling**: Change Rich console colors and formatting
- **New scenes**: Add additional `async def scene_X()` methods

---

## Interactive Live Demo

### Overview

The interactive demo (`scripts/demo_live.py`) provides a menu-driven interface for live presentations.

### Features

- **6 Demo Options**:
  1. Quality-Focused Routing
  2. Cost-Aware Routing
  3. Cascade Routing
  4. Balanced Routing
  5. Custom Query (user input)
  6. Performance Stats

- **Interactive**: Navigate using keyboard
- **Flexible**: Skip/repeat demos as needed
- **Real-time**: Show actual system behavior

### Usage

```bash
python scripts/demo_live.py
```

**Controls**:
- Enter number (1-6) to select demo
- Enter 0 to exit
- Ctrl+C to force quit
- Press Enter to continue after each demo

### Best Practices for Live Demos

1. **Prepare Queries**: Have interesting queries ready for "Custom Query" option
2. **Know Your Audience**: Choose appropriate demos (technical vs. business)
3. **Time Management**: Each demo takes 30-90 seconds
4. **Have Backup**: Always have the automated demo ready if live demo fails

---

## Creating GIF/Video

### Method 1: Terminal to GIF (termtosvg)

```bash
# Install termtosvg
pip install termtosvg

# Record a new session
termtosvg record demo_session.cast

# Run your demo
python scripts/demo_recording.py

# Stop recording (Ctrl+D)

# Render to GIF
termtosvg render demo_session.cast demo.gif

# Optimize GIF size
gifsicle -O3 --colors 256 demo.gif -o demo_optimized.gif
```

### Method 2: SVG to GIF (CairoSVG + ImageMagick)

```bash
# Install dependencies
# macOS:
brew install imagemagick

# Convert SVG to PNG sequence
convert demo_output.svg -resize 800x600 demo_output.png

# Create GIF from PNG
convert -delay 10 -loop 0 demo_output.png demo.gif

# Optimize
gifsicle -O3 --colors 256 --lossy=80 demo.gif -o demo_optimized.gif
```

### Method 3: Screen Recording to Video

```bash
# macOS: Use QuickTime Player
# 1. Open QuickTime Player
# 2. File → New Screen Recording
# 3. Start recording
# 4. Run: python scripts/demo_recording.py
# 5. Stop recording (Cmd+Ctrl+Esc)
# 6. Save as demo.mov

# Convert to MP4
ffmpeg -i demo.mov -vcodec libx264 -crf 23 -preset medium demo.mp4

# Optimize for web
ffmpeg -i demo.mp4 -vcodec libx264 -crf 28 -preset slow -vf "scale=800:-1" demo_optimized.mp4
```

### Method 4: HTML to Video (Puppeteer)

Create `scripts/html_to_video.js`:

```javascript
const puppeteer = require('puppeteer');
const { exec } = require('child_process');

(async () => {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();
  await page.goto('file:///path/to/demo_output.html');

  // Take screenshots
  for (let i = 0; i < 100; i++) {
    await page.screenshot({ path: `frames/frame_${i}.png` });
    await page.evaluate(() => window.scrollBy(0, 10));
    await page.waitForTimeout(100);
  }

  await browser.close();

  // Convert to video
  exec('ffmpeg -framerate 30 -i frames/frame_%d.png -c:v libx264 demo.mp4');
})();
```

### GIF Optimization Tips

```bash
# Reduce colors
gifsicle --colors 256 input.gif -o output.gif

# Reduce size
gifsicle --resize 800x_ input.gif -o output.gif

# Optimize with lossy compression
gifsicle -O3 --lossy=80 --colors 256 input.gif -o output.gif

# Check size
ls -lh demo.gif
# Target: < 5MB for GitHub

# If still too large, reduce fps
gifsicle --delay=15 input.gif -o output.gif  # ~6.7 fps instead of 10 fps
```

---

## Troubleshooting

### Problem: "Module 'rich' not found"

**Solution**:
```bash
pip install rich
# or
pip install -r requirements.txt
```

### Problem: GIF is too large (>5MB)

**Solutions**:
```bash
# Option 1: Reduce colors
gifsicle --colors 128 demo.gif -o demo_small.gif

# Option 2: Reduce resolution
gifsicle --resize 640x_ demo.gif -o demo_small.gif

# Option 3: Reduce frame rate
gifsicle --delay=20 demo.gif -o demo_small.gif

# Option 4: Use lossy compression
gifsicle -O3 --lossy=100 --colors 256 demo.gif -o demo_small.gif

# Combine all
gifsicle -O3 --lossy=100 --colors 128 --resize 640x_ --delay=15 demo.gif -o demo_final.gif
```

### Problem: Demo runs too fast/slow

**Solution**: Edit timing in `scripts/demo_recording.py`:

```python
# Make slower - increase sleep values
await asyncio.sleep(3)  # was: 1.5

# Make faster - decrease sleep values
await asyncio.sleep(0.5)  # was: 2
```

### Problem: Colors look weird in terminal

**Solution**:
```bash
# Use a modern terminal with 256 color support
# macOS: iTerm2, Terminal.app (latest)
# Linux: gnome-terminal, konsole
# Windows: Windows Terminal

# Test color support
python -c "from rich.console import Console; Console().print('[red]Test[/red]')"
```

### Problem: HTML output doesn't render correctly

**Solution**:
```bash
# Open in a modern browser
# Chrome, Firefox, Safari (latest versions)

# If still issues, export to SVG instead
# SVG is more widely supported
```

---

## Presentation Tips

### For Technical Audiences

**Focus on**:
- Architecture details (7-step pipeline)
- Component integration
- Performance metrics
- Code examples

**Demos to emphasize**:
- Quality-Focused Routing (Eagle ELO)
- Quality Verification (Self-Check)
- Performance Stats

### For Business Audiences

**Focus on**:
- Cost savings (CARROT)
- Quality assurance
- Production readiness
- ROI

**Demos to emphasize**:
- Cost-Aware Routing
- Cascade Routing (fast response times)
- Performance Stats (uptime, success rate)

### General Tips

1. **Test First**: Always run the demo once before presenting
2. **Have Backup**: Keep both automated and interactive demos ready
3. **Know Timing**: Each scene takes 5-10 seconds
4. **Explain Context**: Briefly intro each demo before running
5. **Handle Questions**: Pause between demos for Q&A
6. **Show Real Code**: Have the actual codebase ready to show

### Sample Presentation Flow (5 minutes)

```
00:00 - Introduction (30s)
  "Today I'll demo Unified Router v2.0, an intelligent LLM routing system..."

00:30 - Run Automated Demo (65s)
  python scripts/demo_recording.py
  (Let it play through all 8 scenes)

01:35 - Highlight Key Features (60s)
  - "Notice the 21 components loading..."
  - "Eagle ELO selected the highest quality model..."
  - "CARROT optimized for cost/quality balance..."
  - "Quality check verified the response..."

02:35 - Interactive Demo (90s)
  python scripts/demo_live.py
  - Show custom query option
  - Demonstrate different strategies

04:05 - Wrap-up & Q&A (55s)
  - GitHub link
  - Documentation
  - Questions?
```

---

## Advanced: Custom Demos

### Creating Your Own Demo

1. **Copy template**:
```bash
cp scripts/demo_recording.py scripts/my_custom_demo.py
```

2. **Add new scene**:
```python
async def scene_X_my_feature(self):
    """My custom demo scene"""
    self.console.clear()
    self.console.print("[bold cyan]My Feature Demo[/bold cyan]\n")
    # Your code here
    await asyncio.sleep(2)
```

3. **Update recording sequence**:
```python
async def record_demo(self):
    await self.scene_1_intro()
    # ... existing scenes ...
    await self.scene_X_my_feature()  # Add your scene
    await self.scene_8_finale()
```

### Using Demo Config

Load settings from `scripts/demo_config.yaml`:

```python
import yaml

with open('scripts/demo_config.yaml') as f:
    config = yaml.safe_load(f)

# Use config values
duration = config['scenes']['intro']
models = config['models']
```

---

## Recording for Social Media

### Twitter/X (280 characters + GIF)

```
🚀 Unified LLM Router v2.0
✨ 21 components
⚡ 7.84 QPS
🎯 4 routing strategies
📊 99.6% tests passing

Intelligent routing with quality assurance
#AI #LLM #OpenSource

[Attach: demo_optimized.gif]
```

### LinkedIn (Short post + Video)

```
Excited to share Unified LLM Router v2.0! 🚀

Key highlights:
• 21 integrated components
• 4 routing strategies (Quality, Cost, Cascade, Balanced)
• Production-ready with 455/457 tests passing
• Complete Docker deployment

Perfect for teams running multiple LLM models who need intelligent routing and quality assurance.

GitHub: https://github.com/Dimas1962/llm-router-v14

[Attach: demo.mp4]
```

### YouTube (Tutorial Video)

**Recommended structure**:
1. Intro (15s)
2. Automated Demo (65s)
3. Deep Dive - Each component explained (3-5 min)
4. Live Demo (2 min)
5. Integration Example (2 min)
6. Conclusion + CTA (30s)

**Total**: 8-10 minutes

---

## Resources

### Tools

- **rich** - Terminal formatting (required)
- **termtosvg** - Terminal to SVG/GIF
- **gifsicle** - GIF optimization
- **ffmpeg** - Video conversion
- **imagemagick** - Image processing

### Installation

```bash
# Python packages
pip install rich termtosvg

# macOS
brew install gifsicle ffmpeg imagemagick

# Ubuntu/Debian
sudo apt-get install gifsicle ffmpeg imagemagick

# Windows (via Chocolatey)
choco install gifsicle ffmpeg imagemagick
```

### Links

- **Project**: https://github.com/Dimas1962/llm-router-v14
- **Releases**: https://github.com/Dimas1962/llm-router-v14/releases
- **Documentation**: See `docs/` directory
- **Issues**: https://github.com/Dimas1962/llm-router-v14/issues

---

## Support

For help with demos:
1. Check this guide
2. Read inline comments in demo scripts
3. Open an issue on GitHub
4. See examples in `scripts/` directory

---

**Version**: 2.0.0
**Last Updated**: 2025
**Status**: Production Ready

**Happy Demoing!** 🎬
