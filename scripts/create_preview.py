#!/usr/bin/env python3
"""
Create PNG preview from SVG demo output
"""

import sys
from pathlib import Path

try:
    import cairosvg
    from PIL import Image
except ImportError:
    print("❌ Error: cairosvg and/or Pillow not installed")
    print("Install with: pip install cairosvg pillow")
    sys.exit(1)


def svg_to_png(svg_path: str, png_path: str, width: int = 1200):
    """Convert SVG to PNG with specified width"""
    print(f"📄 Converting {svg_path} to PNG...")

    # Convert SVG to PNG using cairosvg
    cairosvg.svg2png(
        url=svg_path,
        write_to=png_path,
        output_width=width
    )

    # Get file size
    png_size = Path(png_path).stat().st_size
    print(f"✅ Created {png_path} ({png_size / 1024:.1f} KB)")

    return png_path


def create_preview_images():
    """Create preview images from demo SVG"""
    svg_file = "demo_output.svg"

    if not Path(svg_file).exists():
        print(f"❌ Error: {svg_file} not found")
        print("Run: python scripts/demo_recording.py first")
        sys.exit(1)

    print("🎨 Creating preview images from demo_output.svg\n")

    # Create full-size PNG preview
    full_preview = svg_to_png(svg_file, "demo_preview.png", width=1200)

    # Create smaller thumbnail
    print("\n📐 Creating thumbnail...")
    with Image.open(full_preview) as img:
        # Calculate height maintaining aspect ratio
        aspect_ratio = img.height / img.width
        thumb_width = 800
        thumb_height = int(thumb_width * aspect_ratio)

        # Resize
        thumb = img.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        thumb.save("demo_thumbnail.png", optimize=True)

        thumb_size = Path("demo_thumbnail.png").stat().st_size
        print(f"✅ Created demo_thumbnail.png ({thumb_size / 1024:.1f} KB)")

    print("\n" + "="*70)
    print("✅ Preview images created successfully!")
    print("="*70)
    print("\nFiles created:")
    print(f"  📄 demo_preview.png - Full-size preview (1200px width)")
    print(f"  📄 demo_thumbnail.png - Thumbnail (800px width)")
    print("\n💡 Add to README.md:")
    print("  ![Demo Preview](demo_preview.png)")
    print("="*70 + "\n")


if __name__ == "__main__":
    create_preview_images()
