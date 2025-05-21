#!/usr/bin/env python
"""
ui_renderer.py – Stage 1 synthetic-data generator for the AI-imagined UI project

Creates episodes of simple UI frames (256×256 RGB) containing:
  • a single button that visually reacts on hover and on press
  • a cursor icon rendered at the current mouse location
  • TWO mask PNGs per frame:
      ######_pointer.png  – 255 where the cursor is present (always)
      ######_click.png    – 255 where the cursor is clicking

For each frame t (t ≥ 1) in an episode we write:
   {t:04d}_prev.png     – frame t-1
   {t:04d}_pointer.png  – pointer-location mask for frame t
   {t:04d}_click.png    – click mask for frame t
   {t:04d}_next.png     – frame t
"""

import argparse, math, random, sys
from pathlib import Path
from multiprocessing import Pool, cpu_count

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# -----------------------------------------------------------------------------#
#  Constants and helpers
# -----------------------------------------------------------------------------#

CANVAS = 256                   # frame side length (square)
BTN_MIN, BTN_MAX = 60, 120     # button size range (pixels)

def _load_font(size: int = 18) -> ImageFont.ImageFont:
    """Try common system fonts; fallback to PIL default."""
    fallback = ImageFont.load_default()
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/SFNSDisplay.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return fallback

FONT = _load_font(18)

def random_colour(low: int = 40, high: int = 215) -> tuple[int, int, int]:
    """Return an RGB tuple with components in [low, high]."""
    return tuple(random.randint(low, high) for _ in range(3))

def load_cursor(path: Path = Path("assets/cursor.png")) -> Image.Image:
    """Load the 32×32 RGBA cursor icon, or exit if missing."""
    if not path.is_file():
        print(f"[ERROR] Cursor icon not found at {path}.", file=sys.stderr)
        sys.exit(1)
    return Image.open(path).convert("RGBA")

CURSOR_IMG = load_cursor()

# -----------------------------------------------------------------------------#
#  Frame rendering
# -----------------------------------------------------------------------------#

def make_frame(
    bg_rgb: tuple[int, int, int],
    btn_rect: tuple[int, int, int, int],
    btn_rgb_normal: tuple[int, int, int],
    text_rgb: tuple[int, int, int],
    text: str,
    cursor_xy: tuple[int, int] | None = None,
    pressed: bool = False,
) -> Image.Image:
    """
    Draw a single UI frame (RGBA → RGB) with normal, hover, and pressed shading.
    'pressed' must only be True if cursor is over the button.
    """
    img = Image.new("RGBA", (CANVAS, CANVAS), bg_rgb + (255,))
    d   = ImageDraw.Draw(img)
    x0, y0, x1, y1 = btn_rect

    # Determine if cursor is inside button
    inside = (
        cursor_xy is not None
        and x0 <= cursor_xy[0] <= x1 - CURSOR_IMG.width
        and y0 <= cursor_xy[1] <= y1 - CURSOR_IMG.height
    )

    # Shading: pressed > hover > normal
    if inside and pressed:
        shade = tuple(max(0, c - 40) for c in btn_rgb_normal)
    elif inside:
        shade = tuple(min(255, c + 40) for c in btn_rgb_normal)
    else:
        shade = btn_rgb_normal

    d.rounded_rectangle(btn_rect, radius=8, fill=shade, outline=(0, 0, 0))

    w, h = d.textbbox((0, 0), text, font=FONT)[2:]
    d.text(
        ((x0 + x1 - w) / 2, (y0 + y1 - h) / 2),
        text,
        fill=text_rgb,
        font=FONT,
    )

    if cursor_xy:
        img.alpha_composite(CURSOR_IMG, cursor_xy)

    return img.convert("RGB")


# -----------------------------------------------------------------------------#
#  Episode builder
# -----------------------------------------------------------------------------#

def make_episode(idx: int, seq_len: int, root: Path) -> None:
    """
    Generate one episode with `seq_len` frames, write PNG files:
      - prev.png, pointer.png, click.png, next.png
    """
    out_dir = root / f"ep_{idx:05d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Random UI layout
    bg = random_colour()
    btn_w = random.randint(BTN_MIN, BTN_MAX)
    btn_h = random.randint(BTN_MIN, BTN_MAX)
    bx0 = random.randint(10, CANVAS - btn_w - 10)
    by0 = random.randint(10, CANVAS - btn_h - 10)
    bx1, by1 = bx0 + btn_w, by0 + btn_h

    btn_rgb_normal = random_colour()
    text_rgb = (255, 255, 255)
    label    = random.choice(["OK", "Submit", "Launch", "Go"])

    # Simulate cursor trajectory
    start = (
        random.randint(0, CANVAS - CURSOR_IMG.width),
        random.randint(0, CANVAS - CURSOR_IMG.height),
    )
    goal = (
        (bx0 + bx1) // 2 - CURSOR_IMG.width // 2,
        (by0 + by1) // 2 - CURSOR_IMG.height // 2,
    )

    frames        = []
    pointer_masks = []
    click_masks   = []

    for t in range(seq_len):
        alpha = t / (seq_len - 1)
        ease  = 1 - math.cos(alpha * math.pi / 2)
        cx = int(start[0] * (1 - ease) + goal[0] * ease)
        cy = int(start[1] * (1 - ease) + goal[1] * ease)
        cursor_xy = (cx, cy)

        inside = (bx0 <= cx <= bx1 - CURSOR_IMG.width) and (
                     by0 <= cy <= by1 - CURSOR_IMG.height
                 )
        pressed = inside and (seq_len // 2 <= t < seq_len // 2 + 3)

        frame = make_frame(
            bg,
            (bx0, by0, bx1, by1),
            btn_rgb_normal,
            text_rgb,
            label,
            cursor_xy,
            pressed,
        )
        frames.append(frame)

        # Pointer-location mask (always)
        pm = Image.new("L", (CANVAS, CANVAS), 0)
        ImageDraw.Draw(pm).rectangle(
            (cx, cy, cx + CURSOR_IMG.width, cy + CURSOR_IMG.height),
            fill=255,
        )
        pointer_masks.append(pm)

        # Click mask (only when pressed)
        cm = Image.new("L", (CANVAS, CANVAS), 0)
        if pressed:
            ImageDraw.Draw(cm).rectangle(
                (cx, cy, cx + CURSOR_IMG.width, cy + CURSOR_IMG.height),
                fill=255,
            )
        click_masks.append(cm)

    # Write aligned files
    for i in range(1, seq_len):
        frames[i - 1].save(out_dir / f"{i:04d}_prev.png",    compress_level=1)
        pointer_masks[i].save(out_dir / f"{i:04d}_pointer.png", compress_level=1)
        click_masks[i].save(out_dir / f"{i:04d}_click.png",   compress_level=1)
        frames[i].save(out_dir / f"{i:04d}_next.png",        compress_level=1)


# -----------------------------------------------------------------------------#
#  Multiprocessing helper & CLI
# -----------------------------------------------------------------------------#

def _episode_star(args):
    return make_episode(*args)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate synthetic UI frame triplets"
    )
    p.add_argument("--episodes", type=int, default=10_000,
                   help="number of episodes")
    p.add_argument("--seq-len",  type=int, default=60,
                   help="frames per episode")
    p.add_argument("--out",      type=Path, default=Path("F:/ai_ui_project/data"),
                   help="output root dir")
    p.add_argument("--workers",  type=int,
                   default=max(1, cpu_count() - 1),
                   help="parallel worker processes")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    worker_args = [(i, args.seq_len, args.out) for i in range(args.episodes)]
    with Pool(processes=args.workers) as pool:
        for _ in tqdm(pool.imap_unordered(_episode_star, worker_args),
                      total=args.episodes, desc="Episodes"):
            pass
    print(f"\n✓ Finished. Episodes stored under: {args.out.resolve()}")

if __name__ == "__main__":
    main()
