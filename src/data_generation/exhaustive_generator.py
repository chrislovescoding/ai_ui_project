"""
Generate *all* possible UI rasters (256×256×2 states) and de-duplicate them.

Usage (from project root):

    python -m src.data_generation.exhaustive_generator \
           --config assets/design_spec.yaml \
           --out   F:/ai_ui_project/data/exhaustive

The script will create the output directory if it doesn’t exist.
"""

import argparse
import csv
import hashlib
import os
import sys
import time
from io import BytesIO
from pathlib import Path

from PIL import Image

# --- bring in your existing helpers -------------------------------------------------
from .config import load_config_and_assets
from .episode_generator import render_full_frame
from .state_logic import determine_ui_state

# ------------------------------------------------------------------------------------


def md5_png(pil_img: Image.Image) -> str:
    """Return the MD5 of a Pillow image saved as PNG in-memory."""
    buf = BytesIO()
    pil_img.save(buf, format="PNG", optimize=True)
    return hashlib.md5(buf.getvalue()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="assets/design_spec.yaml",
                        help="Path to design_spec.yaml")
    parser.add_argument("--out",    required=True,
                        help="Output directory (will be created)")
    args = parser.parse_args()

    cfg  = load_config_and_assets(args.config)
    out_root = Path(args.out)
    img_dir  = out_root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    mapping_path  = out_root / "mapping.csv"
    manifest_path = out_root / "manifest.yaml"

    # Pre-allocate bookkeeping structures
    seen_hash_to_index = {}   # md5 → image index (int)
    mapping_rows       = []   # rows to write to CSV

    img_count      = 0
    duplicate_hits = 0

    mouse_states = ("UP", "DOWN")

    t0 = time.time()
    print("Starting exhaustive generation …")

    for y in range(cfg.WINDOW_HEIGHT):
        for x in range(cfg.WINDOW_WIDTH):
            for m_state in mouse_states:
                ui_state_name = determine_ui_state(cfg, x, y, m_state)
                frame = render_full_frame(cfg, ui_state_name, x, y)

                h = md5_png(frame)

                if h not in seen_hash_to_index:
                    # New, unique raster → persist PNG
                    fname = f"{len(seen_hash_to_index):06d}.png"
                    frame.save(img_dir / fname, format="PNG", optimize=True)
                    seen_hash_to_index[h] = len(seen_hash_to_index)
                else:
                    duplicate_hits += 1

                mapping_rows.append((
                    x, y, m_state,
                    seen_hash_to_index[h]         # integer ID of the representative PNG
                ))

        # progress every few scan-lines
        if (y + 1) % 32 == 0 or y == cfg.WINDOW_HEIGHT - 1:
            done = (y + 1) * cfg.WINDOW_WIDTH * len(mouse_states)
            total = cfg.WINDOW_WIDTH * cfg.WINDOW_HEIGHT * len(mouse_states)
            pct = 100 * done / total
            print(f"\r  {pct:6.2f}%  ({done:,}/{total:,})", end="", flush=True)

    # --- write CSV -------------------------------------------------------------------
    with mapping_path.open("w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["cursor_x", "cursor_y", "mouse_state", "image_index"])
        writer.writerows(mapping_rows)

    # --- write manifest --------------------------------------------------------------
    import yaml  # local import because it’s only needed here
    manifest = {
        "unique_frames": len(seen_hash_to_index),
        "total_positions": cfg.WINDOW_WIDTH * cfg.WINDOW_HEIGHT * len(mouse_states),
        "duplicates_collapsed": duplicate_hits,
        "checksum": "MD5 over PNG bytes",
        "generation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_used": os.path.relpath(args.config),
    }
    with manifest_path.open("w") as f_yaml:
        yaml.dump(manifest, f_yaml, indent=2, sort_keys=False)

    dt = time.time() - t0
    print(f"\nDone.  Unique rasters: {manifest['unique_frames']:,} "
          f"(deduped {duplicate_hits:,} duplicates)  — {dt:.1f}s elapsed.")
    print(f"Images   → {img_dir}")
    print(f"Mapping  → {mapping_path}")
    print(f"Manifest → {manifest_path}")


if __name__ == "__main__":
    # Make “python exhaustive_generator.py …” work without the -m flag
    if __package__ is None:
        sys.path.append(str(Path(__file__).resolve().parents[2]))
    main()
