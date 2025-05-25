#!/usr/bin/env python3
"""
Creates GIFs from generated UI episode data.

This script processes episode directories (e.g., ep_00000, ep_00001, ...)
found in an input directory. For each episode, it generates three GIFs:
1.  A 'visual' GIF from the rendered UI frames (0000_prev.png + all XXXX_next.png).
2.  A 'cursor_pos_mask' GIF from the cursor position mask frames (all XXXX_cursor_pos_mask.png).
3.  A 'click_state_mask' GIF from the click state mask frames (all XXXX_click_state_mask.png).
"""

import argparse
from pathlib import Path
from PIL import Image
import sys

# Add import for config loader
# This assumes the script is in 'scripts/' and 'src/' is a sibling directory
# containing 'data_generation'.
# Best practice is to run this script from the project root.
try:
    sys.path.insert(0, str((Path(__file__).resolve().parent.parent / 'src')))
    from data_generation.config import load_config_and_assets
except ImportError:
    print("Error: Could not import 'data_generation.config'. \n"
          "Ensure that 'src' directory is in the parent directory of 'scripts' "
          "and contains the 'data_generation' package, \n"
          "or that the script is run from a location where 'src' is discoverable.")
    # Fallback if running from project root and 'src' is directly in pythonpath
    try:
        from src.data_generation.config import load_config_and_assets
    except ImportError:
        sys.exit("Failed to import 'load_config_and_assets'. Please check your PYTHONPATH and script location.")


def create_gifs_for_episode(episode_dir: Path, output_gif_dir: Path, frame_duration_ms: int):
    """
    Creates visual, cursor position mask, and click state mask GIFs for a single episode.
    """
    print(f"Processing episode: {episode_dir.name}...")

    # --- 1. Visual Frames GIF ---
    visual_frame_paths = []
    first_visual_frame_path = episode_dir / "0000_prev.png"
    next_frame_files = sorted(list(episode_dir.glob("[0-9][0-9][0-9][0-9]_next.png")))

    has_first_frame = False
    if first_visual_frame_path.is_file():
        visual_frame_paths.append(first_visual_frame_path)
        has_first_frame = True
    
    visual_frame_paths.extend(next_frame_files)

    if not visual_frame_paths:
        print(f"  No visual frame files (_prev.png or _next.png) found in {episode_dir.name}. Skipping visual GIF.")
    else:
        if not has_first_frame and next_frame_files:
            print(f"  Warning: Expected '0000_prev.png' not found in {episode_dir.name}. Visual GIF will start from the first available '_next.png'.")
        
        try:
            pil_visual_frames = [Image.open(fp) for fp in visual_frame_paths]
            if not pil_visual_frames:
                print(f"  Could not load any visual frames for {episode_dir.name}. Skipping visual GIF.")
            else:
                visual_gif_path = output_gif_dir / f"{episode_dir.name}_visual.gif"
                pil_visual_frames[0].save(
                    visual_gif_path,
                    save_all=True,
                    append_images=pil_visual_frames[1:],
                    duration=frame_duration_ms,
                    loop=0
                )
                print(f"  Saved visual GIF: {visual_gif_path} ({len(pil_visual_frames)} frames)")
        except Exception as e:
            print(f"  Error creating visual GIF for {episode_dir.name}: {e}")

    # --- 2. Cursor Position Mask Frames GIF ---
    # Looks for XXXX_cursor_pos_mask.png files
    cursor_pos_mask_files = sorted(list(episode_dir.glob("[0-9][0-9][0-9][0-9]_cursor_pos_mask.png")))

    if not cursor_pos_mask_files:
        print(f"  No cursor position mask frames (*_cursor_pos_mask.png) found in {episode_dir.name}. Skipping cursor position mask GIF.")
    else:
        try:
            pil_cursor_pos_frames = [Image.open(fp) for fp in cursor_pos_mask_files]
            if not pil_cursor_pos_frames:
                print(f"  Could not load any cursor position mask frames for {episode_dir.name}. Skipping GIF.")
            else:
                cursor_pos_gif_path = output_gif_dir / f"{episode_dir.name}_cursor_pos_mask.gif"
                pil_cursor_pos_frames[0].save(
                    cursor_pos_gif_path,
                    save_all=True,
                    append_images=pil_cursor_pos_frames[1:],
                    duration=frame_duration_ms,
                    loop=0
                )
                print(f"  Saved cursor position mask GIF: {cursor_pos_gif_path} ({len(pil_cursor_pos_frames)} frames)")
        except Exception as e:
            print(f"  Error creating cursor position mask GIF for {episode_dir.name}: {e}")

    # --- 3. Click State Mask Frames GIF ---
    # Looks for XXXX_click_state_mask.png files
    click_state_mask_files = sorted(list(episode_dir.glob("[0-9][0-9][0-9][0-9]_click_state_mask.png")))

    if not click_state_mask_files:
        print(f"  No click state mask frames (*_click_state_mask.png) found in {episode_dir.name}. Skipping click state mask GIF.")
    else:
        try:
            pil_click_state_frames = [Image.open(fp) for fp in click_state_mask_files]
            if not pil_click_state_frames:
                print(f"  Could not load any click state mask frames for {episode_dir.name}. Skipping GIF.")
            else:
                click_state_gif_path = output_gif_dir / f"{episode_dir.name}_click_state_mask.gif"
                pil_click_state_frames[0].save(
                    click_state_gif_path,
                    save_all=True,
                    append_images=pil_click_state_frames[1:],
                    duration=frame_duration_ms,
                    loop=0
                )
                print(f"  Saved click state mask GIF: {click_state_gif_path} ({len(pil_click_state_frames)} frames)")
        except Exception as e:
            print(f"  Error creating click state mask GIF for {episode_dir.name}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Create GIFs from generated UI episode data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=None, 
        help="Directory containing episode subdirectories (e.g., data/ep_00000). If not set, uses OUTPUT_DIR_BASE from design_spec.yaml."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data_gifs"),
        help="Directory to save the generated GIFs. Will be created in the current working directory if it doesn't exist."
    )
    parser.add_argument(
        "--frame_duration",
        type=int,
        default=100,
        help="Frame duration in milliseconds for the GIFs."
    )
    parser.add_argument(
        "--episode_prefix",
        type=str,
        default="ep_",
        help="Prefix for episode directories to process."
    )
    parser.add_argument(
        "--config_path",
        type=Path,
        default=Path("assets/design_spec.yaml"), # Relative to CWD
        help="Path to the design_spec.yaml configuration file."
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Maximum number of episode directories to process. If not set, processes all."
    )

    args = parser.parse_args()

    if args.input_dir is None:
        try:
            print(f"Input directory not specified. Loading from config: {args.config_path.resolve()}")
            if not args.config_path.is_file():
                 print(f"Error: Config file '{args.config_path.resolve()}' not found. Please specify --input_dir or ensure config path is correct.")
                 sys.exit(1)
            cfg = load_config_and_assets(yaml_path=str(args.config_path))
            args.input_dir = Path(cfg.OUTPUT_DIR_BASE)
            print(f"Using input directory from config: {args.input_dir.resolve()}")
        except Exception as e:
            print(f"Error loading input directory from config: {e}")
            print("Please specify --input_dir explicitly or ensure 'assets/design_spec.yaml' (or specified config) is valid and accessible.")
            sys.exit(1)


    if not args.input_dir.is_dir():
        print(f"Error: Input directory '{args.input_dir.resolve()}' not found or is not a directory.")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output GIFs will be saved in: {args.output_dir.resolve()}")

    episode_dirs = sorted([
        d for d in args.input_dir.iterdir() 
        if d.is_dir() and d.name.startswith(args.episode_prefix)
    ])

    if args.max_episodes is not None:
        episode_dirs = episode_dirs[:args.max_episodes]

    if not episode_dirs:
        print(f"No episode directories found in '{args.input_dir.resolve()}' with prefix '{args.episode_prefix}'.")
        sys.exit(0)
    
    print(f"Found {len(episode_dirs)} episode(s) to process.")

    for episode_dir in episode_dirs:
        create_gifs_for_episode(episode_dir, args.output_dir, args.frame_duration)

    print("\nGIF generation complete.")

if __name__ == "__main__":
    main()