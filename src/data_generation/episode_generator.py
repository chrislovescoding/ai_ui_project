# src/data_generation/episode_generator.py
import os
from PIL import Image, ImageDraw
from . import drawing_utils
from . import state_logic

def render_full_frame(config, ui_state_name, cursor_root_x, cursor_root_y):
    """ Renders a complete frame based on UI state and cursor position. """
    image = Image.new('RGB', (config.WINDOW_WIDTH, config.WINDOW_HEIGHT), config.WINDOW_BG_COLOR)
    draw_context = ImageDraw.Draw(image)

    drawing_utils.draw_base_window(draw_context, config) # Draw background
    drawing_utils.draw_static_text(draw_context, config) # Draw "Chris Lock"

    # Determine button fill color based on state
    if ui_state_name == "UI_STATE_NORMAL":
        button_fill = config.BUTTON_NORMAL_FILL_COLOR
    elif ui_state_name == "UI_STATE_HOVER":
        button_fill = config.BUTTON_HOVER_FILL_COLOR
    elif ui_state_name == "UI_STATE_PRESSED":
        button_fill = config.BUTTON_PRESSED_FILL_COLOR
    else: # Should not happen
        button_fill = (255, 0, 0) # Error color

    drawing_utils.draw_button(draw_context, config, button_fill)

    if ui_state_name == "UI_STATE_PRESSED":
        drawing_utils.draw_pressed_indicator_text(draw_context, config) # Draw "Pressed!"

    drawing_utils.draw_cursor(image, config, cursor_root_x, cursor_root_y)
    
    return image

def generate_episode_triplets(episode_id, trajectory_callable, config):
    """
    Generates and saves all (prev_frame, cursor_mask, next_frame) triplets for an episode.
    trajectory_callable: A function that when called (e.g. trajectory_callable(config)) returns a generator/iterator
                         yielding (cursor_x, cursor_y, mouse_button_state_str)
    """
    episode_dir = os.path.join(config.OUTPUT_DIR_BASE, f"ep_{episode_id:05d}")
    os.makedirs(episode_dir, exist_ok=True)

    trajectory = trajectory_callable(config) # Get the iterator

    # Initial frame setup
    try:
        # Frame 0: Render it to be `prev_frame` for the first triplet
        curr_cursor_x, curr_cursor_y, curr_mouse_state = next(trajectory)
        curr_ui_state_name = state_logic.determine_ui_state(
            config, curr_cursor_x, curr_cursor_y, curr_mouse_state
        )
        prev_frame_image = render_full_frame(
            config, curr_ui_state_name, curr_cursor_x, curr_cursor_y
        )
    except StopIteration:
        print(f"Warning: Trajectory for episode {episode_id} yielded no frames.")
        return

    # Loop for subsequent frames (0 to FRAMES_PER_EPISODE - 2 for triplets)
    # We need total FRAMES_PER_EPISODE cursor states to make FRAMES_PER_EPISODE-1 triplets
    # The loop below makes config.FRAMES_PER_EPISODE-1 triplets.
    # Prompt says: "30 frames for everyday behaviour" -- this usually means 30 states, or 29 triplets.
    # If it means 30 triplets, then trajectories need to yield 31 states.
    # Let's assume 30 states from trajectory -> 29 triplets.
    # The prompt's data section: "ep_xxxxx/{####_{prev,mask,next}.png}"
    # If an episode has N frames (states), it has N-1 triplets.
    # If 30 frames are rendered for an episode, indices 0..29.
    # Triplet 0: prev=f0, next=f1. Triplet 28: prev=f28, next=f29. Total 29 triplets.
    # So, trajectory should yield config.FRAMES_PER_EPISODE states.

    for frame_idx_for_triplet_output in range(config.FRAMES_PER_EPISODE - 1):
        try:
            next_cursor_x, next_cursor_y, next_mouse_state = next(trajectory)
        except StopIteration:
            # Trajectory ended sooner than FRAMES_PER_EPISODE.
            # This can happen if trajectory generators are not careful about yielding enough frames.
            # Or if FRAMES_PER_EPISODE is 1.
            # print(f"Warning: Trajectory for ep {episode_id} ended at frame {frame_idx_for_triplet_output+1} "
            #       f"instead of {config.FRAMES_PER_EPISODE}.")
            break 

        # Generate Cursor Mask for next_frame's cursor position
        cursor_mask_image = drawing_utils.generate_cursor_mask_image(
            config, next_cursor_x, next_cursor_y
        )

        # Determine UI State for next_frame
        next_ui_state_name = state_logic.determine_ui_state(
            config, next_cursor_x, next_cursor_y, next_mouse_state
        )

        # Render next_frame
        next_frame_image = render_full_frame(
            config, next_ui_state_name, next_cursor_x, next_cursor_y
        )

        # Save Triplet
        frame_num_str = f"{frame_idx_for_triplet_output:04d}"
        prev_frame_image.save(os.path.join(episode_dir, f"{frame_num_str}_prev.png"))
        cursor_mask_image.save(os.path.join(episode_dir, f"{frame_num_str}_mask.png"))
        next_frame_image.save(os.path.join(episode_dir, f"{frame_num_str}_next.png"))

        # Update for next iteration
        prev_frame_image = next_frame_image