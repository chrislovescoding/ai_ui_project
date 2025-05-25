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
    episode_dir = os.path.join(config.OUTPUT_DIR_BASE, f"ep_{episode_id:05d}")
    os.makedirs(episode_dir, exist_ok=True)

    trajectory = trajectory_callable(config)

    try:
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

    for frame_idx_for_triplet_output in range(config.FRAMES_PER_EPISODE - 1):
        try:
            next_cursor_x, next_cursor_y, next_mouse_state = next(trajectory) # mouse_state for t+1
        except StopIteration:
            # print(f"Warning: Trajectory for ep {episode_id} ended early.") # Optional warning
            break

        # Generate Positional Cursor Mask for next_frame's cursor position
        # This uses your existing function, which is fine.
        cursor_pos_mask_image = drawing_utils.generate_cursor_mask_image(
            config, next_cursor_x, next_cursor_y
        )

        # --- NEW: Generate Click State Mask for next_frame's mouse state ---
        # This mask reflects `next_mouse_state` ("UP" or "DOWN")
        click_state_mask_image = drawing_utils.generate_click_state_mask_image(
            config, next_mouse_state  # Pass the mouse state for the next frame
        )
        # --- END NEW ---

        next_ui_state_name = state_logic.determine_ui_state(
            config, next_cursor_x, next_cursor_y, next_mouse_state
        )
        next_frame_image = render_full_frame(
            config, next_ui_state_name, next_cursor_x, next_cursor_y
        )

        # Save Triplet (with updated mask filenames)
        frame_num_str = f"{frame_idx_for_triplet_output:04d}"
        prev_frame_image.save(os.path.join(episode_dir, f"{frame_num_str}_prev.png"))
        
        # --- MODIFIED: Save both masks with new names ---
        cursor_pos_mask_image.save(os.path.join(episode_dir, f"{frame_num_str}_cursor_pos_mask.png")) # Renamed from _mask.png
        click_state_mask_image.save(os.path.join(episode_dir, f"{frame_num_str}_click_state_mask.png")) # New click state mask file
        # --- END MODIFIED ---
        
        next_frame_image.save(os.path.join(episode_dir, f"{frame_num_str}_next.png"))

        prev_frame_image = next_frame_image