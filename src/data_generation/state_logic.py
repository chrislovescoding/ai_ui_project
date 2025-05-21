# src/data_generation/state_logic.py
from PIL import Image # Only for type hinting if needed

def is_cursor_intersecting_button_bbox(config, cursor_root_x, cursor_root_y):
    """ Basic BBox collision check. """
    assets = config.loaded_assets
    cursor_x0 = cursor_root_x
    cursor_y0 = cursor_root_y
    cursor_x1 = cursor_root_x + assets.cursor_width
    cursor_y1 = cursor_root_y + assets.cursor_height

    btn_x0, btn_y0, btn_x1, btn_y1 = config.BUTTON_RECT_PIL

    # Check for non-overlap
    if cursor_x1 <= btn_x0 or cursor_x0 >= btn_x1 or \
       cursor_y1 <= btn_y0 or cursor_y0 >= btn_y1:
        return False
    return True

# TODO: Implement more precise pixel-perfect sprite collision if BBox is not enough.
# For "any part of the cursor touches the button", pixel-perfect is better.
# is_cursor_intersecting_button_pixel_perfect(...)

def determine_ui_state(config, cursor_root_x, cursor_root_y, mouse_button_is_down_str):
    """
    Determines the UI state based on cursor position and mouse button state.
    mouse_button_is_down_str: "UP" or "DOWN"
    """
    # Using BBox collision for now.
    is_over = is_cursor_intersecting_button_bbox(config, cursor_root_x, cursor_root_y)
    
    mouse_down = (mouse_button_is_down_str == "DOWN")

    if is_over and mouse_down:
        return "UI_STATE_PRESSED"
    elif is_over and not mouse_down:
        return "UI_STATE_HOVER"
    else: # not is_over
        return "UI_STATE_NORMAL"