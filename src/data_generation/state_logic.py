# src/data_generation/state_logic.py
from PIL import Image # Only for type hinting if needed

def is_cursor_hotspot_inside_button(config, cursor_hotspot_x: int, cursor_hotspot_y: int) -> bool:
    """
    Checks if the cursor's hotspot (e.g., its top-left point) is inside the button's bounding box.
    """
    btn_x0, btn_y0, btn_x1, btn_y1 = config.BUTTON_RECT_PIL # (left, top, right, bottom)

    # Check if the hotspot point is within the button's rectangle
    # Note: The right (btn_x1) and bottom (btn_y1) edges are typically exclusive 
    # in such checks if you're thinking about pixel coordinates.
    # If BUTTON_RECT_PIL defines x1, y1 as (x0 + width), (y0 + height), 
    # then using '<' for x1 and y1 is correct for point-in-rectangle.
    if (btn_x0 <= cursor_hotspot_x < btn_x1 and
        btn_y0 <= cursor_hotspot_y < btn_y1):
        return True
    return False

# TODO: Implement more precise pixel-perfect sprite collision if BBox is not enough.
# For "any part of the cursor touches the button", pixel-perfect is better.
# is_cursor_intersecting_button_pixel_perfect(...)

def determine_ui_state(config, cursor_root_x, cursor_root_y, mouse_button_is_down_str):
    """
    Determines the UI state based on cursor position and mouse button state.
    mouse_button_is_down_str: "UP" or "DOWN"
    """
    # Given your cursor hotspot is (0,0), cursor_root_x and cursor_root_y 

    is_over = is_cursor_hotspot_inside_button(config, cursor_root_x, cursor_root_y)


    mouse_down = (mouse_button_is_down_str == "DOWN")

    if is_over and mouse_down:
        return "UI_STATE_PRESSED"
    elif is_over and not mouse_down:
        return "UI_STATE_HOVER"
    else: # not is_over (or not is_over and mouse_down, which defaults to normal)
        return "UI_STATE_NORMAL"