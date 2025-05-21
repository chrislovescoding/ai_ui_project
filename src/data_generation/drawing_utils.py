# src/data_generation/drawing_utils.py
from PIL import Image, ImageDraw, ImageFont

def get_text_dimensions(draw_context, text, font):
    # For Pillow >= 9.0.0, textbbox is preferred
    if hasattr(draw_context, 'textbbox'):
        bbox = draw_context.textbbox((0,0), text, font=font) # xy, text, font
        return bbox[2] - bbox[0], bbox[3] - bbox[1] # width, height
    else: # Fallback for older Pillow versions
        return draw_context.textsize(text, font=font) # width, height

def draw_base_window(image_draw_context, config):
    image_draw_context.rectangle(
        [0, 0, config.WINDOW_WIDTH, config.WINDOW_HEIGHT],
        fill=config.WINDOW_BG_COLOR
    )

def draw_static_text(image_draw_context, config): # "Chris Lock"
    font = config.loaded_assets.static_text_font
    text = config.STATIC_TEXT_CONTENT
    text_w, text_h = get_text_dimensions(image_draw_context, text, font)
    
    # For baseline alignment, Pillow's text anchor might be complex.
    # A common approach is to position based on text_h for bottom alignment.
    # Pillow draws text with (0,0) at top-left of text bbox by default.
    # If margin_bottom_px is to baseline, more complex font metrics are needed.
    # Assuming margin_bottom_px is to bottom of text bounding box for simplicity.
    x = (config.WINDOW_WIDTH - text_w) / 2
    y = config.WINDOW_HEIGHT - config.STATIC_TEXT_MARGIN_BOTTOM_PX - text_h 
    
    image_draw_context.text((x, y), text, font=font, fill=config.STATIC_TEXT_COLOR)

def draw_button(image_draw_context, config, button_fill_color):
    # Draw rounded rectangle for button
    image_draw_context.rounded_rectangle(
        config.BUTTON_RECT_PIL,
        radius=config.BUTTON_CORNER_RADIUS,
        fill=button_fill_color
    )
    # Draw button label
    font = config.loaded_assets.button_label_font
    text = config.BUTTON_LABEL_TEXT
    text_w, text_h = get_text_dimensions(image_draw_context, text, font)
    
    btn_x, btn_y, btn_x1, btn_y1 = config.BUTTON_RECT_PIL
    btn_w = btn_x1 - btn_x
    btn_h = btn_y1 - btn_y
    
    text_x = btn_x + (btn_w - text_w) / 2
    text_y = btn_y + (btn_h - text_h) / 2 
    
    image_draw_context.text((text_x, text_y), text, font=font, fill=config.BUTTON_TEXT_COLOR)

def draw_pressed_indicator_text(image_draw_context, config): # "Pressed!"
    font = config.loaded_assets.pressed_indicator_font
    text = config.PRESSED_INDICATOR_TEXT_CONTENT
    text_w, text_h = get_text_dimensions(image_draw_context, text, font)

    btn_x, btn_y, btn_x1, _ = config.BUTTON_RECT_PIL # Use button x, x1 for centering
    btn_center_x = btn_x + (btn_x1 - btn_x) / 2
    
    text_x = btn_center_x - text_w / 2
    # Position bottom of text "margin_above_button_px" above button's top (btn_y)
    text_y = btn_y - config.PRESSED_INDICATOR_MARGIN_ABOVE_BUTTON_PX - text_h
    
    image_draw_context.text((text_x, text_y), text, font=font, fill=config.PRESSED_INDICATOR_TEXT_COLOR)

def draw_cursor(image_obj, config, cursor_root_x, cursor_root_y):
    """ Pastes the cursor sprite onto the image_obj. """
    cursor_sprite = config.loaded_assets.cursor_sprite_image
    # Ensure integer coordinates for paste
    paste_pos = (int(cursor_root_x), int(cursor_root_y))
    
    # Use alpha compositing if image_obj is RGBA, otherwise simple paste with mask
    if image_obj.mode == 'RGBA':
        # Create a temporary RGBA image for the cursor to ensure correct alpha blending
        temp_cursor_layer = Image.new('RGBA', image_obj.size, (0,0,0,0))
        temp_cursor_layer.paste(cursor_sprite, paste_pos, cursor_sprite) # Use cursor's own alpha as mask
        image_obj.alpha_composite(temp_cursor_layer)
    else: # Assuming image_obj is RGB
        image_obj.paste(cursor_sprite, paste_pos, cursor_sprite) # Use cursor's own alpha as mask for RGBA -> RGB

def generate_cursor_mask_image(config, cursor_root_x, cursor_root_y):
    """ Generates a single-channel image with 255 where the cursor sprite will be. """
    mask_image = Image.new('L', (config.WINDOW_WIDTH, config.WINDOW_HEIGHT), 0) # Black background
    cursor_sprite_mask = config.loaded_assets.cursor_sprite_mask # Pre-calculated binary mask
    
    paste_pos = (int(cursor_root_x), int(cursor_root_y))
    mask_image.paste(cursor_sprite_mask, paste_pos) # Paste the binary mask
    return mask_image