# src/data_generation/config.py
import yaml
from PIL import ImageFont, Image

class Config:
    def __init__(self, yaml_path):
        with open(yaml_path, 'r') as f:
            self.data = yaml.safe_load(f)

        # Window
        self.WINDOW_WIDTH = self.data['window']['width']
        self.WINDOW_HEIGHT = self.data['window']['height']
        self.WINDOW_BG_COLOR = tuple(self.data['window']['background_color_rgb'])

        # Button
        self.BUTTON_RECT_XYWH = tuple(self.data['button']['rect_xywh']) # x, y, w, h
        self.BUTTON_CORNER_RADIUS = self.data['button']['corner_radius']
        self.BUTTON_LABEL_TEXT = self.data['button']['label_text']
        self.BUTTON_FONT_PATH = self.data['button']['font_path']
        self.BUTTON_FONT_SIZE_PT = self.data['button']['font_size_pt']
        self.BUTTON_TEXT_COLOR = tuple(self.data['button']['text_color_rgb'])
        self.BUTTON_NORMAL_FILL_COLOR = tuple(self.data['button']['normal_fill_rgb'])
        self.BUTTON_HOVER_FILL_COLOR = tuple(self.data['button']['hover_fill_rgb'])
        self.BUTTON_PRESSED_FILL_COLOR = tuple(self.data['button']['pressed_fill_rgb'])

        # Static Text ("Chris Lock")
        self.STATIC_TEXT_CONTENT = self.data['static_text']['content']
        self.STATIC_TEXT_FONT_PATH = self.data['static_text']['font_path']
        self.STATIC_TEXT_FONT_SIZE_PT = self.data['static_text']['font_size_pt']
        self.STATIC_TEXT_COLOR = tuple(self.data['static_text']['color_rgb'])
        self.STATIC_TEXT_MARGIN_BOTTOM_PX = self.data['static_text']['margin_bottom_px']
        
        # Pressed Indicator Text ("Pressed!")
        self.PRESSED_INDICATOR_TEXT_CONTENT = self.data['pressed_indicator_text']['content']
        self.PRESSED_INDICATOR_TEXT_FONT_PATH = self.data['pressed_indicator_text']['font_path']
        self.PRESSED_INDICATOR_TEXT_FONT_SIZE_PT = self.data['pressed_indicator_text']['font_size_pt']
        self.PRESSED_INDICATOR_TEXT_COLOR = tuple(self.data['pressed_indicator_text']['color_rgb'])
        self.PRESSED_INDICATOR_MARGIN_ABOVE_BUTTON_PX = self.data['pressed_indicator_text']['margin_above_button_px']

        # Cursor
        self.CURSOR_SPRITE_PATH = self.data['cursor']['sprite_path']
        self.CURSOR_HOTSPOT_OFFSET_XY = tuple(self.data['cursor']['hotspot_offset_xy'])

        # Generation Params
        self.FRAMES_PER_EPISODE = self.data['generation_params']['frames_per_episode']
        self.NUM_EVERYDAY_EPISODES = self.data['generation_params']['num_everyday_episodes']
        self.NUM_EDGE_CASE_EPISODES = self.data['generation_params']['num_edge_case_episodes']
        self.TOTAL_EPISODES = self.NUM_EVERYDAY_EPISODES + self.NUM_EDGE_CASE_EPISODES
        self.OUTPUT_DIR_BASE = self.data['generation_params']['output_dir_base']
        self.NUM_WORKERS = self.data['generation_params']['num_workers']

        # Derived button rect for Pillow (x0, y0, x1, y1)
        bx, by, bw, bh = self.BUTTON_RECT_XYWH
        self.BUTTON_RECT_PIL = (bx, by, bx + bw, by + bh)
        
        self.loaded_assets = None # To be populated by load_assets

class Assets:
    def __init__(self):
        self.button_label_font = None
        self.static_text_font = None
        self.pressed_indicator_font = None
        self.cursor_sprite_image = None
        self.cursor_sprite_mask = None # Pre-calculated alpha mask for cursor sprite
        self.cursor_width = 0
        self.cursor_height = 0

def load_config_and_assets(yaml_path="assets/design_spec.yaml"):
    cfg = Config(yaml_path)
    
    assets = Assets()
    try:
        assets.button_label_font = ImageFont.truetype(cfg.BUTTON_FONT_PATH, cfg.BUTTON_FONT_SIZE_PT)
        assets.static_text_font = ImageFont.truetype(cfg.STATIC_TEXT_FONT_PATH, cfg.STATIC_TEXT_FONT_SIZE_PT)
        assets.pressed_indicator_font = ImageFont.truetype(cfg.PRESSED_INDICATOR_TEXT_FONT_PATH, cfg.PRESSED_INDICATOR_TEXT_FONT_SIZE_PT)
        
        raw_cursor_img = Image.open(cfg.CURSOR_SPRITE_PATH).convert("RGBA")
        assets.cursor_sprite_image = raw_cursor_img
        assets.cursor_width, assets.cursor_height = raw_cursor_img.size
        
        # Create a binary mask from cursor alpha channel
        # This mask is 0 where transparent, 255 where opaque
        if raw_cursor_img.mode == 'RGBA':
            alpha = raw_cursor_img.split()[-1]
            assets.cursor_sprite_mask = alpha.point(lambda i: 255 if i > 128 else 0, '1').convert('L')
        else: # Assuming non-alpha image is fully opaque
            assets.cursor_sprite_mask = Image.new('L', raw_cursor_img.size, 255)

    except FileNotFoundError as e:
        print(f"Error loading assets: {e}. Make sure font and cursor files exist.")
        print("Please ensure 'assets/DejaVuSans.ttf' and 'assets/cursor.png' are present.")
        raise
    except OSError as e: # Handles issues with font files specifically
        print(f"Error loading font: {e}. Is '{cfg.BUTTON_FONT_PATH}' a valid font file?")
        raise

    cfg.loaded_assets = assets
    return cfg

if __name__ == '__main__':
    # Test loading
    try:
        cfg_instance = load_config_and_assets()
        print("Config and assets loaded successfully.")
        print(f"Button rect: {cfg_instance.BUTTON_RECT_PIL}")
        print(f"Cursor size: {cfg_instance.loaded_assets.cursor_width}x{cfg_instance.loaded_assets.cursor_height}")
    except Exception as e:
        print(f"Failed to load config: {e}")