# scripts/debug_rollout.py
import argparse, imageio, random, numpy as np, math
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from PIL import Image

import sys
sys.path.append("C:\\Users\\Chris\\Desktop\\ai_ui_project")
from src.train_autoencoder import Encoder, Decoder
from src.train_predictor   import TinyUNet

def create_predetermined_path(total_steps, canvas_size=256):
    """
    Create a predetermined mouse path:
    1. Start from bottom right
    2. Move to button center (assume button is roughly in center)
    3. Click the button for a few frames
    4. Move away from button
    """
    # Load cursor to get actual size
    cursor_img = Image.open("assets/cursor.png")
    cursor_w, cursor_h = cursor_img.size
    
    # Assume button is roughly in the center area
    button_left = canvas_size // 2 - 60  # Assume button width ~120px
    button_right = canvas_size // 2 + 60
    button_top = canvas_size // 2 - 30   # Assume button height ~60px
    button_bottom = canvas_size // 2 + 30
    
    # Click position: 2/3 to the right of the button
    click_x = int(button_left + (button_right - button_left) * (2/3)) - cursor_w // 2
    click_y = (button_top + button_bottom) // 2 - cursor_h // 2
    
    # Start position (bottom right)
    start_x = canvas_size - cursor_w - 10
    start_y = canvas_size - cursor_h - 10
    
    # End position (top left area)
    end_x = 20
    end_y = 20
    
    # Phase breakdown:
    # Phase 1: Move to button (30% of steps)
    # Phase 2: Click button (40% of steps) 
    # Phase 3: Move away (30% of steps)
    
    phase1_steps = int(total_steps * 0.3)
    phase2_steps = int(total_steps * 0.4)
    phase3_steps = total_steps - phase1_steps - phase2_steps
    
    cursor_positions = []
    click_states = []
    
    print(f"Debug: Phase breakdown - Move: {phase1_steps}, Click: {phase2_steps}, Away: {phase3_steps}")
    print(f"Debug: Click position will be at ({click_x}, {click_y})")
    
    # Phase 1: Move to button
    for t in range(phase1_steps):
        alpha = t / (phase1_steps - 1) if phase1_steps > 1 else 0
        ease = 1 - math.cos(alpha * math.pi / 2)  # Ease-out
        
        x = int(start_x * (1 - ease) + click_x * ease)
        y = int(start_y * (1 - ease) + click_y * ease)
        
        cursor_positions.append((x, y))
        click_states.append(False)
    
    # Phase 2: Click button (stay at click position)
    for t in range(phase2_steps):
        cursor_positions.append((click_x, click_y))
        click_states.append(True)
        print(f"Debug: Frame {phase1_steps + t}: CLICKING at ({click_x}, {click_y})")
    
    # Phase 3: Move away from button
    for t in range(phase3_steps):
        alpha = t / (phase3_steps - 1) if phase3_steps > 1 else 0
        ease = 1 - math.cos(alpha * math.pi / 2)  # Ease-out
        
        x = int(click_x * (1 - ease) + end_x * ease)
        y = int(click_y * (1 - ease) + end_y * ease)
        
        cursor_positions.append((x, y))
        click_states.append(False)
    
    return cursor_positions, click_states

def create_masks(cursor_positions, click_states, canvas_size=256):
    """Create cursor and click masks for the predetermined path"""
    # Load cursor image
    cursor_img = Image.open("assets/cursor.png").convert("L")  # Convert to grayscale
    cursor_w, cursor_h = cursor_img.size
    cursor_array = np.array(cursor_img) / 255.0  # Normalize to 0-1
    cursor_array = 1.0 - cursor_array  # Invert: white cursor on black background
    
    cursor_masks = []
    click_masks = []
    
    for i, ((cx, cy), is_clicking) in enumerate(zip(cursor_positions, click_states)):
        # Create 256x256 cursor mask with actual cursor image
        cursor_mask = torch.zeros(1, 1, canvas_size, canvas_size)
        
        # Place cursor image at position (cx, cy)
        # Make sure we don't go out of bounds
        end_x = min(cx + cursor_w, canvas_size)
        end_y = min(cy + cursor_h, canvas_size)
        cursor_w_actual = end_x - cx
        cursor_h_actual = end_y - cy
        
        if cursor_w_actual > 0 and cursor_h_actual > 0:
            cursor_mask[0, 0, cy:end_y, cx:end_x] = torch.from_numpy(
                cursor_array[:cursor_h_actual, :cursor_w_actual]
            ).float()
        
        # Create 256x256 click state mask (full image white or black)
        click_mask = torch.zeros(1, 1, canvas_size, canvas_size)
        if is_clicking:
            click_mask.fill_(1.0)  # Fill entire mask with 1.0 when clicking
            print(f"Debug: Frame {i}: Click state ACTIVE (full image white)")
        
        # Downsample to 16x16 using max pooling
        cursor_16 = F.max_pool2d(cursor_mask, 16).half()
        click_16 = F.max_pool2d(click_mask, 16).half()
        
        cursor_masks.append(cursor_16)
        click_masks.append(click_16)
        
        # Save debug images for first few frames
        if i < 5 or is_clicking:
            # Save 256x256 versions
            cursor_img_debug = (cursor_mask[0, 0] * 255).byte().numpy()
            click_img_debug = (click_mask[0, 0] * 255).byte().numpy()
            
            Image.fromarray(cursor_img_debug, mode='L').save(f"debug_cursor_{i:03d}.png")
            Image.fromarray(click_img_debug, mode='L').save(f"debug_click_{i:03d}.png")
            
            # Save 16x16 versions
            cursor_16_img = (cursor_16[0, 0] * 255).byte().numpy()
            click_16_img = (click_16[0, 0] * 255).byte().numpy()
            
            Image.fromarray(cursor_16_img, mode='L').save(f"debug_cursor_16_{i:03d}.png")
            Image.fromarray(click_16_img, mode='L').save(f"debug_click_16_{i:03d}.png")
    
    return cursor_masks, click_masks

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz_folder", required=True, 
                   help="Folder with NPZ files (used only to get initial frame)")
    p.add_argument("--encoder_pt", required=True)
    p.add_argument("--decoder_pt", required=True)
    p.add_argument("--predictor_pt", required=True)
    p.add_argument("--zdim",   type=int, default=64)
    p.add_argument("--roll",   type=int, default=20,
                   help="Number of steps to predict")
    p.add_argument("--out",    default="debug_rollout.gif")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    enc = Encoder(z_ch=args.zdim).to(device).eval()
    dec = Decoder(z_ch=args.zdim).to(device).eval()
    net = TinyUNet(args.zdim+1+1+2, args.zdim).to(device).eval()

    enc.load_state_dict(torch.load(args.encoder_pt, map_location=device))
    dec.load_state_dict(torch.load(args.decoder_pt, map_location=device))
    net.load_state_dict(torch.load(args.predictor_pt, map_location=device))

    # Get initial frame from a random NPZ file
    files = sorted(Path(args.npz_folder).glob("*.npz"))
    chosen_file = random.choice(files)
    print(f"Using initial frame from: {chosen_file}")
    
    d0    = np.load(chosen_file)
    z_prev= torch.from_numpy(d0["z_prev"]).float()[0].unsqueeze(0).to(device)
    
    # Create coordinate grid (same as in preprocessing)
    coord = torch.cat([
        torch.linspace(0,1,16).view(1,1,16,1).expand(1,1,16,16),
        torch.linspace(0,1,16).view(1,1,1,16).expand(1,1,16,16)
    ], dim=1).half().to(device)

    # Generate predetermined path
    cursor_positions, click_states = create_predetermined_path(args.roll)
    cursor_masks, click_masks = create_masks(cursor_positions, click_states)

    # initial frame
    with torch.no_grad():
        out0 = dec(z_prev)
    frames = [
        out0.squeeze(0).permute(1,2,0)
             .cpu()
             .detach()
             .numpy()
    ]

    # Generate rollout with predetermined path
    for i in range(args.roll):
        cursor = cursor_masks[i].to(device)
        click = click_masks[i].to(device)
        
        # Debug: Print mask statistics
        cursor_sum = cursor.sum().item()
        click_sum = click.sum().item()
        print(f"Frame {i}: cursor_sum={cursor_sum:.3f}, click_sum={click_sum:.3f}")
        
        inp = torch.cat([z_prev, cursor, click, coord], dim=1)
        with torch.no_grad():
            z_next = net(inp)
            rgb    = dec(z_next)
        frames.append(
            rgb.squeeze(0)
               .permute(1,2,0)
               .cpu()
               .detach()
               .numpy()
        )
        z_prev = z_next

    # write GIF
    imageio.mimsave(
        args.out,
        (np.clip(np.stack(frames)*255, 0, 255)
           .astype(np.uint8)),
        fps=10
    )
    print("GIF saved →", args.out)
    print(f"Path: bottom-right → button → click → top-left over {args.roll} steps")
    print("Debug images saved as debug_*.png")

if __name__ == "__main__":
    main() 