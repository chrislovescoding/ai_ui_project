#!/usr/bin/env python
"""
Real-time interactive UI demo using your AE + frame predictor.

Usage:
  python scripts/realtime_demo.py \
    --encoder_pt   "models/ae_v4/ae_encoder.pt" \
    --decoder_pt   "models/ae_v4/ae_decoder.pt" \
    --predictor_pt "models/predictor_v1/predictor.pt" \
    --zdim 64
"""

import argparse
import random
import sys

import numpy as np
import pygame
import torch
import torch.nn.functional as F

import sys
sys.path.append("C:\\Users\\Chris\\Desktop\\ai_ui_project")
from src.train_autoencoder import Encoder, Decoder
from src.train_predictor   import TinyUNet

# Window dimensions
W, H = 256, 256

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder_pt",   required=True)
    p.add_argument("--decoder_pt",   required=True)
    p.add_argument("--predictor_pt", required=True)
    p.add_argument("--zdim",         type=int, default=64)
    return p.parse_args()

def build_mask16(x, y):
    """Return a [1,16,16] float mask with a 1 in the tile containing (x,y)."""
    tx = min(15, max(0, int(x / 16)))
    ty = min(15, max(0, int(y / 16)))
    m = torch.zeros((1,16,16), dtype=torch.float32, device=device)
    m[0,ty,tx] = 1.0
    return m

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    enc = Encoder(z_ch=args.zdim).to(device).eval()
    dec = Decoder(z_ch=args.zdim).to(device).eval()
    pred = TinyUNet(args.zdim+1+1+2, args.zdim).to(device).eval()

    enc.load_state_dict(torch.load(args.encoder_pt,   map_location=device))
    dec.load_state_dict(torch.load(args.decoder_pt,   map_location=device))
    pred.load_state_dict(torch.load(args.predictor_pt, map_location=device))

    # Precompute coordinate grid [1,2,16,16]
    coord = torch.cat([
        torch.linspace(0,1,16,device=device).view(1,1,16,1).expand(1,1,16,16),
        torch.linspace(0,1,16,device=device).view(1,1,1,16).expand(1,1,16,16)
    ], dim=1)

    # Pick a random starting latent from your npz dataset
    # so UI begins in a valid state.
    import glob
    files = glob.glob("C:/Users/Chris/Desktop/ai_ui_project/latent_triplets/*.npz")
    if not files:
        print("❌ No .npz files found; run prepare_latent_dataset first.")
        sys.exit(1)
    d0      = np.load(random.choice(files))
    z_prev  = torch.from_numpy(d0["z_prev"]).float()[0].unsqueeze(0).to(device)

    # Initialize PyGame
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Real-time AI UI Demo")
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont(None, 24)  # Add font for FPS display

    running = True
    while running:
        # Handle quit
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

        # Read mouse state
        mx, my = pygame.mouse.get_pos()
        mb      = pygame.mouse.get_pressed()[0]  # left button

        # Build masks at 16×16 resolution
        cursor_mask = build_mask16(mx, my)      # [1,16,16]
        click_mask  = torch.full((1,16,16), float(mb), device=device)

        # Prepare predictor input: [1, C_in,16,16]
        inp = torch.cat([ z_prev, 
                          cursor_mask.unsqueeze(0), 
                          click_mask.unsqueeze(0),
                          coord ], dim=1)

        with torch.no_grad():
            z_next = pred(inp)              # [1, zdim,16,16]
            out_rgb = dec(z_next)           # [1,3,256,256]
        z_prev = z_next                     # roll

        # Convert to 8-bit image for display
        # Convert to 8-bit image for display
        img = out_rgb.squeeze(0).permute(1,2,0).cpu().numpy()
        img = (img * 255).clip(0,255).astype(np.uint8)

        # Pygame wants (width, height, 3) not (height, width, 3)
        img = np.transpose(img, (1, 0, 2))   # swap X<->Y axes

        surf = pygame.surfarray.make_surface(img)


        # Draw & flip
        screen.blit(surf, (0,0))

        # --- Display FPS in the top-left corner ---
        fps = clock.get_fps()
        fps_text = font.render(f"FPS: {fps:.1f}", True, (255,255,255))
        screen.blit(fps_text, (10, 10))
        # -----------------------------------------

        pygame.display.flip()
        clock.tick()  # cap at 60 FPS

    pygame.quit()
