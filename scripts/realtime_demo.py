#!/usr/bin/env python
"""
Real-time interactive UI demo using your AE + frame predictor,
with optional user-specified starting frame and cursor-mask semantics matching training.
"""

import argparse
import random
import sys

import numpy as np
import pygame
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision.transforms import ToTensor

# adjust your project path as needed:
sys.path.append("C:\\Users\\Chris\\Desktop\\ai_ui_project")
from src.train_autoencoder import Encoder, Decoder
from src.train_predictor   import TinyUNet

# Window dimensions
W, H = 256, 256

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder_pt",    required=True,
                   help="Path to the AE encoder checkpoint (.pt)")
    p.add_argument("--decoder_pt",    required=True,
                   help="Path to the AE decoder checkpoint (.pt)")
    p.add_argument("--predictor_pt",  required=True,
                   help="Path to the frame predictor checkpoint (.pt)")
    p.add_argument("--zdim",          type=int, default=64,
                   help="Latent dimension (must match your models)")
    p.add_argument("--start-frame",   dest="start_frame", default=None,
                   help="(Optional) Path to a 256×256 RGB image to encode as the initial frame")
    return p.parse_args()

if __name__ == "__main__":
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load models ---
    enc  = Encoder(z_ch=args.zdim).to(device).eval()
    dec  = Decoder(z_ch=args.zdim).to(device).eval()
    pred = TinyUNet(args.zdim + 1 + 1 + 2, args.zdim).to(device).eval()

    enc.load_state_dict(torch.load(args.encoder_pt,   map_location=device))
    dec.load_state_dict(torch.load(args.decoder_pt,   map_location=device))
    pred.load_state_dict(torch.load(args.predictor_pt, map_location=device))

    # --- Precompute coordinate grid [1,2,16,16] ---
    coord = torch.cat([
        torch.linspace(0,1,16,device=device).view(1,1,16,1).expand(1,1,16,16),
        torch.linspace(0,1,16,device=device).view(1,1,1,16).expand(1,1,16,16)
    ], dim=1)

    # --- Load cursor sprite & build binary mask sprite ---
    cursor_sprite       = Image.open("assets/cursor.png").convert("RGBA")
    cursor_mask_sprite  = cursor_sprite.split()[-1]  # alpha channel as L mask
    toT = ToTensor()

    # --- Determine initial latent z_prev ---
    if args.start_frame:
        # Encode user-provided image
        img = Image.open(args.start_frame).convert("RGB")
        img_t = toT(img).unsqueeze(0).to(device)       # [1,3,256,256]
        with torch.no_grad():
            z_prev = enc(img_t)                       # [1,zdim,16,16]
    else:
        # Fallback: pick a random latent from your precomputed .npz files
        import glob
        files = glob.glob("C:/Users/Chris/Desktop/ai_ui_project/latent_triplets/*.npz")
        if not files:
            print("❌ No .npz files found; run prepare_latent_dataset first.")
            sys.exit(1)
        d0     = np.load(random.choice(files))
        # d0["z_prev"] is shape (1,zdim,16,16)
        z_prev = torch.from_numpy(d0["z_prev"]).float().to(device)

    # --- Initialize PyGame ---
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Real-time AI UI Demo")
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont(None, 24)  # For FPS display

    running = True
    while running:
        # Handle quit
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

        # Mouse state
        mx, my = pygame.mouse.get_pos()
        mb     = pygame.mouse.get_pressed()[0]  # left button down?

        # --- Build full-res 256×256 cursor mask and pool to 16×16 ---
        mask_img = Image.new("L", (W, H), 0)
        mask_img.paste(cursor_mask_sprite, (int(mx), int(my)))
        pos256   = toT(mask_img).unsqueeze(0).to(device)  # [1,1,256,256]
        cursor16 = F.max_pool2d(pos256, 16)               # [1,1,16,16]

        # --- Click mask as uniform tile [1,1,16,16] ---
        click16 = torch.full((1,1,16,16), float(mb), device=device)

        # --- Stack into predictor input [1, C_in,16,16] ---
        inp = torch.cat([
            z_prev,       # [1,zdim,16,16]
            cursor16,     # [1,1,16,16]
            click16,      # [1,1,16,16]
            coord         # [1,2,16,16]
        ], dim=1)        # → [1, zdim+1+1+2, 16,16]

        # --- Predict & decode ---
        with torch.no_grad():
            z_next  = pred(inp)       # [1,zdim,16,16]
            out_rgb = dec(z_next)     # [1,3,256,256]
        z_prev = z_next               # roll forward

        # Convert to displayable surface
        img = out_rgb.squeeze(0).permute(1,2,0).cpu().numpy()
        img = (img * 255).clip(0,255).astype(np.uint8)
        img = np.transpose(img, (1,0,2))  # to (width,height,3)
        surf = pygame.surfarray.make_surface(img)

        screen.blit(surf, (0,0))
        fps_text = font.render(f"FPS: {clock.get_fps():.1f}", True, (255,255,255))
        screen.blit(fps_text, (10,10))
        pygame.display.flip()
        clock.tick()  # cap at 60 FPS

    pygame.quit()
