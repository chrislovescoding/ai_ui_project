#!/usr/bin/env python
"""
Encode every (prev, next, cursor mask, click mask) triplet into compact NPZs,
with verbose logging of missing files and correct index extraction.
"""
import argparse, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor

import sys
sys.path.append("C:\\Users\\Chris\\Desktop\\ai_ui_project")
from src.train_autoencoder import Encoder

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes-root", required=True,
                   help="Root folder containing ep_xxxxx subfolders")
    p.add_argument("--encoder-pt",    required=True,
                   help="Path to ae_encoder.pt")
    p.add_argument("--zdim",          type=int, default=64)
    p.add_argument("--out",           required=True,
                   help="Where to write *.npz files")
    args = p.parse_args()

    ROOT = Path(args.episodes_root)
    OUT  = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)

    prev_list = sorted(ROOT.rglob("*_prev.png"))
    print(f"Looking under {ROOT} for *_prev.png → found {len(prev_list)} files")
    if not prev_list:
        print("❌ No *_prev.png files found. Check your episodes-root.")
        sys.exit(1)

    # Load encoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = Encoder(z_ch=args.zdim).to(device).eval()
    enc.load_state_dict(torch.load(args.encoder_pt, map_location=device))
    toT = ToTensor()

    # Precompute 16×16 coord grid
    coord = torch.cat([
        torch.linspace(0,1,16).view(1,1,16,1).expand(1,1,16,16),
        torch.linspace(0,1,16).view(1,1,1,16).expand(1,1,16,16)
    ], dim=1).half().to(device)

    count_ok = 0
    for prev_path in prev_list:
        ep_dir = prev_path.parent.name
        stem   = prev_path.stem           # e.g. "0016_prev"
        if not stem.endswith("_prev"):
            print(f"⚠️  Unexpected filename {prev_path.name}, skipping.")
            continue
        idx = stem[:-5]                   # remove "_prev" suffix → "0016"

        # build the exact filenames we expect
        next_name  = f"{idx}_next.png"
        pos_name   = f"{idx}_cursor_pos_mask.png"
        click_name = f"{idx}_click_state_mask.png"

        next_p   = prev_path.with_name(next_name)
        pos_p    = prev_path.with_name(pos_name)
        click_p  = prev_path.with_name(click_name)

        missing = []
        for tag, pth in [("next", next_p), ("cursor mask", pos_p), ("click mask", click_p)]:
            if not pth.exists():
                missing.append(f"{tag} ({pth.name})")

        if missing:
            print(f"⚠️  In {ep_dir}/{prev_path.name}: missing {', '.join(missing)} → skipping.")
            continue

        # All files exist → encode
        prev_img = toT(Image.open(prev_path).convert("RGB")).unsqueeze(0).to(device)
        next_img = toT(Image.open(next_p   ).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            z_prev = enc(prev_img).half()
            z_next = enc(next_img).half()

        pos256 = toT(Image.open(pos_p).convert("L")).unsqueeze(0).to(device)
        clk256 = toT(Image.open(click_p).convert("L")).unsqueeze(0).to(device)
        pos16  = F.max_pool2d(pos256,16).half()
        clk16  = F.max_pool2d(clk256,16).half()

        out_fn = OUT / f"{ep_dir}_{idx}.npz"
        np.savez_compressed(
            out_fn,
            z_prev=z_prev.cpu().numpy(),
            z_next=z_next.cpu().numpy(),
            cursor=pos16.cpu().numpy(),
            click =clk16.cpu().numpy(),
            coord =coord.cpu().numpy()
        )
        count_ok += 1
        if count_ok % 1000 == 0:
            print(f"  ✓ Encoded {count_ok} triplets so far...")

    print(f"✅ Done. Successfully encoded {count_ok} triplets to {OUT}")

if __name__ == "__main__":
    main()
