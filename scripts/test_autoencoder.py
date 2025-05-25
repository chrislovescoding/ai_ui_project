# scripts/test_autoencoder.py
import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import ToTensor

# adjust the import path if needed
import sys
sys.path.append("C:\\Users\\Chris\\Desktop\\ai_ui_project")
from src.train_autoencoder import Encoder, Decoder

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",    required=True,
                   help="Folder with exhaustive PNGs")
    p.add_argument("--encoder_pt",   required=True)
    p.add_argument("--decoder_pt",   required=True)
    p.add_argument("--zdim",         type=int,   required=True,
                   help="Latent-channel count (must match how you trained)")
    p.add_argument("--num",          type=int,   default=10,
                   help="Number of random samples to visualize")
    p.add_argument("--output",       default="recon_test.png",
                   help="Where to save the visualization")
    args = p.parse_args()

    # gather paths
    imgs = list(Path(args.data_root).glob("*.png"))
    if len(imgs) < args.num:
        raise RuntimeError(f"Need ≥{args.num} images in {args.data_root}")
    sample = random.sample(imgs, args.num)

    # device & model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = Encoder(z_ch=args.zdim).to(device).eval()
    dec = Decoder(z_ch=args.zdim).to(device).eval()
    enc.load_state_dict(torch.load(args.encoder_pt, map_location=device))
    dec.load_state_dict(torch.load(args.decoder_pt, map_location=device))

    toT = ToTensor()

    # prepare figure
    fig, axes = plt.subplots(2, args.num, figsize=(args.num*2, 4))
    for i, path in enumerate(sample):
        # load & preprocess
        img = toT(Image.open(path).convert("RGB")).unsqueeze(0).to(device)

        # encode & decode
        with torch.no_grad():
            z     = enc(img)
            recon = dec(z)

        # move to CPU & numpy
        orig_np  = img.squeeze(0).cpu().permute(1,2,0).numpy()
        recon_np = recon.squeeze(0).cpu().permute(1,2,0).numpy()

        # plot
        axes[0,i].imshow(orig_np)
        axes[0,i].axis("off")
        axes[1,i].imshow(recon_np)
        axes[1,i].axis("off")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    plt.show()
    print(f"Saved visualization to {args.output}")

if __name__ == "__main__":
    main()
