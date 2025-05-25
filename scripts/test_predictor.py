import argparse, random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid

import sys
sys.path.append("C:\\Users\\Chris\\Desktop\\ai_ui_project")
from src.train_autoencoder import Decoder, Encoder
from src.train_predictor   import TinyUNet

# ── helper to load one npz & build predictor input ───────────────────────────
def build_input(npz, zdim, device):
    z_prev = torch.from_numpy(npz["z_prev"]).float()[0]
    cursor = torch.from_numpy(npz["cursor"]).float()[0]
    click  = torch.from_numpy(npz["click" ]).float()[0]
    coord  = torch.from_numpy(npz["coord" ]).float()[0]
    inp = torch.cat([z_prev, cursor, click, coord], dim=0)   # [C_in,16,16]
    return inp.unsqueeze(0).to(device)                       # [1,C_in,16,16]

# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz_folder", required=True)
    p.add_argument("--encoder_pt", required=True)
    p.add_argument("--decoder_pt", required=True)
    p.add_argument("--predictor_pt", required=True)
    p.add_argument("--zdim", type=int, default=64)
    p.add_argument("--num",  type=int, default=5)
    p.add_argument("--out",  default="predictor_test.png")
    args = p.parse_args()

    files = sorted(Path(args.npz_folder).glob("*.npz"))
    assert len(files) >= args.num, "Not enough .npz files!"
    sample = random.sample(files, args.num)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = Encoder(z_ch=args.zdim).to(device).eval()
    dec = Decoder(z_ch=args.zdim).to(device).eval()
    net = TinyUNet(args.zdim+1+1+2, args.zdim).to(device).eval()

    enc.load_state_dict(torch.load(args.encoder_pt, map_location=device))
    dec.load_state_dict(torch.load(args.decoder_pt, map_location=device))
    net.load_state_dict(torch.load(args.predictor_pt, map_location=device))

    rows = []
    for f in sample:
        d = np.load(f)
        # gt next frame for visual
        gt_next = torch.from_numpy(d["z_next"]).float()[0].unsqueeze(0).to(device)
        gt_rgb  = dec(gt_next).squeeze(0)

        # predictor output
        inp = build_input(d, args.zdim, device)
        with torch.no_grad():
            pred_lat = net(inp)
            pred_rgb = dec(pred_lat).squeeze(0)

        rows += [gt_rgb, pred_rgb]   # append two 3×256×256 tensors

    grid = make_grid(torch.stack(rows), nrow=2)  # alternating GT / Pred
    plt.imsave(args.out, grid.permute(1,2,0).cpu().numpy())
    print("Saved comparison grid →", args.out)

if __name__ == "__main__":
    main()
