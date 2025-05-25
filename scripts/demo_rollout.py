# scripts/demo_rollout.py
import argparse, imageio, random, numpy as np
from pathlib import Path

import torch
from torchvision.transforms import ToTensor

import sys
sys.path.append("C:\\Users\\Chris\\Desktop\\ai_ui_project")
from src.train_autoencoder import Encoder, Decoder
from src.train_predictor   import TinyUNet

def load_masks(npz):
    return (torch.from_numpy(npz["cursor"]).float()[0],
            torch.from_numpy(npz["click" ]).float()[0],
            torch.from_numpy(npz["coord" ]).float()[0])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz_folder", required=True)
    p.add_argument("--encoder_pt", required=True)
    p.add_argument("--decoder_pt", required=True)
    p.add_argument("--predictor_pt", required=True)
    p.add_argument("--zdim",   type=int, default=64)
    p.add_argument("--roll",   type=int, default=20,
                   help="Number of steps to predict")
    p.add_argument("--out",    default="rollout.gif")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = Encoder(z_ch=args.zdim).to(device).eval()
    dec = Decoder(z_ch=args.zdim).to(device).eval()
    net = TinyUNet(args.zdim+1+1+2, args.zdim).to(device).eval()

    enc.load_state_dict(torch.load(args.encoder_pt, map_location=device))
    dec.load_state_dict(torch.load(args.decoder_pt, map_location=device))
    net.load_state_dict(torch.load(args.predictor_pt, map_location=device))

    files = sorted(Path(args.npz_folder).glob("*.npz"))
    d0    = np.load(random.choice(files))
    z_prev= torch.from_numpy(d0["z_prev"]).float()[0].unsqueeze(0).to(device)
    cursor, click, coord = [t.unsqueeze(0).to(device)
                            for t in load_masks(d0)]

    # initial frame
    with torch.no_grad():
        out0 = dec(z_prev)
    frames = [
        out0.squeeze(0).permute(1,2,0)      # still requires_grad=False
             .cpu()
             .detach()
             .numpy()
    ]

    for _ in range(args.roll):
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

if __name__ == "__main__":
    main()
