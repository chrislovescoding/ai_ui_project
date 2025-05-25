"""
Train a convolutional Auto-Encoder on the exhaustive UI atlas with:
 - combined MSE + L1 loss
 - linear warm-up + cosine LR schedule
 - up to EPOCHS epochs (early-stop at PSNR target)

Usage:
    python -m src.train_autoencoder \
      --data_root     "C:/path/to/exhaustive_data/images" \
      --out_dir       "C:/path/to/models/ae_with_l1" \
      --epochs        50 \
      --warmup_epochs 5 \
      --zdim          64 \
      --batch         256 \
      --lr            3e-4 \
      --l1_weight     0.1 \
      --psnr_target   40.0
"""
import argparse, json, math, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose
import torchvision.utils as vutils
from PIL import Image

# -----------------------------------------------------------------------------
# 1 · DATASET
# -----------------------------------------------------------------------------
class PNGFolder(Dataset):
    """Loads every .png under data_root (flat)."""
    def __init__(self, data_root, transform=None):
        self.paths = sorted(Path(data_root).glob("*.png"))
        if not self.paths:
            raise RuntimeError(f"No PNGs found in {data_root}")
        self.transform = transform or ToTensor()
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)

# -----------------------------------------------------------------------------
# 2 · MODEL
# -----------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, in_ch=3, base_ch=32, z_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,       base_ch,   4,2,1), nn.ReLU(True),
            nn.Conv2d(base_ch,     base_ch,   3,1,1), nn.ReLU(True),
            nn.Conv2d(base_ch,     base_ch*2, 4,2,1), nn.ReLU(True),
            nn.Conv2d(base_ch*2,   base_ch*2, 3,1,1), nn.ReLU(True),
            nn.Conv2d(base_ch*2,   base_ch*4, 4,2,1), nn.ReLU(True),
            nn.Conv2d(base_ch*4,   base_ch*4, 3,1,1), nn.ReLU(True),
            nn.Conv2d(base_ch*4,   z_ch,      4,2,1),                 # 16×16×z_ch
        )
    def forward(self, x): return self.net(x)

class Decoder(nn.Module):
    def __init__(self, out_ch=3, base_ch=32, z_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_ch,     base_ch*4, 4,2,1), nn.ReLU(True),
            nn.Conv2d(base_ch*4,         base_ch*4, 3,1,1), nn.ReLU(True),
            nn.ConvTranspose2d(base_ch*4, base_ch*2, 4,2,1), nn.ReLU(True),
            nn.Conv2d(base_ch*2,         base_ch*2, 3,1,1), nn.ReLU(True),
            nn.ConvTranspose2d(base_ch*2, base_ch,   4,2,1), nn.ReLU(True),
            nn.Conv2d(base_ch,           base_ch,   3,1,1), nn.ReLU(True),
            nn.ConvTranspose2d(base_ch,   out_ch,   4,2,1),
            nn.Sigmoid(),                                   # [0,1]
        )
    def forward(self, z): return self.net(z)

class AutoEncoder(nn.Module):
    def __init__(self, z_ch=64):
        super().__init__()
        self.enc = Encoder(z_ch=z_ch)
        self.dec = Decoder(z_ch=z_ch)
    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z

# -----------------------------------------------------------------------------
# 3 · METRIC
# -----------------------------------------------------------------------------
def psnr_from_mse(mse: float) -> float:
    return 10 * math.log10(1.0 / max(mse, 1e-12))

# -----------------------------------------------------------------------------
# 4 · TRAIN LOOP
# -----------------------------------------------------------------------------
def train_autoencoder(args):
    # reproducibility
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data
    ds = PNGFolder(args.data_root, transform=Compose([ToTensor()]))
    dl = DataLoader(ds,
                    batch_size=args.batch,
                    shuffle=True,
                    num_workers=min(args.workers, os.cpu_count() or 1),
                    pin_memory=True,
                    drop_last=True)

    # model + optimizer
    model = AutoEncoder(z_ch=args.zdim).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)

    # LR schedule: linear warm-up then cosine
    total_steps  = args.epochs * len(dl)
    warmup_steps = args.warmup_epochs * len(dl)
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / max(1, total_steps - warmup_steps)))
    sched = LambdaLR(opt, lr_lambda)

    # output setup
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    best_psnr = 0.0; log = []
    print(f"Training AE: zdim={args.zdim}, batch={args.batch}, lr={args.lr}")

    global_step = 0
    t0 = time.time()
    for epoch in range(1, args.epochs+1):
        model.train(); running_mse = 0.0

        for batch in dl:
            batch = batch.to(device, non_blocking=True)
            x_hat, _ = model(batch)

            mse_loss = nn.functional.mse_loss(x_hat, batch)
            l1_loss  = nn.functional.l1_loss(x_hat, batch)
            loss     = mse_loss + args.l1_weight * l1_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step(); global_step += 1

            running_mse += mse_loss.item() * batch.size(0)

        # end epoch
        epoch_mse  = running_mse / len(ds)
        epoch_psnr = psnr_from_mse(epoch_mse)
        log.append({"epoch":epoch, "mse":epoch_mse, "psnr":epoch_psnr})
        print(f"[{epoch:02d}/{args.epochs}] MSE={epoch_mse:.4e}  PSNR={epoch_psnr:.2f} dB")

        # save best
        if epoch_psnr > best_psnr:
            best_psnr = epoch_psnr
            torch.save(model.enc.state_dict(), out_dir/"ae_encoder.pt")
            torch.save(model.dec.state_dict(), out_dir/"ae_decoder.pt")

        # recon dumps
        if epoch in (1,5,10):
            model.eval()
            with torch.no_grad():
                sample = batch[:4]         # first 4
                recon  = model(sample)[0]
                grid   = torch.cat([sample, recon], dim=0)
                vutils.save_image(grid,
                    out_dir/f"recon_epoch{epoch}.png",
                    nrow=4, normalize=False)

        # early stop
        if epoch_psnr >= args.psnr_target:
            print(f"PSNR≥{args.psnr_target} dB reached; stopping.")
            break

    # write log
    with open(out_dir/"train_log.json","w") as f:
        json.dump(log, f, indent=2)

    print(f"\nDone in {(time.time()-t0):.1f}s — best PSNR={best_psnr:.2f} dB")
    print("Weights →", out_dir/"ae_encoder.pt", out_dir/"ae_decoder.pt")
    print("Recons  →", [out_dir/f"recon_epoch{e}.png" for e in (1,5,10)])
    print("Log     →", out_dir/"train_log.json")

# -----------------------------------------------------------------------------
# 5 · CLI
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",    required=True,
                   help="Folder with exhaustive PNGs (flat)")
    p.add_argument("--out_dir",      required=True,
                   help="Where to write models + logs")
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--warmup_epochs",type=int,   default=5,
                   help="Linear LR warm-up epochs")
    p.add_argument("--batch",        type=int,   default=256)
    p.add_argument("--zdim",         type=int,   default=64,
                   help="Latent-channel count (16×16×zdim)")
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--l1_weight",    type=float, default=0.1,
                   help="Weight for L1 term")
    p.add_argument("--psnr_target",  type=float, default=40.0)
    p.add_argument("--workers",      type=int,   default=8)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_autoencoder(args)
