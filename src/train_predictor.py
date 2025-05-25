# src/train_predictor.py
import argparse, glob, json, random, time
from pathlib import Path

import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# ──────────────────────────────────────────────────────────────────────────────
class LatentTripletDataset(Dataset):
    def __init__(self, folder, split="train", val_frac=0.01, seed=0):
        files = sorted(glob.glob(f"{folder}/*.npz"))
        rng = random.Random(seed)
        rng.shuffle(files)
        split_idx = int(len(files) * (1 - val_frac))
        self.files = files[:split_idx] if split == "train" else files[split_idx:]

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        # shapes: z_prev / z_next = [1,Z,16,16]; cursor/click/coord similar
        z_prev = torch.from_numpy(d["z_prev"]).float()[0]
        z_next = torch.from_numpy(d["z_next"]).float()[0]
        cursor = torch.from_numpy(d["cursor"]).float()[0]
        click  = torch.from_numpy(d["click" ]).float()[0]
        coord  = torch.from_numpy(d["coord" ]).float()[0]
        inp = torch.cat([z_prev, cursor, click, coord], dim=0)   # [C_in,16,16]
        return inp, z_next

# ──────────────────────────────────────────────────────────────────────────────
class TinyUNet(nn.Module):
    """2-layer UNet-style conv stack in latent space (16×16)."""
    def __init__(self, c_in, zdim):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(c_in,  zdim*2, 3,1,1), nn.ReLU(True),
            nn.Conv2d(zdim*2, zdim*2, 3,1,1), nn.ReLU(True)
        )
        self.up   = nn.Conv2d(zdim*2, zdim, 3,1,1)

    def forward(self, x): return self.up(self.down(x))

# ──────────────────────────────────────────────────────────────────────────────
def train(args):
    torch.manual_seed(0); random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds_train = LatentTripletDataset(args.data, split="train")
    ds_val   = LatentTripletDataset(args.data, split="val")
    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=512, shuffle=False,
                          num_workers=args.workers, pin_memory=True)

    c_in = args.zdim + 1 + 1 + 2                     # latent + cursor + click + XY
    net  = TinyUNet(c_in, args.zdim).to(device)
    opt  = optim.Adam(net.parameters(), lr=args.lr)
    mse  = nn.MSELoss()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    best_val = float("inf"); history = []

    for ep in range(1, args.epochs+1):
        # ---- train ----
        net.train(); run = 0.0
        for x,y in dl_train:
            x,y = x.to(device), y.to(device)
            loss = mse(net(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            run += loss.item()*x.size(0)
        train_mse = run/len(ds_train)

        # ---- validate ----
        net.eval(); run = 0.0
        with torch.no_grad():
            for x,y in dl_val:
                x,y = x.to(device), y.to(device)
                run += mse(net(x), y).item()*x.size(0)
        val_mse = run/len(ds_val)

        history.append({"epoch":ep,"train_mse":train_mse,"val_mse":val_mse})
        print(f"[{ep:02d}/{args.epochs}]  train={train_mse:.4e}  val={val_mse:.4e}")

        if val_mse < best_val:
            best_val = val_mse
            torch.save(net.state_dict(), out/"predictor.pt")

    with open(out/"log.json","w") as f:
        json.dump(history,f,indent=2)
    print("✅  Finished. Best val-MSE:", best_val)

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--data",    required=True)
    a.add_argument("--out",     required=True)
    a.add_argument("--epochs",  type=int, default=30)
    a.add_argument("--batch",   type=int, default=256)
    a.add_argument("--zdim",    type=int, default=64)
    a.add_argument("--lr",      type=float, default=1e-3)
    a.add_argument("--workers", type=int, default=8)
    train(a.parse_args())
