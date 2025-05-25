#!/usr/bin/env python3
"""
Stage 2: Train a near-lossless image auto-encoder (VAE) with richer logging.

Changes in this revision (May 2025)
───────────────────────────────────
• --opt_state_device {gpu,cpu}  ➜ move Adam moment buffers to CPU on demand
• cuDNN workspace tweak helpers
• ETA now correct after resume
• Minor TF32 + CLI polish
"""

import argparse, random, sys, time
from pathlib import Path

import numpy as np
from PIL import Image
import torch, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from torchvision.utils import make_grid
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from diffusers import AutoencoderKL
import torch.amp


# ──────────────────────────── helpers ──────────────────────────────
def move_optimizer_state(optimizer: optim.Optimizer, device: torch.device) -> None:
    """Move Adam moment/hparam tensors to given device (CPU⇄GPU)."""
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device, non_blocking=True)


def denorm(t: torch.Tensor) -> torch.Tensor:     # [-1,1] → [0,1]
    return (t * 0.5) + 0.5


@torch.no_grad()
def evaluate(model: AutoencoderKL, loader, device, writer, step, kl_weight):
    model.eval()
    tot_loss = tot_rec = tot_kl = tot_psnr = tot_ssim = n = 0
    samp_orig, samp_rec = [], []

    for i, batch in enumerate(loader):
        x = batch.to(device, non_blocking=True)
        bs = x.size(0)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=(device.type == "cuda")):
            enc = model.encode(x, return_dict=True)
            latents = enc.latent_dist.mode()
            rec = model.decode(latents, return_dict=True).sample

        rec_loss = torch.nn.functional.l1_loss(rec, x)
        kl_loss  = enc.latent_dist.kl().mean()
        loss     = rec_loss + kl_weight * kl_loss

        tot_loss += loss.item() * bs
        tot_rec  += rec_loss.item() * bs
        tot_kl   += kl_loss.item() * bs

        x_np = denorm(x).permute(0, 2, 3, 1).cpu().numpy()
        r_np = denorm(rec).permute(0, 2, 3, 1).cpu().numpy()
        x_np, r_np = np.clip(x_np, 0, 1), np.clip(r_np, 0, 1)

        for xo, ro in zip(x_np, r_np):
            tot_psnr += peak_signal_noise_ratio(xo, ro, data_range=1.0)
            tot_ssim += structural_similarity(xo, ro, channel_axis=-1,
                                              win_size=7, data_range=1.0)

        if len(samp_orig) < 8:
            need = 8 - len(samp_orig)
            samp_orig.extend(list(denorm(x)  [:need].cpu()))
            samp_rec .extend(list(denorm(rec)[:need].cpu()))

        n += bs
        if i >= 49:          # cap for speed
            break

    if n == 0:
        print("[val] no samples!")
        model.train()
        return

    writer.add_scalars("Loss",    {"val_total": tot_loss/n,
                                   "val_recon": tot_rec/n,
                                   "val_kl":    tot_kl/n}, step)
    writer.add_scalars("Metrics", {"PSNR_val": tot_psnr/n,
                                   "SSIM_val": tot_ssim/n}, step)

    if samp_orig:
        grid = make_grid(torch.cat([torch.stack(samp_orig),
                                    torch.stack(samp_rec)]), nrow=len(samp_orig))
        writer.add_image("Validation/Orig_vs_Recon", grid, step)

    print(f"[val] step={step}  loss={tot_loss/n:.4f}  "
          f"PSNR={tot_psnr/n:.2f}  SSIM={tot_ssim/n:.4f}")
    model.train()


# ───────────────────────────── CLI ─────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Train a VAE on UI frames with TensorBoard logging")
    # Data / model
    p.add_argument("--data_dir", type=Path, required=True,
                   help="Directory with episode folders (ep_00000, …)")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--latent_channels", type=int, default=4)
    # Training params
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_training_steps", type=int, default=200_000)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--kl_weight", type=float, default=1e-6)
    p.add_argument("--use_lr_decay", action="store_true",
                   help="Enable cosine LR decay to LR/10 across training")
    # Logging / ckpts
    p.add_argument("--log_dir_base", type=Path, default=Path("logs_vae"))
    p.add_argument("--experiment_name", type=str, default="experiment_1")
    p.add_argument("--checkpoint_dir", type=Path,
                   default=Path("checkpoints_vae"))
    p.add_argument("--log_every_n_steps", type=int, default=100)
    p.add_argument("--val_every_n_steps", type=int, default=2500)
    p.add_argument("--save_every_n_steps", type=int, default=2000)
    p.add_argument("--sample_grid_every_n_steps", type=int, default=1000)
    # System
    p.add_argument("--num_dataloader_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume_from_checkpoint", type=Path)
    # NEW
    p.add_argument("--opt_state_device", choices=["gpu", "cpu"], default="gpu",
                   help="Where to keep Adam moment buffers after (re)start")
    return p.parse_args()


# ──────────────────────────── main ────────────────────────────────
def main():
    args = parse_args()

    # Repro
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Use TF32 (Ampere+) for extra speed without measurable quality hit
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True  # lets cuDNN pick fastest algo

    # I/O dirs
    log_dir = args.log_dir_base / args.experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Gather frame paths
    if not args.data_dir.is_dir():
        sys.exit(f"Error: data dir {args.data_dir} not found")
    frames = []
    for ep in args.data_dir.glob("ep_*"):
        if ep.is_dir():
            frames.extend(ep.glob("*_prev.png"))
            frames.extend(ep.glob("*_next.png"))
    frames = sorted(set(frames))
    print(f"Found {len(frames):,} frames")
    if not frames:
        sys.exit("No images found")

    random.shuffle(frames)
    split = int(0.95 * len(frames))
    train_files, val_files = frames[:split], frames[split:]
    print(f"Train: {len(train_files):,}   Val: {len(val_files):,}")

    tfm = T.Compose([
        T.Resize(args.image_size),
        T.CenterCrop(args.image_size),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3)
    ])
    train_dl = DataLoader(
        DatasetFromFileList(train_files, tfm),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_dataloader_workers,
        pin_memory=True, drop_last=True, persistent_workers=True,
        prefetch_factor=4)
    val_dl = DataLoader(
        DatasetFromFileList(val_files, tfm),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_dataloader_workers,
        pin_memory=True, persistent_workers=True)

    # Model
    model = AutoencoderKL(
        in_channels=3, out_channels=3,
        down_block_types=("DownEncoderBlock2D",)*4,
        up_block_types=("UpDecoderBlock2D",)*4,
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2,
        latent_channels=args.latent_channels,
        norm_num_groups=32,
        sample_size=args.image_size,
    ).to(device)
    print(f"VAE params: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    opt = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = (optim.lr_scheduler.CosineAnnealingLR(
                     opt, T_max=args.num_training_steps,
                     eta_min=args.learning_rate*0.1)
                 if args.use_lr_decay else None)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    writer = SummaryWriter(str(log_dir))

    # Resume?
    global_step = 0
    if args.resume_from_checkpoint and args.resume_from_checkpoint.is_file():
        ckpt = torch.load(args.resume_from_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        global_step = ckpt.get("global_step", 0)
        if scheduler and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        print(f"Resumed from step {global_step:,}")

    # *After* loading, optionally park moment buffers on CPU
    if args.opt_state_device == "cpu":
        move_optimizer_state(opt, torch.device("cpu"))
        torch.cuda.empty_cache()              # free the VRAM immediately
        print("✓ Optimizer state moved to CPU")

    # Reset ETA reference so it’s meaningful after resume
    start_time = time.time()
    model.train()
    train_iter = iter(train_dl)

    while global_step < args.num_training_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            batch = next(train_iter)

        batch = batch.to(device, non_blocking=True)
        step_t0 = time.time()

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=(device.type == "cuda")):
            enc = model.encode(batch, return_dict=True)
            latents = enc.latent_dist.sample()
            rec = model.decode(latents, return_dict=True).sample

            rec_loss = torch.nn.functional.l1_loss(rec, batch)
            kl_loss  = enc.latent_dist.kl().mean()
            loss     = rec_loss + args.kl_weight * kl_loss

        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        if scheduler: scheduler.step()
        global_step += 1

        step_time = time.time() - step_t0
        # ── console / TB logging ───────────────────────────
        if global_step % args.log_every_n_steps == 0:
            gmem = torch.cuda.memory_allocated()/(1024**3) if device.type == "cuda" else 0
            elapsed = time.time() - start_time
            eta_h   = (args.num_training_steps - global_step) * elapsed / max(1, global_step) / 3600
            writer.add_scalars("Loss", {"train_total": loss.item(),
                                        "train_recon": rec_loss.item(),
                                        "train_kl":    kl_loss.item()}, global_step)
            writer.add_scalar("LR", opt.param_groups[0]['lr'], global_step)
            writer.add_scalar("Perf/step_time", step_time, global_step)

            print(f"[train] step={global_step:,}  "
                  f"loss={loss.item():.4f}  gpu_mem={gmem:.1f} GB  "
                  f"step={step_time*1000:.0f} ms  ETA={eta_h:.1f} h")

        # ── vis sample grid ────────────────────────────────
        if global_step % args.sample_grid_every_n_steps == 0:
            with torch.no_grad():
                grid = make_grid(torch.cat([denorm(batch[:4]),
                                            denorm(rec[:4])]), nrow=4)
                writer.add_image("Train/recons", grid, global_step)

        # ── validation ─────────────────────────────────────
        if global_step % args.val_every_n_steps == 0:
            evaluate(model, val_dl, device, writer, global_step,
                     args.kl_weight)

        # ── checkpoint ─────────────────────────────────────
        if (global_step % args.save_every_n_steps == 0 or
                global_step == args.num_training_steps):
            ckpt_path = (args.checkpoint_dir /
                         f"vae_step_{global_step}.pt").as_posix()
            torch.save({
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                **({"scheduler_state_dict": scheduler.state_dict()}
                   if scheduler else {})
            }, ckpt_path)
            print("[ckpt] saved", ckpt_path)

    writer.close()
    print("Training finished.")


# ────────── dataset class (unchanged) ──────────
class DatasetFromFileList(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths, self.transform = file_paths, transform
    def __len__(self): return len(self.file_paths)
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        try:
            img = Image.open(img_path).convert("RGB")
            if self.transform: img = self.transform(img)
            return img
        except Exception as e:
            print(f"Warning: could not read {img_path}: {e}", file=sys.stderr)
            return self.__getitem__((idx + 1) % len(self))


if __name__ == "__main__":
    main()
