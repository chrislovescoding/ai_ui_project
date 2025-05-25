# run inside the same venv, one batch, one step
import torch, torchvision
from src.train_autoencoder import Encoder, Decoder
x = torchvision.io.read_image(
        "C:/Users/Chris/Desktop/ai_ui_project/exhaustive_data/images/000000.png"
    ).float() / 255.     # shape (3,H,W), range 0–1
x = x.unsqueeze(0).cuda()
ae = torch.nn.Sequential(Encoder(z_ch=8), Decoder(z_ch=8)).cuda()
y = ae(x)
print("MSE one-shot:", torch.nn.functional.mse_loss(y,x).item())
