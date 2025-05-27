# Real-time AI UI Demo

A proof-of-concept project demonstrating a real-time AI that "imagines" and renders a graphical user interface frame-by-frame based solely on the previous frame, mouse position, and click state. Inspired by OasisAI's Minecraft world model, this system uses a simple autoencoder+predictor pipeline to generate UI frames without any hard-coded logic—behavior is entirely learned from training data. In the future, this approach could be extended to imagine multi-layered websites and complex interfaces from only a handful of design snapshots.

## Demo

Watch the AI-generated UI in action! The system learns to respond to mouse movements and clicks, generating realistic UI behavior frame-by-frame. In this demo, you can see the button responding to interactions, and even some fun "breaking" of the interface when pushed to its limits:

![AI UI Demo](assets/demo.gif)

*The AI learns UI behavior patterns from training data and generates frames in real-time based on cursor position and click state.*

---

## Features

- **Synthetic UI generator**– Renders a window (256×256 px), button, cursor sprite, and dynamic text.– Generates aligned triplets: previous frame, cursor position mask, click state mask, next frame.
- **Autoencoder-based compression**– Trains a convolutional autoencoder (AE) to encode/decode UI frames (16×16 latent).
- **Latent-space frame predictor**– A TinyUNet learns to roll out future latent representations given cursor & click inputs.
- **Real-time interactive demo**– PyGame application for live inference: move/click to see AI-generated UI react.
- **Data preparation & evaluation**
  – Scripts to generate synthetic episodes, prepare latent datasets, and visualize reconstructions/predictions.

---

## Requirements

- **Python** 3.8+
- **CUDA** toolkit (if using GPU acceleration)
- All other Python dependencies are pinned in `requirements.txt`

---

## Installation

1. **Clone** the repository:

   ```bash
   git clone https://github.com/chrislovescoding/ai_ui_project.git
   cd ai_ui_project
   ```
2. **Install dependencies** :

```bash
   pip install --upgrade pip
   pip install -r requirements.txt
```

2. **Verify assets** : ensure the following files exist:

* `assets/DejaVuSans.ttf`
* `assets/cursor.png`

2. **Generate synthetic data** :

```bash
   python -m src.data_generation.run_generation
```

   This uses `assets/design_spec.yaml` to render UI episodes under the configured output directory.

## Running the Pretrained Model

To run the real-time demo using the pretrained models provided in the repository, follow these steps:

1. Ensure you have the following files:
   - Encoder: `models/ae_v4/ae_encoder.pt`
   - Decoder: `models/ae_v4/ae_decoder.pt`
   - Predictor: `models/predictor_v1/predictor.pt`

2. Launch the real-time interactive demo:

   ```bash
   python scripts/realtime_demo.py \
     --encoder_pt models/ae_v4/ae_encoder.pt \
     --decoder_pt models/ae_v4/ae_decoder.pt \
     --predictor_pt models/predictor_v1/predictor.pt \
     --zdim 64
   ```

   Move and click in the window to see the AI-generated UI react in real time.

---

## Training

### 1. Train Autoencoder

```bash
python -m src.train_autoencoder \
  --data_root path/to/exhaustive_data/images \
  --out_dir models/ae_v4 \
  --epochs 50 \
  --warmup_epochs 5 \
  --zdim 64 \
  --batch 256 \
  --lr 3e-4 \
  --l1_weight 0.1 \
  --psnr_target 40.0
```

### 2. Prepare Latent Dataset

```bash
python scripts/prepare_latent_dataset.py \
  --episodes-root path/to/generated/episodes \
  --encoder-pt models/ae_v4/ae_encoder.pt \
  --out latent_triplets
```

### 3. Train Predictor

```bash
python -m src.train_predictor \
  --data latent_triplets \
  --out models/predictor_v1 \
  --epochs 30 \
  --batch 256 \
  --zdim 64 \
  --lr 1e-3
```

---

## Evaluation & Visualization

* **Reconstruction tests** :

```bash
  python scripts/test_autoencoder.py \
    --data_root path/to/exhaustive_data/images \
    --encoder_pt models/ae_v4/ae_encoder.pt \
    --decoder_pt models/ae_v4/ae_decoder.pt \
    --zdim 64 \
    --num 10
```

* **Prediction tests** :

```bash
  python scripts/test_predictor.py \
    --npz_folder latent_triplets \
    --encoder_pt models/ae_v4/ae_encoder.pt \
    --decoder_pt models/ae_v4/ae_decoder.pt \
    --predictor_pt models/predictor_v1/predictor.pt \
    --zdim 64 \
    --num 5
```

* **Create GIFs** from a set of episodes:
  ```bash
  python scripts/create_gifs.py --input_dir path/to/episodes --output_dir data_gifs
  ```

---

## Future Directions

* **Generalized UI imagination** : scale beyond a single button to multi-layered websites, dynamic menus, and forms.
* **Self-supervised pretraining** : learn from real UI screenshots to boost realism.
* **Guided generation** : incorporate high-level user intents to steer frame-by-frame outputs.
