

"""
Train a GAN (MNIST) and save artifacts to ./outputs so Azure ML captures them.

Usage (local):
  python train_gan.py --epochs 50 --batch-size 512 --noise-dim 100 --save-interval 5

Notes:
- Figures and models are saved to ./outputs (collected by Azure ML jobs).
- If Graphviz/pydot are installed, model diagrams are saved too (best-effort).
"""

import os
import json
import math
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

# Keras 3 imports (backend-agnostic, typically uses TensorFlow on Azure ML curated TF envs)
from keras import Input
from keras.models import Sequential, load_model
from keras.layers import (
    Dense, BatchNormalization, LeakyReLU, Reshape,
    Conv2DTranspose, Conv2D, Flatten, Dropout
)
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.utils import plot_model, model_to_dot

# -----------------------------
# Utility: ensure outputs folder
# -----------------------------
def ensure_out_dir(path="outputs"):
    os.makedirs(path, exist_ok=True)
    return path

# -----------------------------
# Models
# -----------------------------
def build_generator(noise_dim: int) -> Sequential:
    """Generator: z -> 28x28x1"""
    g = Sequential(name="Generator")
    g.add(Input(shape=(noise_dim,)))
    g.add(Dense(7 * 7 * 128))
    g.add(BatchNormalization())
    g.add(LeakyReLU(negative_slope=0.2))
    g.add(Reshape((7, 7, 128)))

    g.add(Conv2DTranspose(128, (5, 5), strides=(2, 2), padding="same"))
    g.add(BatchNormalization())
    g.add(LeakyReLU(negative_slope=0.2))

    g.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same"))
    g.add(BatchNormalization())
    g.add(LeakyReLU(negative_slope=0.2))

    # Final conv to get single-channel image in [0,1]
    g.add(Conv2D(1, (7, 7), activation="sigmoid", padding="same"))

    # Compiling the generator separately is optional; included for clarity
    g.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
        metrics=["accuracy"],
    )
    return g


def build_discriminator(in_shape=(28, 28, 1)) -> Sequential:
    """Discriminator: 28x28x1 -> real/fake (sigmoid)"""
    d = Sequential(name="Discriminator")
    d.add(Input(shape=in_shape))
    d.add(Conv2D(64, (5, 5), strides=(2, 2), padding="same"))
    d.add(LeakyReLU(negative_slope=0.2))
    d.add(Dropout(0.3))

    d.add(Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
    d.add(LeakyReLU(negative_slope=0.2))
    d.add(Dropout(0.3))

    d.add(Flatten())
    d.add(Dense(1, activation="sigmoid"))

    d.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
        metrics=["accuracy"],
    )
    return d


def build_gan(g_model: Sequential, d_model: Sequential) -> Sequential:
    """Combined GAN: d frozen, g + d chained."""
    d_model.trainable = False  # must freeze BEFORE compile
    gan = Sequential(name="GAN")
    gan.add(g_model)
    gan.add(d_model)

    gan.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
    )
    return gan

# -----------------------------
# Data helpers
# -----------------------------
def make_real_data(dataset, n_samples):
    idx = np.random.randint(0, dataset.shape[0], n_samples)
    X = dataset[idx]
    y = np.ones((n_samples, 1))
    return X, y

def make_noises(noise_dim, n_samples):
    z = np.random.randn(noise_dim * n_samples)
    return z.reshape(n_samples, noise_dim)

def make_fake_data(g_model, noise_dim, n_samples):
    z = make_noises(noise_dim, n_samples)
    X = g_model.predict(z, verbose=0)
    y = np.zeros((n_samples, 1))
    return X, y

# -----------------------------
# Visualization helpers
# -----------------------------
def save_sample_grid(examples, filename, rows=5, cols=5):
    """Save a grid of generated samples to PNG."""
    plt.figure(figsize=(rows * 2, cols * 2))
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        plt.axis("off")
        plt.imshow(examples[i, :, :, 0], cmap="gray_r")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, transparent=True, bbox_inches="tight")
    plt.close()

def try_save_model_plots(model, basename, out_dir):
    """Best-effort save model diagrams if pydot/Graphviz are available."""
    try:
        plot_model(model, to_file=os.path.join(out_dir, f"{basename}.png"),
                   show_shapes=True, show_layer_names=True, dpi=200)
        # Optional: SVG via model_to_dot
        try:
            model_to_dot(model, show_shapes=True, show_layer_names=True).write_svg(
                os.path.join(out_dir, f"{basename}.svg")
            )
        except Exception:
            pass
    except Exception:
        # silently continue if diagram dependencies aren’t present
        pass

# -----------------------------
# Training
# -----------------------------
def train(args):
    out_dir = ensure_out_dir(args.output_dir)
    np.random.seed(args.seed)

    # Load MNIST
    (X_train, _), (_, _) = mnist.load_data()
    X_train = np.expand_dims(X_train.astype("float32") / 255.0, axis=-1)

    # Build/load models
    if args.load_generator and os.path.exists(args.generator_path):
        g_model = load_model(args.generator_path)
    else:
        g_model = build_generator(args.noise_dim)

    if args.load_discriminator and os.path.exists(args.discriminator_path):
        d_model = load_model(args.discriminator_path)
    else:
        d_model = build_discriminator()

    if args.load_gan and os.path.exists(args.gan_path):
        gan_model = load_model(args.gan_path)
    else:
        gan_model = build_gan(g_model, d_model)

    # Save model diagrams (optional)
    try_save_model_plots(g_model, "generator_model", out_dir)
    try_save_model_plots(d_model, "discriminator_model", out_dir)
    try_save_model_plots(gan_model, "gan_model", out_dir)

    # Training loop
    epochs = args.epochs
    batch_size = args.batch_size
    half_batch = batch_size // 2
    bat_per_epo = max(1, X_train.shape[0] // batch_size)

    losses = {"d": [], "g": []}
    start_time = datetime.now()

    for epoch in range(1, epochs + 1):
        for _ in range(bat_per_epo):
            # --- Train Discriminator on mixed real/fake ---
            X_real, y_real = make_real_data(X_train, half_batch)
            X_fake, y_fake = make_fake_data(g_model, args.noise_dim, half_batch)
            X_mix = np.vstack((X_real, X_fake))
            y_mix = np.vstack((y_real, y_fake))

            d_model.trainable = True
            d_loss, d_acc = d_model.train_on_batch(X_mix, y_mix)

            # --- Train Generator via GAN (Discriminator frozen) ---
            d_model.trainable = False
            z = make_noises(args.noise_dim, batch_size)
            y_gan = np.ones((batch_size, 1))
            g_loss = gan_model.train_on_batch(z, y_gan)

        losses["d"].append(float(d_loss))
        losses["g"].append(float(g_loss))

        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"Epoch {epoch:03d} | D loss={d_loss:.4f} | G loss={g_loss:.4f} | elapsed={int(elapsed)}s")

        # Save sample images at intervals
        if epoch % args.save_interval == 0:
            z = make_noises(args.noise_dim, args.sample_count)
            samples = g_model.predict(z, verbose=0)
            save_sample_grid(samples, os.path.join(out_dir, f"samples_epoch_{epoch:03d}.png"),
                             rows=int(math.sqrt(args.sample_count)),
                             cols=int(math.sqrt(args.sample_count)))

    # Save final artifacts
    g_model.save(os.path.join(out_dir, "generator.keras"))
    d_model.save(os.path.join(out_dir, "discriminator.keras"))
    gan_model.save(os.path.join(out_dir, "gan.keras"))

    # Save loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(losses["d"], label="Discriminator loss")
    plt.plot(losses["g"], label="GAN loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GAN training losses")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_losses.png"), dpi=150, bbox_inches="tight", transparent=True)
    plt.close()

    # Save metrics JSON
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"final_d_loss": float(losses["d"][-1]),
                   "final_g_loss": float(losses["g"][-1]),
                   "epochs": int(epochs),
                   "batch_size": int(batch_size)}, f, indent=2)

    print(f"Training complete. Artifacts saved under: {out_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--noise-dim", type=int, default=100)
    p.add_argument("--save-interval", type=int, default=5, help="Save sample grid every N epochs.")
    p.add_argument("--sample-count", type=int, default=25, help="Must be a square number (e.g., 25=5x5).")
    p.add_argument("--output-dir", type=str, default="outputs")

    # Optional load paths (if resuming)
    p.add_argument("--load-generator", action="store_true")
    p.add_argument("--load-discriminator", action="store_true")
    p.add_argument("--load-gan", action="store_true")
    p.add_argument("--generator-path", type=str, default="generator.keras")
    p.add_argument("--discriminator-path", type=str, default="discriminator.keras")
    p.add_argument("--gan-path", type=str, default="gan.keras")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
