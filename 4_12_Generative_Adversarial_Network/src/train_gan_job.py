
"""
Train a GAN (MNIST) and save artifacts to ./outputs so Azure ML captures them.

Usage:
  python train_gan_job.py --epochs 50 --batch-size 512 --noise-dim 100 --save-interval 5 --sample-count 25 --output-dir outputs

Notes:
    - CPU / single-GPU / multi-GPU (MirroredStrategy)
    - MLflow tracking (Azure ML compatible)
    - Manual training loop (GAN-safe)
"""

import os
import json
import math
import argparse
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, datasets

# -----------------------
# MLflow
# -----------------------
import mlflow
import mlflow.tensorflow

# -----------------------
# Utilities
# -----------------------
def ensure_out_dir(path="outputs"):
    os.makedirs(path, exist_ok=True)
    return path


def save_sample_grid(images, path, rows, cols):
    plt.figure(figsize=(cols * 2.2, rows * 2.2))
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        plt.axis("off")
        plt.imshow(images[i, :, :, 0], cmap="gray")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# -----------------------
# Models
# -----------------------
def build_generator(noise_dim):
    model = models.Sequential(name="Generator")
    model.add(layers.Input(shape=(noise_dim,)))
    model.add(layers.Dense(7 * 7 * 128))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Reshape((7, 7, 128)))

    model.add(layers.Conv2DTranspose(128, 5, strides=2, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))

    model.add(layers.Conv2DTranspose(64, 5, strides=2, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))

    model.add(layers.Conv2D(1, 7, activation="sigmoid", padding="same"))
    return model


def build_discriminator():
    model = models.Sequential(name="Discriminator")
    model.add(layers.Input(shape=(28, 28, 1)))
    model.add(layers.Conv2D(64, 5, strides=2, padding="same"))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, 5, strides=2, padding="same"))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="sigmoid"))
    return model


# -----------------------
# Training
# -----------------------
def train(args):
    out_dir = ensure_out_dir(args.output_dir)

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # ---- TF / CUDA safety (Azure + Colab) ----
    tf.config.optimizer.set_experimental_options({
        "layout_optimizer": False
    })

    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)

    print(f"Detected {len(gpus)} GPU(s)")

    # ---- Distribution strategy ----
    strategy = tf.distribute.MirroredStrategy()
    print("Using strategy:", strategy.__class__.__name__)
    print("Replicas:", strategy.num_replicas_in_sync)

    # ---- MLflow (manual logging only) ----
    mlflow.tensorflow.autolog(
        log_models=False,
        log_input_examples=False,
        log_datasets=False,
    )

    with mlflow.start_run():
        mlflow.log_params({
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "effective_batch_size": args.batch_size * strategy.num_replicas_in_sync,
            "noise_dim": args.noise_dim,
            "save_interval": args.save_interval,
            "num_gpus": len(gpus),
            "seed": args.seed,
        })

        # ---- Data ----
        (X_train, _), _ = datasets.mnist.load_data()
        X_train = X_train.astype("float32") / 255.0
        X_train = np.expand_dims(X_train, axis=-1)

        batch_size = args.batch_size
        half_batch = batch_size // 2
        steps_per_epoch = X_train.shape[0] // batch_size

        # ---- Models (inside strategy) ----
        with strategy.scope():
            generator = build_generator(args.noise_dim)
            discriminator = build_discriminator()

            d_optimizer = optimizers.Adam(2e-4, beta_1=0.5)
            g_optimizer = optimizers.Adam(2e-4, beta_1=0.5)

            discriminator.compile(
                optimizer=d_optimizer,
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )

            discriminator.trainable = False
            gan_input = layers.Input(shape=(args.noise_dim,))
            gan_output = discriminator(generator(gan_input))
            gan = models.Model(gan_input, gan_output, name="GAN")
            gan.compile(
                optimizer=g_optimizer,
                loss="binary_crossentropy",
            )

        # ---- Training loop ----
        start = datetime.now()
        d_losses, g_losses = [], []

        for epoch in range(1, args.epochs + 1):
            for _ in range(steps_per_epoch):
                # ---- Train Discriminator ----
                idx = np.random.randint(0, X_train.shape[0], half_batch)
                real_imgs = X_train[idx]
                fake_noise = np.random.randn(half_batch, args.noise_dim)
                fake_imgs = generator.predict(fake_noise, verbose=0)

                X = np.concatenate([real_imgs, fake_imgs])
                y = np.concatenate([
                    np.ones((half_batch, 1)),
                    np.zeros((half_batch, 1))
                ])

                discriminator.trainable = True
                d_loss, _ = discriminator.train_on_batch(X, y)

                # ---- Train Generator ----
                noise = np.random.randn(batch_size, args.noise_dim)
                valid = np.ones((batch_size, 1))

                discriminator.trainable = False
                g_loss = gan.train_on_batch(noise, valid)

            d_losses.append(float(d_loss))
            g_losses.append(float(g_loss))

            mlflow.log_metrics({
                "d_loss": float(d_loss),
                "g_loss": float(g_loss),
            }, step=epoch)

            elapsed = int((datetime.now() - start).total_seconds())
            print(f"Epoch {epoch:03d} | D={d_loss:.4f} | G={g_loss:.4f} | {elapsed}s")

            # ---- Samples ----
            if epoch % args.save_interval == 0:
                noise = np.random.randn(args.sample_count, args.noise_dim)
                samples = generator.predict(noise, verbose=0)
                img_path = os.path.join(out_dir, f"samples_epoch_{epoch:03d}.png")
                save_sample_grid(
                    samples,
                    img_path,
                    rows=int(math.sqrt(args.sample_count)),
                    cols=int(math.sqrt(args.sample_count)),
                )
                mlflow.log_artifact(img_path)

        # ---- Artifacts ----
        generator.save(os.path.join(out_dir, "generator.keras"))
        discriminator.save(os.path.join(out_dir, "discriminator.keras"))
        gan.save(os.path.join(out_dir, "gan.keras"))

        # mlflow.tensorflow.log_model(generator, name="generator")
        # mlflow.tensorflow.log_model(discriminator, name="discriminator")
        # mlflow.tensorflow.log_model(gan, name="gan")

        metrics_path = os.path.join(out_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump({
                "final_d_loss": d_losses[-1],
                "final_g_loss": g_losses[-1],
                "epochs": args.epochs,
                "num_gpus": len(gpus),
            }, f, indent=2)
        mlflow.log_artifact(metrics_path)


# -----------------------
# Args
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--noise-dim", type=int, default=100)
    p.add_argument("--save-interval", type=int, default=5)
    p.add_argument("--sample-count", type=int, default=25)
    p.add_argument("--output-dir", type=str, default="outputs")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
