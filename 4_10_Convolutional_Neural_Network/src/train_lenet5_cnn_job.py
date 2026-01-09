"""
Train a CNN on CIFAR-10 for Azure ML jobs (non-interactive).
- Uses tf.keras; no plt.show(); writes artifacts to ./outputs.
- Optional data augmentation via Keras preprocessing layers.
- Saves: training curves, sample predictions grid, metrics.json
"""

import os
import json
import argparse
import math
from datetime import datetime

# Non-interactive plotting backend for headless jobs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers

# -----------------------
# Azure-friendly utilities
# -----------------------
def ensure_out_dir(path="outputs"):
    os.makedirs(path, exist_ok=True)
    return path

def save_training_curves(history, out_dir):
    """Save loss/accuracy curves."""
    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(history.history.get("loss", []), label="train loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"),
                dpi=150, bbox_inches="tight", transparent=True)
    plt.close()

    # Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(history.history.get("accuracy", []), label="train acc")
    if "val_accuracy" in history.history:
        plt.plot(history.history["val_accuracy"], label="val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_curve.png"),
                dpi=150, bbox_inches="tight", transparent=True)
    plt.close()

def save_sample_predictions(images, labels, preds, class_names, filename, rows=5, cols=5):
    """Save a grid of sample predictions (no interactive display)."""
    plt.figure(figsize=(cols * 2.2, rows * 2.2))
    for i in range(rows * cols):
        if i >= images.shape[0]:
            break
        plt.subplot(rows, cols, i + 1)
        plt.xticks([]); plt.yticks([]); plt.grid(False)
        plt.imshow(images[i])
        pred_label = class_names[int(preds[i])]
        true_label = class_names[int(labels[i])]
        color = "green" if pred_label == true_label else "red"
        plt.xlabel(f"P:{pred_label}\nT:{true_label}", color=color)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight", transparent=True)
    plt.close()

# -----------------------
# Model definition (tf.keras)
# Pattern follows common Conv2D+MaxPool stacks for CIFAR-10. [1](https://www.tensorflow.org/tutorials/images/cnn)
# -----------------------
def build_cnn(input_shape=(32, 32, 3), num_classes=10, dropout_rate=0.5):
    model = models.Sequential(name="CIFAR10_CNN")
    # Block 1
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    # Block 2
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    # Block 3
    model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    # Head
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(num_classes, activation="softmax"))
    return model

def build_augmentation():
    """Optional data augmentation pipeline using Keras preprocessing layers."""
    # Horizontal flips + slight rotations/zoom to improve generalization. [1](https://www.tensorflow.org/tutorials/images/cnn)
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ], name="Augment")

# -----------------------
# Training routine
# -----------------------
def train(args):
    out_dir = ensure_out_dir(args.output_dir)
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # TF GPU memory growth can also be set via env var TF_FORCE_GPU_ALLOW_GROWTH=true
    # If you want to enforce here:
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

    # Load CIFAR-10 (built-in) and normalize to [0,1]. [1](https://www.tensorflow.org/tutorials/images/cnn)
    (X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
    X_train = X_train.astype("float32") / 255.0
    X_test  = X_test.astype("float32") / 255.0
    # Keep labels as integers; use sparse_categorical_crossentropy.
    y_train = y_train.reshape(-1)
    y_test  = y_test.reshape(-1)

    class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']  # [1](https://www.tensorflow.org/tutorials/images/cnn)

    # Build model
    model = build_cnn(input_shape=(32, 32, 3), num_classes=10, dropout_rate=args.dropout)
    opt = optimizers.Adam(learning_rate=args.lr)
    model.compile(optimizer=opt,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    # Optional augmentation
    augment = build_augmentation() if args.augment else None
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(10000, seed=args.seed)
    if augment:
        train_ds = train_ds.map(lambda x, y: (augment(x), y),
                                num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    val_split = args.val_split
    if val_split > 0:
        # Create a validation set from the end of training data
        val_size = int(len(X_train) * val_split)
        X_val, y_val = X_train[-val_size:], y_train[-val_size:]
        X_train2, y_train2 = X_train[:-val_size], y_train[:-val_size]
        base_ds = tf.data.Dataset.from_tensor_slices((X_train2, y_train2)).shuffle(10000, seed=args.seed)
        if augment:
            base_ds = base_ds.map(lambda x, y: (augment(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = base_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    else:
        val_ds = None

    # Train
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy" if val_ds else "accuracy",
                                             factor=0.5, patience=3, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy" if val_ds else "accuracy",
                                         patience=8, restore_best_weights=True, verbose=1)
    ]
    start = datetime.now()
    history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        verbose=2,
        callbacks=callbacks
    )
    elapsed = int((datetime.now() - start).total_seconds())
    print(f"Training elapsed: {elapsed}s")

    # Evaluate on test set
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(args.batch_size)
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Test: loss={test_loss:.4f} acc={test_acc:.4f}")

    # Save training curves
    save_training_curves(history, out_dir)

    # Save sample predictions grid
    # Pick first N samples
    N = min(args.sample_count, X_test.shape[0])
    preds = model.predict(X_test[:N], verbose=0).argmax(axis=1)
    save_sample_predictions(
        images=X_test[:N],
        labels=y_test[:N],
        preds=preds,
        class_names=class_names,
        filename=os.path.join(out_dir, "sample_predictions.png"),
        rows=int(math.sqrt(N)),
        cols=int(math.sqrt(N)),
    )

    # Save metrics JSON
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "augment": bool(args.augment),
            "lr": float(args.lr),
            "val_split": float(args.val_split),
            "elapsed_seconds": int(elapsed)
        }, f, indent=2)

    print(f"Done. Artifacts saved in: {out_dir}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--val-split", type=float, default=0.1, help="Fraction of train data for validation (0..1).")
    p.add_argument("--augment", action="store_true", help="Use Keras preprocessing data augmentation.")
    p.add_argument("--sample-count", type=int, default=25, help="grid size (must be a square number, e.g., 25=5x5).")
    p.add_argument("--output-dir", type=str, default="outputs")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
