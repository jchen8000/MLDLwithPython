# -*- coding: utf-8 -*-
"""
LeNet-5-style model on MNIST (28×28 grayscale), mirrored from the provided Jupyter cells,
packaged as a non-interactive script for Azure ML command jobs.

Key choices (matching notebook cells):
- Input: MNIST, shape (28, 28, 1).
- Normalization: Keras Rescaling(1./255).
- Architecture: Conv(6, 5x5, same, ReLU) → AvgPool(2x2) → Conv(16, 5x5, same, ReLU) → AvgPool(2x2)
               → Flatten → Dense(120, ReLU) → Dense(84, ReLU) → Dense(10, softmax).
- Optimizer: Adam(lr=1e-3) by default.
- Loss/Metric: SparseCategoricalCrossentropy, Accuracy.
- Training: batch_size=128, epochs=100, validation_split=0.1, EarlyStopping(patience=3, restore_best_weights=True).
- Artifacts: training curves SVG (same filename used in notebook, but under outputs/),
             metrics.json, classification_report.txt, confusion_matrix.json.

This script avoids interactive plotting (no plt.show) and writes artifacts into the Azure ML
"outputs" folder so they are captured by the job.
"""

import os
import json
import argparse
from datetime import datetime

# Headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import metrics

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def ensure_out_dir(path: str = "outputs") -> str:
    os.makedirs(path, exist_ok=True)
    return path


def build_lenet5_model() -> keras.Model:
    """Construct the LeNet-5-style model exactly as in the notebook cells."""
    num_classes = 10
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Rescaling(1./255),
        # C1
        layers.Conv2D(6, kernel_size=(5, 5), padding="same", activation="relu"),
        layers.AveragePooling2D(pool_size=(2, 2)),
        # C3
        layers.Conv2D(16, kernel_size=(5, 5), padding="same", activation="relu"),
        layers.AveragePooling2D(pool_size=(2, 2)),
        # Head
        layers.Flatten(),
        layers.Dense(120, activation="relu"),
        layers.Dense(84,  activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ], name="LeNet5_MNIST")
    return model


def save_training_plots(history: keras.callbacks.History, out_dir: str) -> None:
    """Mirror the notebook plots and save to SVG under outputs/.
    Note: Filename mirrors the notebook ('lenet5_on_cifar-10.svg') to keep parity.
    """
    fig = plt.figure(1, figsize=(16, 6))
    # Accuracy subplot
    plt.subplot(1, 2, 1)
    plt.title('Accuracy', fontsize=16)
    plt.plot(history.history.get('accuracy', []), label="Trainset", c='blue')
    plt.plot(history.history.get('val_accuracy', []), label="Testset", c='blue', ls='--')
    plt.ylabel('Accuracy', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.legend(loc='best', fontsize=12)
    # Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(history.history.get('loss', []), label="Trainset", c='k')
    plt.plot(history.history.get('val_loss', []), label="Testset", c='k', ls='--')
    plt.title('Loss', fontsize=16)
    plt.ylabel('Loss', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.legend(loc='best', fontsize=12)
    # Save (same filename as notebook, placed under outputs/)
    plt.savefig(os.path.join(out_dir, "lenet5_on_cifar-10.svg"),
                format="svg", transparent=True, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------
# Training routine
# ---------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    out_dir = ensure_out_dir(args.output_dir)

    # Mirror notebook seed behavior
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Load MNIST
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    print("X_train", X_train.shape)
    print("y_train", y_train.shape)
    print("X_test", X_test.shape)
    print("y_test", y_test.shape)

    # Expand channel dimension: (28,28) -> (28,28,1)
    X_train = X_train[..., np.newaxis]
    X_test  = X_test[..., np.newaxis]

    # Build and compile model
    model = build_lenet5_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    # Summary
    model.summary()

    # Callbacks (mirror notebook: EarlyStopping only)
    callbacks = [
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    ]

    # Train (mirror notebook settings)
    start = datetime.now()
    history = model.fit(
        X_train, y_train.ravel(),
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=args.val_split,
        verbose=2,
        callbacks=callbacks,
    )
    elapsed = int((datetime.now() - start).total_seconds())
    print(f"Training elapsed: {elapsed}s")

    # Save plots
    save_training_plots(history, out_dir)

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test.ravel(), verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    # Predictions & sklearn metrics
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    a_score = metrics.accuracy_score(y_test, y_pred)
    c_matrix = metrics.confusion_matrix(y_test, y_pred)
    c_report = metrics.classification_report(y_test, y_pred)

    print("Accuracy Score:\n", a_score)
    print("Confusion matrix:\n", c_matrix)
    print("Classification Report:\n", c_report)

    # Persist artifacts
    # metrics.json
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "elapsed_seconds": int(elapsed),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "val_split": float(args.val_split),
        }, f, indent=2)

    # classification_report.txt
    with open(os.path.join(out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(c_report)

    # confusion_matrix.json
    with open(os.path.join(out_dir, "confusion_matrix.json"), "w", encoding="utf-8") as f:
        json.dump(c_matrix.tolist(), f)

    # predictions.npy (optional)
    np.save(os.path.join(out_dir, "y_pred.npy"), y_pred)

    print(f"Done. Artifacts saved in: {out_dir}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--output-dir", type=str, default="outputs")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
