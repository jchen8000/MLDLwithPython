"""
Train a CNN on CIFAR-10 for Azure ML jobs (non-interactive).
- Optional data augmentation via Keras preprocessing layers.
- Saves: training curves, sample predictions grid, metrics.json
- CPU / single-GPU / multi-GPU (MirroredStrategy)
- MLflow tracking (Azure ML compatible)
"""

import os
import json
import argparse
import math
from datetime import datetime
import mlflow
import mlflow.tensorflow
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers


# -----------------------
# Utilities
# -----------------------
def ensure_out_dir(path="outputs"):
    os.makedirs(path, exist_ok=True)
    return path


def save_training_curves(history, out_dir):
    loss_path = os.path.join(out_dir, "loss_curve.png")
    acc_path = os.path.join(out_dir, "accuracy_curve.png")

    plt.figure()
    plt.plot(history.history["loss"], label="train")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_path, dpi=150)
    plt.close()

    plt.figure()
    plt.plot(history.history["accuracy"], label="train")
    if "val_accuracy" in history.history:
        plt.plot(history.history["val_accuracy"], label="val")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(acc_path, dpi=150)
    plt.close()

    return loss_path, acc_path


def save_sample_predictions(images, labels, preds, class_names, path, rows, cols):
    plt.figure(figsize=(cols * 2.2, rows * 2.2))
    for i in range(rows * cols):
        if i >= len(images):
            break
        plt.subplot(rows, cols, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(images[i])
        color = "green" if preds[i] == labels[i] else "red"
        plt.xlabel(
            f"P:{class_names[preds[i]]}\nT:{class_names[labels[i]]}",
            color=color,
        )
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# -----------------------
# Model
# -----------------------
def build_cnn(input_shape=(32, 32, 3), num_classes=10, dropout=0.5):
    model = models.Sequential(name="CIFAR10_CNN")

    model.add(layers.Input(shape=input_shape))

    for filters in [32, 64, 128]:
        model.add(layers.Conv2D(filters, 3, padding="same", activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(filters, 3, padding="same", activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D())
        model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(num_classes, activation="softmax"))

    return model


def build_augmentation():
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ],
        name="augmentation",
    )


# -----------------------
# Training
# -----------------------
def train(args):
    out_dir = ensure_out_dir(args.output_dir)

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    tf.config.optimizer.set_experimental_options({
        "layout_optimizer": False
    })

    # ---- GPU detection ----
    gpus = tf.config.list_physical_devices("GPU")
    num_gpus = len(gpus)
    print(f"Detected {num_gpus} GPU(s)")

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # ---- Distribution strategy (safe for 0/1/N GPUs) ----
    strategy = tf.distribute.MirroredStrategy()
    print("Using strategy:", strategy.__class__.__name__)
    print("Replicas:", strategy.num_replicas_in_sync)

    # ---- MLflow (NO param conflicts) ----
    mlflow.tensorflow.autolog(
        log_models=False,
        log_input_examples=False,
        log_datasets=False,
    )

    with mlflow.start_run():
        # Manual params (single source of truth)
        mlflow.log_params({
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "effective_batch_size": args.batch_size * strategy.num_replicas_in_sync,
            "lr": args.lr,
            "dropout": args.dropout,
            "val_split": args.val_split,
            "augment": args.augment,
            "num_gpus": num_gpus,
            "seed": args.seed,
        })

        # ---- Data ----
        (X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
        X_train = X_train.astype("float32") / 255.0
        X_test = X_test.astype("float32") / 255.0
        y_train = y_train.squeeze()
        y_test = y_test.squeeze()

        class_names = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]

        if args.val_split > 0:
            val_size = int(len(X_train) * args.val_split)
            X_val, y_val = X_train[-val_size:], y_train[-val_size:]
            X_train, y_train = X_train[:-val_size], y_train[:-val_size]
        else:
            X_val = y_val = None

        augment = build_augmentation() if args.augment else None

        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)) \
            .shuffle(10000)
        if augment:
            train_ds = train_ds.map(
                lambda x, y: (augment(x), y),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        train_ds = train_ds.batch(args.batch_size) \
                           .prefetch(tf.data.AUTOTUNE)

        val_ds = None
        if X_val is not None:
            val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)) \
                .batch(args.batch_size) \
                .prefetch(tf.data.AUTOTUNE)

        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)) \
            .batch(args.batch_size)

        # ---- Model (inside strategy) ----
        with strategy.scope():
            model = build_cnn(dropout=args.dropout)
            model.compile(
                optimizer=optimizers.Adam(learning_rate=args.lr),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_accuracy" if val_ds else "accuracy",
                patience=3,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy" if val_ds else "accuracy",
                patience=8,
                restore_best_weights=True,
            ),
        ]

        start = datetime.now()
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=2,
        )

        elapsed = int((datetime.now() - start).total_seconds())
        mlflow.log_metric("elapsed_seconds", elapsed)

        test_loss, test_acc = model.evaluate(test_ds, verbose=0)
        mlflow.log_metrics({
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
        })

        # ---- Artifacts ----
        loss_img, acc_img = save_training_curves(history, out_dir)
        mlflow.log_artifact(loss_img)
        mlflow.log_artifact(acc_img)

        N = min(args.sample_count, len(X_test))
        preds = model.predict(X_test[:N], verbose=0).argmax(axis=1)
        sample_path = os.path.join(out_dir, "sample_predictions.png")
        save_sample_predictions(
            X_test[:N],
            y_test[:N],
            preds,
            class_names,
            sample_path,
            rows=int(math.sqrt(N)),
            cols=int(math.sqrt(N)),
        )
        mlflow.log_artifact(sample_path)

        metrics_path = os.path.join(out_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump({
                "test_loss": float(test_loss),
                "test_accuracy": float(test_acc),
                "num_gpus": num_gpus,
                "elapsed_seconds": elapsed,
            }, f, indent=2)
        mlflow.log_artifact(metrics_path)

        # ---- Final model ----
        # mlflow.tensorflow.log_model(
        #     model,
        #     name="model",
        # )


# -----------------------
# Args
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--sample-count", type=int, default=25)
    p.add_argument("--output-dir", type=str, default="outputs")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
