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
- Azure ML & Colab compatible
- CPU / single-GPU / multi-GPU (MirroredStrategy)
- MLflow tracking (metrics, artifacts, final model)

"""

import os
import json
import argparse
from datetime import datetime
import mlflow
import mlflow.tensorflow
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
    """LeNet-5 style CNN"""
    model = keras.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Rescaling(1.0 / 255),

            layers.Conv2D(6, (5, 5), padding="same", activation="relu"),
            layers.AveragePooling2D(pool_size=(2, 2)),

            layers.Conv2D(16, (5, 5), padding="same", activation="relu"),
            layers.AveragePooling2D(pool_size=(2, 2)),

            layers.Flatten(),
            layers.Dense(120, activation="relu"),
            layers.Dense(84, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ],
        name="LeNet5_MNIST",
    )
    return model


def save_training_plots(history, out_dir):
    fig = plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    plt.title("Accuracy")
    plt.plot(history.history["accuracy"], label="train")
    if "val_accuracy" in history.history:
        plt.plot(history.history["val_accuracy"], label="val", linestyle="--")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.title("Loss")
    plt.plot(history.history["loss"], label="train")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val", linestyle="--")
    plt.legend()
    plt.grid(True)

    path = os.path.join(out_dir, "lenet5_training_curves.svg")
    plt.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)

    return path


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------
def train(args: argparse.Namespace) -> None:
    out_dir = ensure_out_dir(args.output_dir)

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # ---- TF + CUDA 12 stability fix ----
    tf.config.optimizer.set_experimental_options({
        "layout_optimizer": False
    })

    # ---- GPU detection ----
    gpus = tf.config.list_physical_devices("GPU")
    num_gpus = len(gpus)
    print(f"Detected {num_gpus} GPU(s)")

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # ---- Distribution strategy (0/1/N GPUs safe) ----
    strategy = tf.distribute.MirroredStrategy()
    print("Using strategy:", strategy.__class__.__name__)
    print("Replicas:", strategy.num_replicas_in_sync)

    # ---- MLflow (disable param autolog to avoid conflicts) ----
    mlflow.tensorflow.autolog(
        log_models=False,
        log_input_examples=False,
        log_datasets=False,
    )

    with mlflow.start_run():
        # ---- Log params manually ----
        mlflow.log_params({
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "effective_batch_size": args.batch_size * strategy.num_replicas_in_sync,
            "lr": args.lr,
            "val_split": args.val_split,
            "num_gpus": num_gpus,
            "seed": args.seed,
        })

        # ---- Load MNIST ----
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

        X_train = X_train[..., np.newaxis]
        X_test = X_test[..., np.newaxis]

        # ---- Build model inside strategy ----
        with strategy.scope():
            model = build_lenet5_model()
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=args.lr),
                loss=keras.losses.SparseCategoricalCrossentropy(),
                metrics=["accuracy"],
            )

        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=3,
                restore_best_weights=True
            )
        ]

        start = datetime.now()
        history = model.fit(
            X_train,
            y_train,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_split=args.val_split,
            callbacks=callbacks,
            verbose=2,
        )
        elapsed = int((datetime.now() - start).total_seconds())
        mlflow.log_metric("elapsed_seconds", elapsed)

        # ---- Evaluation ----
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        mlflow.log_metrics({
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
        })

        # ---- Predictions ----
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

        acc_score = metrics.accuracy_score(y_test, y_pred)
        conf_matrix = metrics.confusion_matrix(y_test, y_pred)
        class_report = metrics.classification_report(y_test, y_pred)

        # ---- Artifacts ----
        curve_path = save_training_plots(history, out_dir)
        mlflow.log_artifact(curve_path)

        metrics_path = os.path.join(out_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump({
                "test_loss": float(test_loss),
                "test_accuracy": float(test_acc),
                "accuracy_score": float(acc_score),
                "elapsed_seconds": elapsed,
                "num_gpus": num_gpus,
            }, f, indent=2)
        mlflow.log_artifact(metrics_path)

        report_path = os.path.join(out_dir, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(class_report)
        mlflow.log_artifact(report_path)

        cm_path = os.path.join(out_dir, "confusion_matrix.json")
        with open(cm_path, "w") as f:
            json.dump(conf_matrix.tolist(), f)
        mlflow.log_artifact(cm_path)

        # ---- Final model ----
        # mlflow.tensorflow.log_model(
        #     model,
        #     name="model",
        # )

        print("Training complete. Artifacts logged to MLflow.")


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
    train(parse_args())
