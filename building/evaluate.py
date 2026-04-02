"""
evaluate.py — multi-class evaluation utilities.

Extracted and extended from utils.py for N-class (sigmoid) models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

if TYPE_CHECKING:
    import keras
    import tensorflow as tf


def plot_training_history(history: Any, title: str | None = None) -> None:
    import matplotlib.pyplot as plt

    h = history.history if hasattr(history, "history") else history
    loss_key = "loss"
    val_loss_key = "val_loss"
    acc_key = next(
        (k for k in ("accuracy", "sparse_categorical_accuracy", "categorical_accuracy") if k in h),
        None,
    )

    epochs = np.arange(1, len(h[loss_key]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title or "Training History")

    axes[0].plot(epochs, h[loss_key], label="Train")
    axes[0].plot(epochs, h[val_loss_key], label="Val")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    if acc_key:
        axes[1].plot(epochs, h[acc_key], label="Train")
        axes[1].plot(epochs, h[f"val_{acc_key}"], label="Val")
        axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
        axes[1].legend()
        axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()


def evaluate_multiclass(
    model: "keras.Model",
    dataset: "tf.data.Dataset",
    label_names: np.ndarray,
    threshold: float = 0.5,
    display: bool = True,
) -> dict:
    """Evaluate a sigmoid multi-class/multi-label model on a tf.data.Dataset.

    Computes per-class and macro metrics, confusion matrix, and per-class ROC-AUC.
    """
    import tensorflow as tf

    y_true_list: list[np.ndarray] = []
    y_score_list: list[np.ndarray] = []

    for x_batch, y_batch in dataset:
        preds = model(x_batch, training=False)
        y_score_list.append(preds.numpy())
        y_true_list.append(y_batch.numpy())

    y_true = np.concatenate(y_true_list, axis=0)   # (N, C)
    y_score = np.concatenate(y_score_list, axis=0)  # (N, C)

    # For integer-label datasets (sparse), convert to one-hot
    if y_true.ndim == 1 or y_true.shape[1] == 1:
        n_classes = len(label_names)
        y_true_int = y_true.astype(int).reshape(-1)
        y_true_oh = np.eye(n_classes)[y_true_int]
    else:
        y_true_oh = y_true

    y_pred = (y_score >= threshold).astype(int)
    y_pred_single = y_score.argmax(axis=1)
    y_true_single = y_true_oh.argmax(axis=1)

    # Per-class ROC-AUC (one-vs-rest)
    try:
        auc_per_class = roc_auc_score(y_true_oh, y_score, average=None)
        macro_auc = float(np.mean(auc_per_class))
    except Exception:
        auc_per_class = np.zeros(len(label_names))
        macro_auc = 0.0

    report = classification_report(
        y_true_single,
        y_pred_single,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true_single, y_pred_single)

    if display:
        print(classification_report(y_true_single, y_pred_single, target_names=label_names, zero_division=0))
        print("Confusion Matrix:\n", cm)
        print(f"Macro ROC-AUC: {macro_auc:.4f}")
        print("Per-class AUC:")
        for name, auc in zip(label_names, auc_per_class):
            print(f"  {name:40s} {auc:.4f}")

        _plot_confusion_matrix(cm, label_names)

    return {
        "report": report,
        "confusion_matrix": cm,
        "macro_auc": macro_auc,
        "auc_per_class": dict(zip(label_names.tolist(), auc_per_class.tolist())),
        "y_true": y_true_oh,
        "y_score": y_score,
    }


def _plot_confusion_matrix(cm: np.ndarray, label_names: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, len(label_names)), max(5, len(label_names) - 1)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(label_names)),
        yticks=np.arange(len(label_names)),
        xticklabels=label_names,
        yticklabels=label_names,
        xlabel="Predicted",
        ylabel="True",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.show()


def evaluate_tflite_multiclass(
    tflite_path: str,
    dataset: "tf.data.Dataset",
    label_names: np.ndarray,
    display: bool = True,
) -> dict:
    """Same as evaluate_multiclass but runs a TFLite interpreter."""
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    y_true_list: list[np.ndarray] = []
    y_score_list: list[np.ndarray] = []

    for x_batch, y_batch in dataset.unbatch().batch(1):
        x_np = x_batch.numpy().astype(np.float32)
        interpreter.resize_tensor_input(inp["index"], x_np.shape)
        interpreter.allocate_tensors()
        interpreter.set_tensor(inp["index"], x_np)
        interpreter.invoke()
        preds = interpreter.get_tensor(out["index"])
        y_score_list.append(preds)
        y_true_list.append(y_batch.numpy())

    y_true = np.concatenate(y_true_list, axis=0)
    y_score = np.concatenate(y_score_list, axis=0)

    # Delegate to eager version by faking a model
    class _FakeTFLite:
        def __call__(self, x, training=False):
            return y_score[: len(x)]

    # Re-use the numpy results directly
    if y_true.ndim == 1 or y_true.shape[-1] == 1:
        n_classes = len(label_names)
        y_true_int = y_true.astype(int).reshape(-1)
        y_true_oh = np.eye(n_classes)[y_true_int]
    else:
        y_true_oh = y_true

    y_pred_single = y_score.argmax(axis=1)
    y_true_single = y_true_oh.argmax(axis=1)

    try:
        auc_per_class = roc_auc_score(y_true_oh, y_score, average=None)
        macro_auc = float(np.mean(auc_per_class))
    except Exception:
        auc_per_class = np.zeros(len(label_names))
        macro_auc = 0.0

    report = classification_report(y_true_single, y_pred_single, target_names=label_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true_single, y_pred_single)

    if display:
        print(classification_report(y_true_single, y_pred_single, target_names=label_names, zero_division=0))
        print("Confusion Matrix:\n", cm)
        print(f"Macro ROC-AUC: {macro_auc:.4f}")
        _plot_confusion_matrix(cm, label_names)

    return {
        "report": report,
        "confusion_matrix": cm,
        "macro_auc": macro_auc,
        "auc_per_class": dict(zip(label_names.tolist(), auc_per_class.tolist())),
        "y_true": y_true_oh,
        "y_score": y_score,
    }
