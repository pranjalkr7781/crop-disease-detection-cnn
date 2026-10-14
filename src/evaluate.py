"""Step 8: evaluate on the untouched test set.
Run: python -m src.evaluate
Outputs: outputs/classification_report.txt, outputs/confusion_matrix.png
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

import config
from src.data import load_datasets


def main():
    _, _, test_ds, class_names = load_datasets()
    model = tf.keras.models.load_model(config.MODEL_DIR / "best_model.keras")

    y_true = np.concatenate([np.argmax(y, axis=1) for _, y in test_ds])
    y_pred = np.argmax(model.predict(test_ds, verbose=1), axis=1)

    report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    print(report)
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    (config.OUTPUT_DIR / "classification_report.txt").write_text(report)

    cm = confusion_matrix(y_true, y_pred)
    size = max(6, len(class_names) * 0.45)
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix (test set)")
    if len(class_names) <= 15:
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "confusion_matrix.png", dpi=130)
    print(f"Saved report and confusion matrix to {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
