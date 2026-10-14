"""Step 7: two-stage training (head, then fine-tune).
Run: python -m src.train
Outputs: models/best_model.keras, models/class_names.json, outputs/training_curves.png
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

import config
from src.data import load_datasets
from src.model import build_model, compile_model, unfreeze_top_layers


def get_class_weights(train_ds, num_classes):
    labels = np.concatenate([np.argmax(y, axis=1) for _, y in train_ds])
    weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=labels)
    return dict(enumerate(weights))


def plot_history(histories, path):
    acc = sum([h.history["accuracy"] for h in histories], [])
    val_acc = sum([h.history["val_accuracy"] for h in histories], [])
    loss = sum([h.history["loss"] for h in histories], [])
    val_loss = sum([h.history["val_loss"] for h in histories], [])
    split = len(histories[0].history["accuracy"])

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(acc, label="train"); ax[0].plot(val_acc, label="val")
    ax[0].axvline(split - 1, ls="--", c="gray", label="fine-tune starts")
    ax[0].set_title("Accuracy"); ax[0].legend()
    ax[1].plot(loss, label="train"); ax[1].plot(val_loss, label="val")
    ax[1].axvline(split - 1, ls="--", c="gray")
    ax[1].set_title("Loss"); ax[1].legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main():
    config.MODEL_DIR.mkdir(exist_ok=True)
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    tf.keras.utils.set_random_seed(config.SEED)

    train_ds, val_ds, test_ds, class_names = load_datasets()
    num_classes = len(class_names)
    print(f"{num_classes} classes: {class_names}")

    with open(config.MODEL_DIR / "class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)

    class_weights = get_class_weights(train_ds, num_classes)
    model, base = build_model(num_classes)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(config.MODEL_DIR / "best_model.keras"),
            monitor="val_accuracy", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=2, verbose=1),
    ]

    print("\n=== Stage 1: training head ===")
    compile_model(model, config.LR_HEAD)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=config.EPOCHS_HEAD,
                   class_weight=class_weights, callbacks=callbacks)

    print("\n=== Stage 2: fine-tuning ===")
    unfreeze_top_layers(base, config.UNFREEZE_LAST)
    compile_model(model, config.LR_FINE)   # must recompile after changing trainable
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=config.EPOCHS_FINE,
                   class_weight=class_weights, callbacks=callbacks)

    plot_history([h1, h2], config.OUTPUT_DIR / "training_curves.png")
    print(f"\nBest model saved to {config.MODEL_DIR / 'best_model.keras'}")
    print("Training complete. Next: python -m src.evaluate")


if __name__ == "__main__":
    main()
