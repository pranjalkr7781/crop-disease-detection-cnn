"""Step 9: Grad-CAM heatmap showing where the model looks.
Run: python -m src.gradcam path/to/leaf.jpg
Output: outputs/gradcam_<name>.png
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

import config
from src.inference import load_model
from src.model import get_base


def make_heatmap(model, img_batch, class_index=None):
    base = get_base(model)
    names = [l.name for l in model.layers]
    head = [model.get_layer(n) for n in names[names.index(base.name) + 1:]]  # gap, dropout, predictions

    x = tf.convert_to_tensor(img_batch / 127.5 - 1.0)   # same scaling as the model's Rescaling layer
    with tf.GradientTape() as tape:
        conv_out = base(x, training=False)
        tape.watch(conv_out)
        y = conv_out
        for layer in head:
            y = layer(y, training=False)
        if class_index is None:
            class_index = int(tf.argmax(y[0]))
        score = y[:, class_index]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)[0]
    cam = tf.nn.relu(cam)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy(), class_index


def overlay(image: Image.Image, cam: np.ndarray, alpha=0.4):
    image = image.convert("RGB").resize(config.IMG_SIZE)
    cam_img = Image.fromarray(np.uint8(255 * cam)).resize(config.IMG_SIZE, Image.BILINEAR)
    heat = plt.get_cmap("jet")(np.asarray(cam_img) / 255.0)[..., :3]
    blended = (1 - alpha) * np.asarray(image) / 255.0 + alpha * heat
    return np.clip(blended, 0, 1)


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m src.gradcam path/to/leaf.jpg")
    path = Path(sys.argv[1])
    model, class_names = load_model()
    image = Image.open(path).convert("RGB").resize(config.IMG_SIZE)
    batch = np.expand_dims(np.asarray(image, dtype=np.float32), 0)
    cam, idx = make_heatmap(model, batch)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(image); ax[0].set_title("Input"); ax[0].axis("off")
    ax[1].imshow(overlay(image, cam)); ax[1].set_title(f"Grad-CAM: {class_names[idx]}", fontsize=9)
    ax[1].axis("off")
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    out = config.OUTPUT_DIR / f"gradcam_{path.stem}.png"
    plt.tight_layout(); plt.savefig(out, dpi=130)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
