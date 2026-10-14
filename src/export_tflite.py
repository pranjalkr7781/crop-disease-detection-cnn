"""Step 10: convert to TensorFlow Lite for on-device (offline) use.
Run: python -m src.export_tflite
Outputs: models/model.tflite, models/labels.txt
"""
import json
import numpy as np
import tensorflow as tf

import config
from src.data import load_datasets
from src.model import build_inference_model


def main():
    model = tf.keras.models.load_model(config.MODEL_DIR / "best_model.keras")
    infer = build_inference_model(model)

    converter = tf.lite.TFLiteConverter.from_keras_model(infer)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]   # dynamic-range quantization
    tflite_model = converter.convert()
    out = config.MODEL_DIR / "model.tflite"
    out.write_bytes(tflite_model)
    print(f"Saved {out} ({len(tflite_model) / 1e6:.1f} MB)")

    with open(config.MODEL_DIR / "class_names.json") as f:
        names = json.load(f)
    (config.MODEL_DIR / "labels.txt").write_text("\n".join(names))

    # Sanity check: compare TFLite accuracy to the Keras model on part of the test set
    interp = tf.lite.Interpreter(model_content=tflite_model)
    interp.allocate_tensors()
    inp, outp = interp.get_input_details()[0], interp.get_output_details()[0]

    _, _, test_ds, _ = load_datasets()
    correct = total = 0
    for images, labels in test_ds.take(10):
        for img, lab in zip(images.numpy(), labels.numpy()):
            interp.set_tensor(inp["index"], img[None].astype(np.float32))
            interp.invoke()
            pred = np.argmax(interp.get_tensor(outp["index"])[0])
            correct += int(pred == np.argmax(lab)); total += 1
    print(f"TFLite accuracy on {total} test images: {correct / total:.1%}")


if __name__ == "__main__":
    main()
