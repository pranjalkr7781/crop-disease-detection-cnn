"""Step 6: MobileNetV2 transfer-learning model."""
import tensorflow as tf
import config
from src.data import build_augmentation


def build_model(num_classes):
    base = tf.keras.applications.MobileNetV2(
        input_shape=config.IMG_SIZE + (3,), include_top=False, weights="imagenet")
    base.trainable = False

    inputs = tf.keras.Input(shape=config.IMG_SIZE + (3,))
    x = build_augmentation()(inputs)                       # active only during training
    x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1, name="rescale")(x)  # [0,255] -> [-1,1]
    x = base(x, training=False)                            # keeps BatchNorm in inference mode
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base


def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def unfreeze_top_layers(base, n_layers):
    """Step 7b: unfreeze the last n layers (BatchNorm layers stay frozen)."""
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers[-n_layers:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


def get_base(model):
    """Return the nested MobileNetV2 sub-model of a trained model."""
    return next(l for l in model.layers if l.name.startswith("mobilenetv2"))


def build_inference_model(trained):
    """Same weights, but without augmentation layers (for export / serving)."""
    base = get_base(trained)
    inputs = tf.keras.Input(shape=config.IMG_SIZE + (3,))
    x = trained.get_layer("rescale")(inputs)
    x = base(x, training=False)
    x = trained.get_layer("gap")(x)
    outputs = trained.get_layer("predictions")(x)   # dropout is a no-op at inference
    return tf.keras.Model(inputs, outputs, name="inference_model")
