"""Step 5: data loading + augmentation (augmentation applies to training only)."""
import tensorflow as tf
import config


def load_datasets():
    common = dict(
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode="categorical",
    )
    train = tf.keras.utils.image_dataset_from_directory(
        config.SPLIT_DIR / "train", shuffle=True, seed=config.SEED, **common)
    val = tf.keras.utils.image_dataset_from_directory(
        config.SPLIT_DIR / "val", shuffle=False, **common)
    test = tf.keras.utils.image_dataset_from_directory(
        config.SPLIT_DIR / "test", shuffle=False, **common)

    class_names = train.class_names
    autotune = tf.data.AUTOTUNE
    return (train.prefetch(autotune), val.prefetch(autotune),
            test.prefetch(autotune), class_names)


def build_augmentation():
    """Simulates varied smartphone photos: angle, zoom, lighting."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomContrast(0.2),
    ], name="augmentation")
