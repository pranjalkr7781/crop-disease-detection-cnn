"""Step 4b: split raw data into train / val / test folders (70/15/15).
Run: python -m src.prepare_data
The test set is never touched during training.
"""
import random
import shutil
import config

EXTS = {".jpg", ".jpeg", ".png"}


def main():
    random.seed(config.SEED)
    if config.SPLIT_DIR.exists():
        print(f"{config.SPLIT_DIR} already exists. Delete it to re-split.")
        return

    for class_dir in sorted(d for d in config.RAW_DIR.iterdir() if d.is_dir()):
        files = [p for p in class_dir.iterdir() if p.suffix.lower() in EXTS]
        random.shuffle(files)
        n = len(files)
        n_train = int(n * config.TRAIN_FRAC)
        n_val = int(n * config.VAL_FRAC)
        parts = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }
        for split, items in parts.items():
            out = config.SPLIT_DIR / split / class_dir.name
            out.mkdir(parents=True, exist_ok=True)
            for f in items:
                shutil.copy2(f, out / f.name)
        print(f"{class_dir.name:45s} train={len(parts['train'])} "
              f"val={len(parts['val'])} test={len(parts['test'])}")
    print("Done.")


if __name__ == "__main__":
    main()
