"""Step 4a: count images per class and show a sample grid.
Run: python -m src.explore_data
"""
import random
import matplotlib.pyplot as plt
from PIL import Image
import config

EXTS = {".jpg", ".jpeg", ".png"}


def list_images(folder):
    return [p for p in folder.iterdir() if p.suffix.lower() in EXTS]


def main():
    classes = sorted(d for d in config.RAW_DIR.iterdir() if d.is_dir())
    if not classes:
        raise SystemExit(f"No class folders found in {config.RAW_DIR}")

    counts = {c.name: len(list_images(c)) for c in classes}
    print(f"{'Class':45s} Images")
    for name, n in counts.items():
        print(f"{name:45s} {n}")
    print(f"\nTotal: {sum(counts.values())} images, {len(counts)} classes")
    print(f"Imbalance ratio (max/min): {max(counts.values()) / max(1, min(counts.values())):.1f}")

    config.OUTPUT_DIR.mkdir(exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(counts)), counts.values())
    plt.xticks(range(len(counts)), counts.keys(), rotation=90, fontsize=7)
    plt.title("Images per class")
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "class_distribution.png", dpi=120)
    plt.close()

    show = classes[:12]
    cols = 4
    rows = (len(show) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = axes.flatten()
    for ax in axes:
        ax.axis("off")
    for ax, c in zip(axes, show):
        img = Image.open(random.choice(list_images(c)))
        ax.imshow(img)
        ax.set_title(c.name, fontsize=8)
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "samples.png", dpi=120)
    print(f"Saved plots to {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
