"""Central settings for the project. Edit these first."""
from pathlib import Path

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw"          # put class folders here (e.g. Tomato_Early_blight/)
SPLIT_DIR = ROOT / "data" / "split"      # created by src/prepare_data.py
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

EPOCHS_HEAD = 10        # stage 1: train only the new classifier head
EPOCHS_FINE = 8         # stage 2: fine-tune last layers
LR_HEAD = 1e-3
LR_FINE = 1e-5
UNFREEZE_LAST = 30      # number of base-model layers to unfreeze
