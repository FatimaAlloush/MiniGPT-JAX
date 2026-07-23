from pathlib import Path


# =========================
# Project Paths
# =========================

# Project root (folder containing configs.py)
PROJECT_ROOT = Path(__file__).resolve().parent


# =========================
# Dataset
# =========================

DATA_PATH = PROJECT_ROOT / "Data" / "TinyStories-1000.txt"
MAX_STORIES = 100


# =========================
# Model Configuration
# =========================

MAXLEN = 128
EMBED_DIM = 192
NUM_HEADS = 6
FEED_FORWARD_DIM = 512
NUM_TRANSFORMER_BLOCKS = 6


# =========================
# Training Configuration
# =========================

BATCH_SIZE = 32
NUM_EPOCHS = 3
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01


# =========================
# Inference / Evaluation
# =========================

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

PRETRAINED_CHECKPOINT = (
    CHECKPOINT_DIR / "pretrained_checkpoint.orbax"
)

TRAINED_CHECKPOINT = (
    CHECKPOINT_DIR / "trained_small_checkpoint.orbax"
)

PROMPT = "Once upon a time"

TEMPERATURE = 0.2
MAX_NEW_TOKENS = 50


# =========================
# Reproducibility
# =========================

SEED = 42