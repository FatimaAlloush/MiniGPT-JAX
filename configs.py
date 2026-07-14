from pathlib import Path

# Project root (folder containing configs.py)
PROJECT_ROOT = Path(__file__).resolve().parent

# Dataset
DATA_PATH = PROJECT_ROOT / "Data" / "TinyStories-1000.txt"
MAX_STORIES = 100

# Model
MAXLEN = 128
EMBED_DIM = 192
NUM_HEADS = 6
FEED_FORWARD_DIM = 512
NUM_TRANSFORMER_BLOCKS = 6

# Training
BATCH_SIZE = 32
NUM_EPOCHS = 3
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01

# Inference / Evaluation
PRETRAINED_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / "pretrained_checkpoint.orbax"
).resolve()

TRAINED_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / "trained_small_checkpoint.orbax"
).resolve()

PROMPT = "Once upon a time"

TEMPERATURE = 0.2
MAX_NEW_TOKENS = 50

# Reproducibility
SEED = 42