from pathlib import Path
from huggingface_hub import snapshot_download

HF_REPO = "FatimaAlloush/MiniGPT-JAX"

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Absolute checkpoints directory
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


def download_checkpoints():
    pretrained = CHECKPOINT_DIR / "pretrained_checkpoint.orbax"
    trained = CHECKPOINT_DIR / "trained_small_checkpoint.orbax"

    if pretrained.exists() and trained.exists():
        print("Checkpoints already available.")
        return CHECKPOINT_DIR.resolve()

    print("Downloading checkpoints from Hugging Face...")

    snapshot_download(
        repo_id=HF_REPO,
        local_dir=CHECKPOINT_DIR,
        allow_patterns=[
            "pretrained_checkpoint.orbax/**",
            "trained_small_checkpoint.orbax/**",
        ],
    )

    print("Checkpoint download completed.")

    return CHECKPOINT_DIR.resolve()