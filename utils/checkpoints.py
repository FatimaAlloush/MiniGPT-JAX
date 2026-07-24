from pathlib import Path
from huggingface_hub import snapshot_download


HF_REPO = "FatimaAlloush/MiniGPT-JAX"


def download_checkpoints():
    checkpoint_dir = Path("checkpoints")

    pretrained = checkpoint_dir / "pretrained_checkpoint.orbax"
    trained = checkpoint_dir / "trained_small_checkpoint.orbax"

    if pretrained.exists() and trained.exists():
        print("Checkpoints already available.")
        return checkpoint_dir

    print("Downloading checkpoints from Hugging Face...")

    snapshot_download(
        repo_id=HF_REPO,
        local_dir=checkpoint_dir,
        allow_patterns=[
            "pretrained_checkpoint.orbax/**",
            "trained_small_checkpoint.orbax/**"
        ],
    )

    print("Checkpoint download completed.")

    return checkpoint_dir