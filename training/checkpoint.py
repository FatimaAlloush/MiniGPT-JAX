from pathlib import Path

import flax.nnx as nnx
import orbax.checkpoint as ocp


def save_checkpoint(model, checkpoint_name="small_checkpoint.orbax"):

    # Absolute path to the project's checkpoints folder
    checkpoint_dir = (Path.cwd() / "checkpoints").resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = (checkpoint_dir / checkpoint_name).resolve()

    print(f"Saving checkpoint to:\n{checkpoint_path}")
    print(f"Absolute path? {checkpoint_path.is_absolute()}")

    checkpointer = ocp.PyTreeCheckpointer()

    checkpointer.save(
        checkpoint_path,
        nnx.state(model),
        force=True
    )

    print(f"Model saved as {checkpoint_path}")