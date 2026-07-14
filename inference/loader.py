import jax
import flax.nnx as nnx
import orbax.checkpoint as ocp

from jax.sharding import SingleDeviceSharding

import configs

from models.mini_gpt import MiniGPT
from utils.tokenizer_utils import VOCAB_SIZE


def load_model(checkpoint_path):

    # Create a model with the same architecture
    # The parameters will be replaced by the checkpoint
    model = MiniGPT(
        maxlen=configs.MAXLEN,
        vocab_size=VOCAB_SIZE,
        embed_dim=configs.EMBED_DIM,
        num_heads=configs.NUM_HEADS,
        feed_forward_dim=configs.FEED_FORWARD_DIM,
        num_transformer_blocks=configs.NUM_TRANSFORMER_BLOCKS,
        rngs=nnx.Rngs(configs.SEED)
    )


    # Force checkpoint arrays to CPU
    cpu_device = jax.devices("cpu")[0]
    cpu_sharding = SingleDeviceSharding(cpu_device)


    # Create restore instructions for every parameter
    restore_args = jax.tree_util.tree_map(
        lambda _: ocp.ArrayRestoreArgs(
            sharding=cpu_sharding
        ),
        nnx.state(model)
    )


    # Restore checkpoint
    checkpointer = ocp.PyTreeCheckpointer()

    restored_state = checkpointer.restore(
        checkpoint_path,
        item=nnx.state(model),
        restore_args=restore_args
    )


    # Replace initialized parameters with checkpoint parameters
    nnx.update(model, restored_state)


    return model