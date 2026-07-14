
"""
Training entry point for MiniGPT-JAX.

This script runs the complete training pipeline:
1. Initialize the MiniGPT transformer model
2. Load and preprocess TinyStories data
3. Train the model using JAX/Flax NNX
4. Save training metrics visualization
5. Save the trained model checkpoint
"""

import matplotlib.pyplot as plt
import flax.nnx as nnx

import configs
from models.mini_gpt import MiniGPT
from data.load_stories import load_stories_from_file
from data.dataset import create_dataloader
from utils.tokenizer_utils import tokenizer, VOCAB_SIZE

from training.train import train
from training.checkpoint import save_checkpoint


def main():

    # Initialize model architecture
    model = MiniGPT(
        maxlen=configs.MAXLEN,
        vocab_size=VOCAB_SIZE,
        embed_dim=configs.EMBED_DIM,
        num_heads=configs.NUM_HEADS,
        feed_forward_dim=configs.FEED_FORWARD_DIM,
        num_transformer_blocks=configs.NUM_TRANSFORMER_BLOCKS,
        rngs=nnx.Rngs(configs.SEED)
    )


    # Load training data
    stories = load_stories_from_file(
        configs.DATA_PATH,
        max_stories=configs.MAX_STORIES
    )


    # Create data pipeline
    text_dl, batches_per_epoch = create_dataloader(
        stories=stories,
        tokenizer=tokenizer,
        maxlen=configs.MAXLEN,
        batch_size=configs.BATCH_SIZE,
        shuffle=False,
        num_epochs=configs.NUM_EPOCHS,
        seed=configs.SEED,
        worker_count=0
    )


    print("\nDataLoader created successfully")
    print(f"Will produce {batches_per_epoch} batches per epoch")


    # Train model
    trained_model, metrics_history = train(
        model=model,
        text_dl=text_dl,
        batches_per_epoch=batches_per_epoch,
    )


    # Save training curve
    plt.plot(metrics_history["train_loss"])
    plt.title("MiniGPT Training Loss")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.show()
    
    # Save trained parameters
    save_checkpoint(
        trained_model,
        "trained_small_checkpoint.orbax"
    )


if __name__ == "__main__":
    main()