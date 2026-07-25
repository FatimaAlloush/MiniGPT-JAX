import jax
import flax.nnx as nnx
from pathlib import Path
import orbax
from orbax import checkpoint
from jax.sharding import SingleDeviceSharding 
from inference.generate import generate_story 
from models.mini_gpt import MiniGPT
from utils.tokenizer_utils import VOCAB_SIZE
import gradio as gr
from utils.checkpoints import download_checkpoints

#create a model of same architecture, with random initialized params
model = MiniGPT(
    maxlen=128,
    vocab_size= VOCAB_SIZE,
    embed_dim=192,
    num_heads=6,
    feed_forward_dim = 512,
    num_transformer_blocks=6,
    rngs=nnx.Rngs(0)
)

cpu_device = jax.devices('cpu')[0] #gets your cpu, to force the checkpoint onto cpu
cpu_sharding = SingleDeviceSharding(cpu_device) #load it all on the cpu device

restore_args = jax.tree_util.tree_map(
    lambda _: checkpoint.ArrayRestoreArgs(sharding=cpu_sharding),
    nnx.state(model)
) # the model state is a tree, called pytree, want to apply the fct to every leaf by tree_map, to say for every parameter, restore it onto CPU.

nnx.state(model)

checkpoint_dir = download_checkpoints()
checkpoint_path = (checkpoint_dir / "pretrained_checkpoint.orbax").resolve()

checkpointer = orbax.checkpoint.PyTreeCheckpointer()

print("Starting checkpoint restore...")
#the actual parameters restore
restored_state = checkpointer.restore(
    checkpoint_path,
    item=nnx.state(model),
    restore_args=restore_args)

print("Checkpoint restored successfully!")

#update the random initialized parameters to the pretrained loaded ones
nnx.update(model,restored_state)

print("Model ready!")

#run inference fct
def create_story(story_prompt, temperature, max_new_tokens):
    return generate_story(model, story_prompt, temperature, max_new_tokens)

#example
#create_story("Once upon a time a big bear ", 0.2, 30)

#use gradio, to create a web interface

demo = gr.Interface(
    fn=create_story,
    inputs=[
        gr.Textbox(label="Story Prompt"),         
        gr.Slider(
            minimum=0, maximum=1.0, value=0.8, step=0.01, label="Temperature"
        ),
        gr.Slider(minimum=0, maximum=200, value=10, step=1, label="Max Tokens"
        )
    ],
    outputs=["text"]
)

demo.launch(
    server_name="0.0.0.0",
    server_port=7860
)