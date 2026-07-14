import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from training.loss import loss_fn
import configs

@nnx.jit
def train_step(model, optimizer, metrics, batch):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)

    metrics.update(
        loss=loss,
        logits=logits,
        labels=batch[1]
    )

    optimizer.update(grads)

def train(model,text_dl,batches_per_epoch):
        
    total_steps = batches_per_epoch * configs.NUM_EPOCHS

    warmup_steps = max(1, total_steps // 10)  # 10% warmup

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=configs.LEARNING_RATE,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=1e-5
    )

    #print(f"Total training steps: {total_steps:,}")
    #print(f"Warmup steps: {warmup_steps:,}")

    optimizer = nnx.Optimizer(
        model,
        optax.adamw(learning_rate=lr_schedule, weight_decay=configs.WEIGHT_DECAY)
    )

    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss")
    )

    metrics_history = {'train_loss': []}

    prep_target_batch = jax.vmap(
        lambda tokens: jnp.concatenate((tokens[1:], jnp.array([0]))))

    for epoch in range(configs.NUM_EPOCHS):
        step = 0
        for batch in text_dl:
            input_batch = jnp.array(jnp.array(batch).T).astype(jnp.int32)
            target_batch = prep_target_batch(
                jnp.array(jnp.array(batch).T)).astype(jnp.int32)
            print(".", end="")
            train_step(model, optimizer, metrics, (input_batch, target_batch))

            if (step + 1) % 2 == 0:
                for metric, value in metrics.compute().items():
                    metrics_history[f'train_{metric}'].append(value)
                metrics.reset()

                current_lr = lr_schedule(step)
                print(f"\nEpoch: {epoch + 1}, Step {step + 1}, Loss: {metrics_history['train_loss'][-1]:.4f}, "
                    f"LR: {current_lr:.2e}")
            step += 1
    return model, metrics_history