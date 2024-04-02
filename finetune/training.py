import os
import random
import itertools
import torch
import constants
import wandb
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
import numpy as np
import finetune as ft


class LisaCallback():
    def __init__(self, n_layers, interval_steps, model):
        self.n_layers = n_layers
        self.interval_steps = interval_steps
        self.model = model

        self.layers_attribute = "model.model.layers"
        self.total_layers = len(eval("self." + self.layers_attribute))

        self.active_layers_indices = []

    def freeze_all_layers(self):
        layers = eval("self." + self.layers_attribute)
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False

    def on_step_begin(self, global_step):
        if global_step % self.interval_steps == 0:
            self.switch_active_layers()

    def switch_active_layers(self):
        self.freeze_all_layers()

        layers = eval("self." + self.layers_attribute)
        self.active_layers_indices = np.random.choice(
            range(self.total_layers), self.n_layers, replace=False)
        print(
            f"Activating layers at indices: {self.active_layers_indices} for the next steps.")

        for idx in self.active_layers_indices:
            for param in layers[idx].parameters():
                param.requires_grad = True


def get_comparison_results(comparison_model, comparison_tokenizer):
    # Load the dataset
    loader = ft.dataset.CortexSubsetLoader(
        latest=False,
        random_seed=random.randint(0, 100000000),
        max_samples=100,
        steps=5,
        page_size=5
    )
    batches = loader.tokenize(comparison_tokenizer)

    # Calculate losses
    device = "cuda" if torch.cuda.is_available() else "cpu"
    losses = ft.validation.compute_losses(
        model=comparison_model,
        batches=batches,
        device=device
    )

    # Return average and standard deviation of losses
    return torch.mean(torch.tensor(losses)), torch.std(torch.tensor(losses))


def generate_hyperparam_combinations(hyperparams):
    keys, values = zip(*hyperparams.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return combinations


def tokenize_dataset(tokenizer, dataset):
    batches = []
    for sample in dataset:
        conversation = [
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["response"]}
        ]
        prompt_ids = tokenizer.apply_chat_template(
            [conversation[0]], truncation=True, max_length=constants.sequence_length,
            add_generation_prompt=True
        )
        ids = tokenizer.apply_chat_template(
            conversation, truncation=True, max_length=constants.sequence_length
        )
        batches.append((torch.stack([torch.tensor(ids)]), len(prompt_ids)))
    return batches


def train(
    model,
    batches,
    eval_batches,
    epochs,
    accumulation_steps,
    eval_steps,
    learning_rate,
    base_lr_adjustment_steps,
    min_lr,
    wandb_project
):
    """Trains a LoRA model on the provided data with the provided hyperparams."""
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    run = wandb.init(
        project=wandb_project,
        config={"learning_rate": learning_rate, "epochs": epochs}
    )

    # Set up constants
    warmup_start_lr = 5e-15
    warmup_end_lr = 5e-7
    warmup_lr_increase_steps = 150
    end_lr = 5e-15
    lr_change_steps = 1000
    grad_clip = 5.0

    # Warmup optimizer
    warmup_optimizer = torch.optim.AdamW(
        model.parameters(), lr=warmup_start_lr, weight_decay=0.01)  # basically zero learning rate
    print("Beginning warmup period")

    # Set up LISA
    lisa_callback = LisaCallback(8, 5, model)

    current_lr = warmup_start_lr
    global_step = 0
    n_acc_steps = 0
    total_loss = 0.0
    losses = 0.0

    while global_step < epochs * len(batches):
        current_index = global_step % len(batches)
        batch, prompt_len = batches[current_index]

        # Move the input batch to the device
        inputs = batch.to(model.device)
        labels = inputs.clone()
        labels[:, :prompt_len] = -100

        # Forward pass
        outputs = model(inputs, labels=labels)

        # Calculate loss
        loss = outputs.loss / accumulation_steps
        loss.backward()

        if (global_step + 1) % accumulation_steps == 0:
            warmup_optimizer.step()
            lisa_callback.on_step_begin(n_acc_steps)

            if grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip)

            n_acc_steps += 1

            if current_lr < warmup_end_lr:
                if n_acc_steps % warmup_lr_increase_steps == 0:
                    current_lr = current_lr * 10
                    for g in warmup_optimizer.param_groups:
                        g['lr'] = current_lr
            else:
                if n_acc_steps % lr_change_steps == 0:
                    current_lr = current_lr / 10
                    for g in warmup_optimizer.param_groups:
                        g['lr'] = current_lr

            warmup_optimizer.zero_grad(set_to_none=True)

            print(
                f"Step: {n_acc_steps}, Training Loss: {outputs.loss.detach().item()}")
            run.log({"loss": outputs.loss.detach().item()})

        torch.cuda.empty_cache()

        if (global_step + 1) % eval_steps == 0:
            # Evaluate the model
            eval_losses = ft.validation.compute_losses(
                model=model, batches=eval_batches, device=model.device
            )
            eval_loss = torch.mean(torch.tensor(eval_losses))
            print(f"Step: {n_acc_steps}, Eval Loss: {eval_loss}")
            run.log({"eval_loss": eval_loss})

        torch.cuda.empty_cache()

        global_step += 1
        total_loss += outputs.loss.detach().item()
        losses += outputs.loss.detach().item()

    # Return average loss, standard deviation of loss, final learning rate, and model
    return total_loss / global_step, torch.std(torch.tensor(losses)), current_lr, model
