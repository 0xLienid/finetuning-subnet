import os
import random
import itertools
import torch
import constants
import wandb
from peft import LoraConfig, get_peft_model
import finetune as ft


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
    r,
    alpha,
    T_max,
    eta_min_factor
):
    """Trains a LoRA model on the provided data with the provided hyperparams."""
    wandb.login(key=os.getenv["WANDB_API_KEY"])
    run = wandb.init(
        project="nous-lora-finetuning",
        config={"learning_rate": learning_rate, "epochs": epochs}
    )

    # Set up LoRA model
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules="all",
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM"
    )
    lora_model = get_peft_model(model, config)

    # Build optimizer
    optimizer = torch.optim.AdamW(
        lora_model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=T_max, eta_min=eta_min_factor * learning_rate)

    # Train model
    epoch_step = 0
    global_step = 0
    n_acc_steps = 0
    total_loss = 0.0
    losses = []

    while epoch_step < epochs or epochs == -1:
        epoch_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        for i, (batch, prompt_len) in enumerate(batches):
            # Move the input batch to the device
            inputs = batch.to(lora_model.device)
            labels = inputs.clone()
            labels[:, :prompt_len] = -100

            # Forward pass
            outputs = lora_model(inputs, labels=labels)

            # Calculate loss
            loss = outputs.loss / accumulation_steps  # Scale loss
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                n_acc_steps += 1
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                print(
                    f"Step: {n_acc_steps}, Training Loss: {outputs.loss.detach().item()}")
                run.log({"loss": outputs.loss.detach().item()})

            torch.cuda.empty_cache()

            if (i + 1) % eval_steps == 0:
                # Evaluate the model
                eval_losses = ft.validation.compute_losses(
                    model=lora_model, batches=eval_batches, device=lora_model.device
                )
                eval_loss = torch.mean(torch.tensor(eval_losses))
                print(f"Step: {n_acc_steps}, Eval Loss: {eval_loss}")
                run.log({"eval_loss": eval_loss})

            torch.cuda.empty_cache()

            n_batches += 1
            global_step += 1
            epoch_loss += outputs.loss.detach().item()
            total_loss += outputs.loss.detach().item()
            losses.append(outputs.loss.detach().item())

        # Clear memory
        lr = scheduler.get_last_lr()[0]
        del optimizer, scheduler

        # Log the average loss for the epoch
        print(f"Epoch: {epoch_step} average loss: {epoch_loss / len(batches)}")
        epoch_step += 1

    # Log the average loss for the training
    print(f"Training average loss: {total_loss / global_step}")

    # Return average loss, standard deviation of loss, final learning rate, and model
    return total_loss / global_step, torch.std(torch.tensor(losses)), lr, lora_model
