# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 const

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import asyncio
import math
import os
import wandb
import torch
import random
import argparse
import constants
import typing
from model.model_updater import ModelUpdater
from model.storage.chain.chain_model_metadata_store import ChainModelMetadataStore
from model.storage.hugging_face.hugging_face_model_store import HuggingFaceModelStore
import finetune as ft
import bittensor as bt
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainingArguments
from datasets import Dataset
from trl import DPOTrainer
from finetune.mining import Actions
from utilities import utils
import datetime as dt
import pandas as pd

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

os.environ["TOKENIZERS_PARALLELISM"] = "true"


# === Config ===
def get_config():
    """
    Set up and parse the command-line arguments to configure the system.

    The configuration is responsible for setting up the environment including
    the model path, device to use, and the bittensor wallet and logging configurations.

    Returns:
        A namespace object containing the configuration parameters.
    """

    # Initialize an argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--offline",
        action="store_true",
        help="Does not launch a wandb run, does not send model to wandb, does not check if registered",
    )
    parser.add_argument(
        "--wandb_project", type=str, help="The wandb project to log to."
    )
    parser.add_argument("--wandb_entity", type=str,
                        help="The wandb entity to log to.")
    parser.add_argument(
        "--hf_repo_id",
        type=str,
        help="The hugging face repo id, which should include the org or user and repo name. E.g. jdoe/finetuned",
    )
    parser.add_argument(
        "--avg_loss_upload_threshold",
        type=float,
        default=0,  # Default to never uploading.
        help="The threshold for avg_loss the model must achieve to upload it to hugging face. A miner can only advertise one model, so it should be the best one.",
    )
    parser.add_argument(
        "--model_dir",
        default=os.path.join(constants.ROOT_DIR, "local-models/"),
        help="Where to download/save models for training",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="The device on which to run. cpu or cuda",
    )
    parser.add_argument(
        "--load_best",
        action="store_true",
        help="If set, the miner loads the best model from wandb to train off.",
    )
    parser.add_argument(
        "--load_uid",
        type=int,
        default=None,
        help="If passed loads the model under the specified uid.",
    )
    parser.add_argument(
        "--load_model_dir",
        type=str,
        default=None,
        help="If provided, loads a previously trained HF model from the specified directory",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=-1,
        help="Number of training epochs (-1 is infinite)",
    )
    parser.add_argument("--lr", type=float, default=0.00001,
                        help="Learning rate.")
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=32,
        help="The number of training accumulation steps.",
    )
    parser.add_argument(
        "--cortex_steps",
        type=int,
        default=5,
        help="Number of Cortex steps to sample data from",
    )
    parser.add_argument(
        "--cortex_samples_per_epoch",
        type=int,
        default=8192,
        help="Number of samples trained on per epoch",
    )
    parser.add_argument(
        "--attn_implementation",
        default="flash_attention_2",
        help="Implementation of attention to use",
    )
    parser.add_argument(
        "--netuid",
        type=str,
        default=constants.SUBNET_UID,
        help="The subnet UID.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="datatype to load model in, either bfloat16 or float16",
    )
    parser.add_argument(
        "--competition_id",
        type=str,
        default=constants.ORIGINAL_COMPETITION_ID,
        help="competition to mine for (use --list-competitions to get all competitions)"
    )
    parser.add_argument(
        "--list_competitions",
        action="store_true",
        help="Print out all competitions"
    )

    # Include wallet and logging arguments from bittensor
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)

    # Parse the arguments and create a configuration namespace
    config = bt.config(parser)

    return config


async def load_starting_model(
    actions: Actions, config: bt.config, metagraph: bt.metagraph,
    model_parameters: constants.CompetitionParameters
) -> typing.Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Loads the model to train based on the provided config."""

    # Initialize the model based on the best on the network.
    if config.load_best:
        # Get the best UID be incentive and load it.
        best_uid = ft.graph.best_uid(metagraph)
        model, tokenizer = await actions.load_remote_model(best_uid, metagraph, config.model_dir)
        bt.logging.success(
            f"Training with model from best uid: {best_uid}. Model={str(model)}"
        )
        return model, tokenizer

    # Initialize the model based on a passed uid.
    if config.load_uid is not None:
        # Sync the state from the passed uid.
        model, tokenizer = await actions.load_remote_model(
            config.load_uid, metagraph, config.model_dir
        )
        bt.logging.success(
            f"Training with model from uid: {config.load_uid}. Model={str(model)}"
        )
        return model, tokenizer

    # Check if we should load a model from a local directory.
    if config.load_model_dir:
        model, tokenizer = actions.load_local_model(
            config.load_model_dir, model_parameters)
        bt.logging.success(
            f"Training with model from disk. Model={str(model)}")
        return model, tokenizer

    raise RuntimeError(
        "No starting model specified, pass either --load_best, --load_uid, or --load_model_dir")


async def main(config: bt.config):
    # Create bittensor objects.
    bt.logging(config=config)

    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(config.netuid)

    # If running online, make sure the miner is registered, has a hugging face access token, and has provided a repo id.
    my_uid = None
    if not config.offline:
        my_uid = utils.assert_registered(wallet, metagraph)
        HuggingFaceModelStore.assert_access_token_exists()

    # Configure the stores and miner actions.
    miner_actions = ft.mining.Actions.create(config, wallet, subtensor)

    # Create a unique run id for this run.
    run_id = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_dir = ft.mining.model_path(config.model_dir, run_id)
    os.makedirs(model_dir, exist_ok=True)

    model_parameters = ModelUpdater.get_competition_parameters(
        config.competition_id)
    model_parameters.kwargs["torch_dtype"] = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16
    model_parameters.kwargs["attn_implementation"] = config.attn_implementation

    # Init model.
    model, tokenizer = await load_starting_model(miner_actions, config, metagraph, model_parameters)
    model = model.train()
    model = model.to(config.device)

    # Adjust chat template
    if not tokenizer.chat_template.endswith("{{ eos_token }}"):
        template = tokenizer.chat_template
        template = template + "{{ eos_token }}"
        tokenizer.chat_template = template
        del template

    bt.logging.success(f"Saving model to path: {model_dir}.")
    miner_actions.save(model, tokenizer, model_dir)

    # Build optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=0.01)

    # Start the training loop
    global_step = 0
    n_acc_steps = 0
    accumulation_steps = config.accumulation_steps

    try:
        # Initialize loss accumulator for the epoch
        epoch_loss = 0.0

        # Prepare the data loader with random pages for each epoch
        bt.logging.success(
            f"Loading {config.cortex_samples_per_epoch} pages for training this epoch"
        )
        loader = ft.dataset.CortexSubsetLoader(
            latest=False,
            random_seed=random.randint(0, 100000000),
            max_samples=config.cortex_samples_per_epoch,
            steps=config.cortex_steps,
            page_size=config.cortex_steps,
        )
        batches = loader.tokenize(tokenizer)

        # Enumerate over the data loader
        n_batches = 0
        optimizer.zero_grad()  # Initialize gradients to zero

        for i, (batch, prompt_len) in enumerate(batches):
            # Move the input batch to the device
            inputs = batch.to(model.device)
            labels = inputs.clone()
            labels[:, :prompt_len] = -100

            # Forward pass: compute the model output and loss
            outputs = model(inputs, labels=labels)

            loss = outputs.loss / accumulation_steps  # Scale loss
            loss.backward()  # Accumulate gradients

            if (i + 1) % accumulation_steps == 0:
                n_acc_steps += 1
                optimizer.step()  # Perform a single optimization step
                optimizer.zero_grad()  # Clear gradients
                bt.logging.success(
                    f"Step: {n_acc_steps} loss: {outputs.loss.detach().item()}"
                )

            torch.cuda.empty_cache()

            n_batches += 1
            global_step += 1
            epoch_loss += outputs.loss.detach().item()

        # Calculate the average loss for the epoch
        avg_loss = epoch_loss / n_batches

        # Clear memory
        del loader, batches, optimizer

        # Log the average loss for the epoch
        bt.logging.success(f"Pretraining average loss: {avg_loss}")

        # Self-Play
        while True:
            # Load data for head-to-head evaluation against best model
            eval_loader = ft.dataset.CortexSubsetLoader(
                latest=True, running=True,
                random_seed=random.randint(0, 100000000),
                max_samples=50,
                steps=1,
                page_size=1
            )
            prompts = eval_loader.get_prompts()
            eval_batches = eval_loader.tokenize(tokenizer)

            # Compute losses and get responses for local model
            local_losses_and_outputs = ft.validation.compute_losses_with_outputs(
                model=model,
                tokenizer=tokenizer,
                batches=eval_batches,
                temperature=1.1,
                device=config.device
            )
            bt.logging.success("Local model evaluated")

            # Load the best model from the network and compute losses
            best_model, best_tokenizer = await miner_actions.load_remote_model(config.load_uid, metagraph, config.model_dir)
            best_losses_and_outputs = ft.validation.compute_losses_with_outputs(
                model=best_model,
                tokenizer=best_tokenizer,
                batches=eval_batches,
                temperature=0.8,
                device=config.device
            )
            bt.logging.success("Best model evaluated")
            del best_tokenizer, eval_loader, eval_batches

            # Compare the losses and build DPO dataset of winning responses
            num_wins = 0
            sum_loss = 0.0
            dpo_dataset = []
            for result_pair in zip(local_losses_and_outputs, best_losses_and_outputs, prompts):
                local = result_pair[0]
                best = result_pair[1]
                prompt = result_pair[2]
                iswin = ft.validation.iswin(local["loss"], best["loss"], 1, 0)
                sum_loss += local["loss"]

                if iswin:
                    num_wins += 1
                    dpo_dataset.append({
                        "question": prompt,
                        "chosen": local["response"],
                        "rejected": best["response"]
                    })
                else:
                    dpo_dataset.append({
                        "question": prompt,
                        "chosen": best["response"],
                        "rejected": local["response"]
                    })

            # Log the win rate and average loss
            bt.logging.success(
                f"Initial win rate: {num_wins / len(local_losses_and_outputs)}")
            bt.logging.success(
                f"Initial average loss: {sum_loss / len(local_losses_and_outputs)}")

            # Clear memory
            del local_losses_and_outputs, best_losses_and_outputs

            # Set up DPO training
            prompt_column = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": sample["question"]}],
                    truncation=True,
                    return_tensors="pt",
                    max_length=constants.sequence_length,
                    tokenize=False,
                    add_generation_prompt=True
                )
                for sample in dpo_dataset]
            self_play_dataset = Dataset.from_pandas(
                pd.DataFrame(data=dpo_dataset))
            self_play_dataset = self_play_dataset.add_column(
                "prompt", prompt_column)

            # Train the DPO model
            dpo_trainer = DPOTrainer(
                model,
                best_model,
                args=TrainingArguments(
                    output_dir=model_dir,
                    per_device_train_batch_size=2,
                    gradient_accumulation_steps=32,
                    learning_rate=1e-9,
                    num_train_epochs=1,
                ),
                beta=0.1,
                train_dataset=self_play_dataset,
                tokenizer=tokenizer,
            )
            dpo_trainer.train()

            # Save model locally
            miner_actions.save(model, tokenizer, model_dir)
            bt.logging.success(f"Saved model to {model_dir}")

            # Clear memory
            del best_model, self_play_dataset, dpo_trainer

            bt.logging.success("Finished self-play loop")

            # Get final eval dataset
            eval_loader = ft.dataset.CortexSubsetLoader(
                latest=True, running=True,
                random_seed=random.randint(0, 100000000),
                max_samples=20,
                steps=1,
                page_size=1
            )
            eval_batches = eval_loader.tokenize(tokenizer)

            # Compute losses for local model
            local_losses = ft.validation.compute_losses(
                model=model,
                batches=eval_batches,
                device=config.device
            )
            bt.logging.success("Evaluated local model for comparison")

            # Get comparison model and compute losses
            comparison_uid = 246
            comparison_model, _ = await miner_actions.load_remote_model(comparison_uid, metagraph, "temp_comparison_model")
            comparison_losses = ft.validation.compute_losses(
                model=comparison_model,
                batches=eval_batches,
                device=config.device
            )
            bt.logging.success("Evaluated external model for comparison")

            # Clear memory
            del eval_loader, eval_batches, comparison_model

            print(local_losses)
            print(comparison_losses)

            # Calculate win rate
            num_wins = 0
            sum_loss = 0.0
            for result_pair in zip(local_losses, comparison_losses):
                local_loss = result_pair[0]
                comparison_loss = result_pair[1]
                iswin = ft.validation.iswin(local_loss, comparison_loss, 1, 0)
                sum_loss += local_loss

                if iswin:
                    num_wins += 1

            bt.logging.success(
                f"Comparison win rate: {num_wins / len(local_losses)}")
            bt.logging.success(
                f"Comparison average loss: {sum_loss / len(local_losses)}")

            if not config.offline:
                model_to_upload, tokenizer_to_upload = miner_actions.load_local_model(
                    model_dir, model_parameters)
                await miner_actions.push(model_to_upload, tokenizer_to_upload, model_parameters)
    except Exception as e:
        bt.logging.error(f"Failed with error: {e}")


if __name__ == "__main__":
    # Parse and print configuration
    config = get_config()
    if config.list_competitions:
        print(constants.COMPETITION_SCHEDULE)
    else:
        print(config)
        asyncio.run(main(config))
