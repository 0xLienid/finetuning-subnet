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
import copy
import wandb
import torch
import random
import argparse
import constants
import typing
import numpy as np
from model.model_updater import ModelUpdater
from model.storage.chain.chain_model_metadata_store import ChainModelMetadataStore
from model.storage.hugging_face.hugging_face_model_store import HuggingFaceModelStore
import finetune as ft
import bittensor as bt
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainingArguments, AutoTokenizer
from datasets import Dataset, load_dataset
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
        "--load_repo",
        type=str,
        default=None,
        help="If provided, loads a previously trained HF model from the specified repo",
    )
    parser.add_argument(
        "--download_repo_latest_commit",
        type=str,
        default=None,
        help="Specifies the latest commit for the remote repo. This is required to identify the local snapshot download path.",
    )
    parser.add_argument(
        "--comparison_uid",
        type=int,
        help="Compares the model under the specified uid to the model being trained.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
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
        default=20480,
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
    parser.add_argument(
        "--use_cortex",
        action="store_true",
        help="If set, the miner uses Cortex subnet data to train.",
    )
    parser.add_argument(
        "--dataset_repo_id",
        type=str,
        default=None,
        help="If provided, uses the dataset from the specified repo to train.",
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

    # Check if we should load a model from a remote repo.
    if config.load_repo:
        if not config.download_repo_latest_commit:
            raise RuntimeError(
                f"Latest commit hash not specified for repo {config.load_repo}")

        model, tokenizer = await actions.load_repo_model(
            config.load_repo, config.download_repo_latest_commit, config.model_dir, model_parameters)
        return model, tokenizer

    raise RuntimeError(
        "No starting model specified, pass either --load_best, --load_uid, --load_model_dir, or --load_repo")


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
        hf_token = HuggingFaceModelStore.assert_access_token_exists()

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
    model, _ = await load_starting_model(miner_actions, config, metagraph, model_parameters)
    model = model.train()
    model = model.to(config.device)

    # Init tokenizer and adjust chat template
    tokenizer = AutoTokenizer.from_pretrained(
        "NousResearch/gemma-2b-it-tokenizer")
    if not tokenizer.chat_template.endswith("{{ eos_token }}"):
        template = tokenizer.chat_template
        template = template + "{{ eos_token }}"
        tokenizer.chat_template = template
        del template

    bt.logging.success(f"Saving model to path: {model_dir}.")
    miner_actions.save(model, tokenizer, model_dir)

    # Set up hyperparameter search
    hyperparams = {
        "learning_rate": [1e-5, 1e-6, 1e-8, 1e-10],
        "r": [32, 64, 128],
        "alpha": [16]
    }
    hyperparam_combinations = ft.training.generate_hyperparam_combinations(
        hyperparams)

    # Load and set up the dataset
    batches = []
    if config.use_cortex:
        loader = ft.dataset.CortexSubsetLoader(
            latest=True, running=True,
            random_seed=random.randint(0, 100000000),
            max_samples=config.cortex_samples_per_epoch,
            steps=config.cortex_steps,
            page_size=1
        )
        batches = loader.tokenize(tokenizer)
        del loader
    elif config.dataset_repo_id is not None:
        dataset = load_dataset(config.dataset_repo_id, split="train")

        # Optional: Dataset preprocessing
        # perplexity_column = np.array(dataset["perplexity"])
        # perplexity_column = perplexity_column[~np.isnan(perplexity_column)]
        # thirtieth_percentile = np.percentile(perplexity_column, 30)
        # eightieth_percentile = np.percentile(perplexity_column, 80)
        # dataset = dataset.filter(
        #     lambda example: example["perplexity"] > thirtieth_percentile and example["perplexity"] < eightieth_percentile)

        batches = ft.training.tokenize_dataset(tokenizer, dataset)
        del dataset

    # Load and set up eval dataset
    eval_loader = ft.dataset.CortexSubsetLoader(
        latest=True, running=True,
        random_seed=random.randint(0, 100000000),
        max_samples=32 * 3,
        steps=1,
        page_size=1
    )
    eval_batches = eval_loader.tokenize(tokenizer)

    # Calculate T_max and eta_min factor for learning rate scheduler
    T_max = config.num_epochs * \
        (int(len(batches) / config.accumulation_steps) + 1)
    eta_min_factor = 0.01

    # Store data from best run
    best_loss = math.inf
    best_std = math.inf
    best_hyperparams = None

    num_search_batches = len(batches) // 10

    # Train the model with each hyperparameter combination
    for i, combination in enumerate(hyperparam_combinations):
        print(f"Hyperparam combination: {i}")

        run_model = copy.deepcopy(model)
        run_avg_loss, run_loss_std, _, _ = ft.training.train(
            run_model,
            batches[:num_search_batches],
            1,
            config.accumulation_steps,
            combination["learning_rate"],
            combination["r"],
            combination["alpha"],
            T_max,
            eta_min_factor,
            config.wandb_project
        )

        if run_avg_loss < best_loss:
            best_loss = run_avg_loss
            best_std = run_loss_std
            best_hyperparams = combination

    # Log the best hyperparameters
    bt.logging.success(f"Best loss: {best_loss}")
    bt.logging.success(f"Best loss std: {best_std}")
    bt.logging.success(f"Best hyperparams: {best_hyperparams}")

    # Run full training with best hyperparameters
    _, _, _, lora_model = ft.training.train(
        model,
        batches,
        eval_batches,
        config.num_epochs,
        config.accumulation_steps,
        best_hyperparams["learning_rate"],
        best_hyperparams["r"],
        best_hyperparams["alpha"],
        T_max,
        eta_min_factor,
        config.wandb_project
    )
    del model, batches, eval_batches

    # Merge weights and save the model
    model = lora_model.merge_and_unload()
    miner_actions.save(model, tokenizer, model_dir +
                       best_hyperparams["learning_rate"] + "_" + best_hyperparams["r"] + "_" + best_hyperparams["alpha"])
    bt.logging.success("Saving merged LoRA model")

    # Get new eval batches
    eval_loader = ft.dataset.CortexSubsetLoader(
        latest=True, running=True,
        random_seed=random.randint(0, 100000000),
        max_samples=32 * 3,
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

    # Compute losses for comparison model
    comparison_model, _ = await miner_actions.load_remote_model(config.comparison_uid, metagraph, "temp_comparison_model")
    comparison_losses = ft.validation.compute_losses(
        model=comparison_model,
        batches=eval_batches,
        device=config.device
    )
    bt.logging.success("Evaluated comparison model for comparison")

    # Clear memory
    del comparison_model, eval_loader, eval_batches

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

    bt.logging.success(f"Comparison win rate: {num_wins / len(local_losses)}")
    bt.logging.success(
        f"Comparison average loss: {sum_loss / len(local_losses)}")

    if not config.offline:
        model_to_upload, tokenizer_to_upload = miner_actions.load_local_model(
            model_dir, model_parameters)
        await miner_actions.push(model_to_upload, tokenizer_to_upload, model_parameters)


if __name__ == "__main__":
    # Parse and print configuration
    config = get_config()
    if config.list_competitions:
        print(constants.COMPETITION_SCHEDULE)
    else:
        print(config)
        asyncio.run(main(config))
