"""
A script that creates a HuggingFace dataset from a set of Cortex data and the associated
perplexity scores as calculated by a reference model.

Prerequisites:
    1. HF_ACCESS_TOKEN is set in the environment or .env file
"""

import asyncio
import argparse
import torch
import random
import constants
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset, concatenate_datasets
import bittensor as bt
import finetune as ft
from model.storage.hugging_face.hugging_face_model_store import HuggingFaceModelStore

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.


def get_config():
    # Initialize an argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ref_model",
        type=str,
        help="The hugging face repo id of the reference model, which should include the org or user and repo name. E.g. jdoe/finetuned",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="The hugging face repo id of the tokenizer, which should include the org or user and repo name. E.g. jdoe/finetuned",
    )
    parser.add_argument(
        "--cortex_data_points",
        type=int,
        default=50000,
        help="The number of data points to pull from the Cortex dataset"
    )
    parser.add_argument(
        "--hf_dataset_id",
        type=str,
        help="The hugging face dataset id to save to, which should include the org or user and repo name. E.g. jdoe/finetuned",
    )

    return parser.parse_args()


def tokenize_data(tokenizer, dataset):
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


async def main(ref_model: str, tokenizer: str, cortex_data_points: int, hf_dataset_id: str):
    # Make sure we have a HuggingFace token
    hf_token = HuggingFaceModelStore.assert_access_token_exists()

    # Create model and tokenizer objects
    model = AutoModelForCausalLM.from_pretrained(
        model=ref_model,
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    # Prepare the data loader
    loader = ft.dataset.CortexSubsetLoader(
        latest=False,
        random_seed=random.randint(0, 100000000),
        max_samples=cortex_data_points,
        steps=5,
        page_size=5
    )
    batches = loader.tokenize(tokenizer)

    # Calculate losses
    device = "cuda" if torch.cuda.is_available() else "cpu"
    losses = ft.validation.compute_losses(
        model=model,
        batches=batches,
        device=device
    )
    perplexities = torch.exp(torch.tensor([loss for loss in losses]))

    # Create the dataset
    prompts = [sample[0] for sample in loader.buffer]
    responses = [sample[1] for sample in loader.buffer]
    dataset = Dataset.from_dict({
        "question": prompts,
        "response": responses,
        "perplexity": perplexities
    })

    # Clear memory
    del loader, batches, losses, perplexities

    # Load OpenOrca dataset
    open_orca = load_dataset("Open-Orca/OpenOrca", split="train")
    open_orca = open_orca.remove_columns(
        [col for col in open_orca.column_names if col not in ["question", "response"]])
    open_orca = open_orca.shuffle(seed=42).select(range(50000))
    batches = tokenize_data(tokenizer, open_orca)

    # Calculate losses
    losses = ft.validation.compute_losses(
        model=model,
        batches=batches,
        device=device
    )
    perplexities = torch.exp(torch.tensor([loss for loss in losses]))

    # Create the dataset
    prompts = open_orca["question"]
    responses = open_orca["response"]
    dataset = concatenate_datasets([
        dataset,
        Dataset.from_dict({
            "question": prompts,
            "response": responses,
            "perplexity": perplexities
        })
    ])

    # Clear memory
    del open_orca, batches, losses, perplexities

    # Save the dataset
    dataset.push_to_hub(hf_dataset_id, token=hf_token)

if __name__ == "__main__":
    config = get_config()
    asyncio.run(main(config.ref_model, config.tokenizer,
                     config.cortex_data_points, config.hf_dataset_id))
