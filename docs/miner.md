# Miner

Miners train locally and periodically publish their best model to 🤗 Hugging Face and commit the metadata for that model to the Bittensor chain.

Miners can only have one model associated with them on the chain for evaluation by validators at a time.

The communication between a miner and a validator happens asynchronously and therefore Miners do not need to be running continuously. Validators will use whichever metadata was most recently published by the miner to know which model to download from 🤗 Hugging Face. In the event that your model is marked as stale by the validator set, you will need to resubmit a new version of the model to continue being evaluated.

# System Requirements

Miners will need enough disk space to store their model as they work on. Each uploaded model (As of Jan 1st, 2024) may not be more than 15 GB. It is recommended to have at least 50 GB of disk space.

Miners will need enough processing power to train their model. The device the model is trained on is recommended to be a large GPU with atleast 48 GB of VRAM for 2b models. For this miner version you will need 80 GB for 7b models given the way models are cloned for hyperparameter search.

# Getting started

## Prerequisites

1. Get a Hugging Face Account:

Miner and validators use 🤗 Hugging Face in order to share model state information. Miners will be uploading to 🤗 Hugging Face and therefore must attain a account from [🤗 Hugging Face](https://huggingface.co/) along with a user access token which can be found by following the instructions [here](https://huggingface.co/docs/hub/security-tokens).

Make sure that any repo you create for uploading is public so that the validators can download from it for evaluation.

2. Clone the repo

```shell
git clone https://github.com/0xLienid/finetuning-subnet.git
```

3. Setup your python [virtual environment](https://docs.python.org/3/library/venv.html) or [Conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands).

4. Install the requirements. From your virtual environment, run

```shell
cd finetuning-subnet
python -m pip install -e .
```

If this gives you issues with `packaging` or `torch` run

```shell
python -m pip install -U wheel packaging torch
```

5. Make sure you've [created a Wallet](https://docs.bittensor.com/getting-started/wallets) and [registered a hotkey](https://docs.bittensor.com/subnets/register-and-participate).

6. (Optional) Run a Subtensor instance:

Your node will run better if you are connecting to a local Bittensor chain entrypoint node rather than using Opentensor's.
We recommend running a local node as follows and passing the `--subtensor.network local` flag to your running miners/validators.
To install and run a local subtensor node follow the commands below with Docker and Docker-Compose previously installed.

```bash
git clone https://github.com/opentensor/subtensor.git
cd subtensor
docker compose up --detach
```

---

# Running the Miner

The mining script uploads a model to 🤗 Hugging Face which will be evaluated by validators.

See [Validator Psuedocode](docs/validator.md#validator) for more information on how they the evaluation occurs.

## Env File

The miner requires a .env file with your 🤗 Hugging Face access token in order to upload models and a WandB API key for tracking training runs.

Create a `.env` file in the `finetuning-subnet` directory and add the following to it:

```shell
HF_ACCESS_TOKEN="YOUR_HF_ACCESS_TOKEN"
WANDB_API_KEY="YOUR_WANDB_API_KEY"
```

## Hyperparameter Search

The miner runs hyperparameter search over a search space defined in `neurons/miner.py`. The default is:

```python
hyperparams = {
    "learning_rate": [1e-5, 1e-6, 1e-8, 1e-10],
    "r": [32, 64, 128],
    "alpha": [16]
}
```

If you would like a different search space, adjust these arrays as needed before running the miner.

## Starting the Miner

To start your miner the most basic command is

```shell
python neurons/miner.py --wallet.name coldkey --wallet.hotkey hotkey --hf_repo_id my-username/my-project --avg_loss_upload_threshold YOUR_THRESHOLD
```

- `--wallet.name`: should be the name of the coldkey that contains the hotkey your miner is registered with.

- `--wallet.hotkey`: should be the name of the hotkey that your miner is registered with.

- `--hf_repo_id`: should be the namespace/model_name that matches the hugging face repo you want to upload to. Must be public so that the validators can download from it.

- `--avg_loss_upload_threshold`: should be the minimum average loss before you want your miner to upload the model.

- `--competition_id`: competition you wish to mine for; run `--list_competitions` to get a list of available competitions

### Flags

The Miner offers some flags to customize properties, such as how to train the model and which hugging face repo to upload to.

You can view the full set of flags by running

```shell
python ./neurons/miner.py -h
```

Some flags you may find useful:

- `--offline`: when set you can run the miner without being registered and it will not attempt to upload the model.

- `--wandb_entity` + `--wandb_project`: when both flags are set the miner will log its training to the provided wandb project.

- `--device`: by default the miner will use your gpu but you can specify with this flag if you have multiple.

- `--comparison_uid`: sets the UID of the the miner you wish to run head-to-head loss comparison against at the end of the training run

- `--use_cortex`: denotes that you would like to pull data from the Cortex subnet for training

- `--dataset_repo_id`: if you are not using Cortex subnet data, this determines what dataset from 🤗 Hugging Face to train on

#### Training from pre-existing models

- `--load_best`: when set you will download and train the model from the current best miner on the network.
- `--load_uid`: when passing a uid you will download and train the model from the matching miner on the network.
- `--load_model_dir`: the path to a local model directory [saved via Hugging Face API].
- `--load_repo`: the 🤗 Hugging Face repo id to load. you will also need to pass the latest commit hash with `--download_repo_latest_commit`

---

## Manually uploading a model

In some cases you may have failed to upload a model or wish to upload a model without further training.

Due to rate limiting by the Bittensor chain you may only upload a model every 20 minutes.

You can manually upload with the following command:

```shell
python scripts/upload_model.py --load_model_dir <path to model> --hf_repo_id my-username/my-project --wallet.name coldkey --wallet.hotkey hotkey
```

## Running a custom Miner

As of March 1st, 2024 the subnet works with mistral models supported by [LlamaForCausalLM](https://huggingface.co/docs/transformers/v4.37.2/en/model_doc/llama2#transformers.LlamaForCausalLM) or the Gemma model subject to the following constraints:

1. Has less than 7B parameters.
2. Total size of the repo is less than 15 Gigabytes.
3. 2K max token sequence length.
4. Utilizes the default chat format of the model selected.

The `finetune/mining.py` file has several methods that you may find useful. Example below.

```python
import pretrainas ft
import bittensor as bt
from transformers import PreTrainedModel

config = bt.config(...)
wallet = bt.wallet()
metagraph = bt.metagraph(netuid=6)

actions = ft.mining.actions.Actions.create(config, wallet)

# Load a model from another miner.
model: PreTrainedModel = actions.load_remote_model(uid=123, metagraph=metagraph, download_dir="mydir")

# Save the model to local file.
actions.save(model, "model-foo/")

# Load the model from disk.
actions.load_local_model("model-foo/")

# Publish the model for validator evaluation.
actions.push(model)
```
