# Setup

Create a new environment using conda.

```bash
conda create -p .venv_lflexmdm python=3.11.10 pip ipykernel -y
conda activate ./.venv_lflexmdm
pip install -r requirements.txt
```

Create `.env` file in the root directory from where you plan to run the  experiments, edit and add the following environment variables in the file:
```bash
# wandb
WANDB_ENTITY=???
WANDB_PROJECT=???
# data
DATA_DIR=data
HF_HOME=hf_home
HF_DATASETS_CACHE=hf_datasets_cache
# output
LOG_DIR=logs
# misc
TOKENIZERS_PARALLELISM=false
PROJECT_ROOT=.
# hydra
HYDRA_FULL_ERROR=1
OC_CAUSE=1
# emails from slurm scheduler if applicable
EMAIL=???
# torch compile logs
TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
```