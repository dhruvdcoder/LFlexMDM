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
# paths to appropriate directories
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

# Download pretrained weights

```bash
pip install gdown
gdown --folder -O logs 1m_b8IQ0r4LJJheIhoqn3jWper1X41Lvb
```

# Eval

**(Baseline) FlexMDM on Graph Traversal**

```bash
DATASET=star_hard # or star_easy, star_medium
MODEL=flexmdm
CONFIDENCE=null # or prob_diff or null or top_prob
EXPERIMENT=${DATASET}_${MODEL}
xlm job_type=eval job_name=${DATASET}_${MODEL} experiment=$EXPERIMENT ++eval.model_only_checkpoint_path=logs/${DATASET}_${MODEL}/checkpoints/model_state_dict.pth ++eval.split=test per_device_batch_size=64 global_batch_size=64 trainer_strategy=single_device ++trainer.precision=32-true compile=false +tags.eval_type=exact_match ~datamodule.dataset_managers.test.lm ++predictor.confidence=${CONFIDENCE} predictor.max_steps=500 ++predictor.top_k=1 ++predictor.top_p=null
```

**(Ours) LFlexMDM on Graph Traversal**
```bash
DATASET=star_hard # or star_easy, star_medium
MODEL=lflexmdm
CONFIDENCE=null # or prob_diff or null or top_prob
BACKBONE=separate # or shared
if [ "$BACKBONE" = "separate" ]; then
    EXPERIMENT="[${DATASET}_${MODEL},b_um_1_a1]"
else
    EXPERIMENT="[${DATASET}_${MODEL}_shared,b_um_1_a1]"
fi
CHECKPOINT_PATH=logs/${DATASET}_${MODEL}_${BACKBONE}/checkpoints/model_state_dict.pth
xlm job_type=eval job_name=${DATASET}_${MODEL}_${BACKBONE} experiment=$EXPERIMENT ++eval.model_only_checkpoint_path=${CHECKPOINT_PATH} ++eval.split=test per_device_batch_size=64 global_batch_size=64 trainer_strategy=single_device ++trainer.precision=32-true compile=false +tags.eval_type=exact_match ~datamodule.dataset_managers.test.lm ++predictor.confidence=${CONFIDENCE} predictor.max_steps=500 ++predictor.top_k=1 ++predictor.top_p=null
```

**(Baseline) FlexMDM on molecule generation**
```bash
DATASET=safe
MODEL=flexmdm
CONFIDENCE=null # or prob_diff or null or top_prob
TOP_P=0.2 # or 0.5 or 1.0
EXPERIMENT=${DATASET}_${MODEL}
CHECKPOINT_PATH=logs/${DATASET}_${MODEL}/checkpoints/model_state_dict.pth
xlm job_type=eval job_name=${DATASET}_${MODEL}_${BACKBONE} experiment=$EXPERIMENT ++eval.model_only_checkpoint_path=${CHECKPOINT_PATH} ++eval.split=test per_device_batch_size=64 global_batch_size=64 trainer_strategy=single_device ++trainer.precision=32-true compile=false +tags.eval_type=molgen datamodule.dataset_managers.test.unconditional_prediction.num_examples=1000 datamodule.dataset_managers.val.conditional_prediction.num_examples=1000 ~datamodule.dataset_managers.val.lm ~datamodule.dataset_managers.test.lm ++predictor.confidence=${CONFIDENCE} predictor.max_steps=1024 ++predictor.top_k=null ++predictor.top_p=${TOP_P}
```

**(Ours) LFlexMDM on molecule generation**
```bash
DATASET=safe
MODEL=lflexmdm
CONFIDENCE=null # or prob_diff or null or top_prob
BACKBONE=separate # or shared
TOP_P=0.2 # or 0.5 or 1.0
aux_size=medium # or tiny or xtiny
if [ "$BACKBONE" = "separate" ]; then
    EXPERIMENT="[${DATASET}_${MODEL},b_um_1_a1,aux_size=${aux_size}]"
else
    EXPERIMENT="[${DATASET}_${MODEL}_shared,b_um_1_a1]"
fi
CHECKPOINT_PATH=logs/${DATASET}_${MODEL}_${BACKBONE}_${aux_size}/checkpoints/model_state_dict.pth
xlm job_type=eval job_name=${DATASET}_${MODEL}_${BACKBONE} experiment=$EXPERIMENT ++eval.model_only_checkpoint_path=${CHECKPOINT_PATH} ++eval.split=test per_device_batch_size=64 global_batch_size=64 trainer_strategy=single_device ++trainer.precision=32-true compile=false +tags.eval_type=molgen datamodule.dataset_managers.test.unconditional_prediction.num_examples=1000 datamodule.dataset_managers.val.conditional_prediction.num_examples=1000 ++predictor.confidence=${CONFIDENCE} predictor.max_steps=1024 ++predictor.top_k=null ++predictor.top_p=${TOP_P}
```

# Train

**(Baseline) FlexMDM on Graph Traversal**
```bash
DATASET=star_hard # or star_easy, star_medium
MODEL=flexmdm
EXPERIMENT=${DATASET}_${MODEL}
xlm job_name=${DATASET}_${MODEL} job_type=train experiment=${DATASET}_${MODEL} per_device_batch_size=64 trainer_strategy=single_device trainer.devices=1 trainer.num_nodes=1 ++trainer.precision=bf16-mixed compile=False 
```

**(Ours) LFlexMDM on Graph Traversal**
```bash
DATASET=star_hard # or star_easy, star_medium
MODEL=lflexmdm
CONFIDENCE=null # or prob_diff or null or top_prob
BACKBONE=separate # or shared
aux_size=medium # or tiny or xtiny
if [ "$BACKBONE" = "separate" ]; then
    if [ "$aux_size" = "medium" ]; then
        EXPERIMENT="[${DATASET}_${MODEL},b_um_1_a1]"
    else
        EXPERIMENT="[${DATASET}_${MODEL},b_um_1_a1,${aux_size}_aux]"
else
    EXPERIMENT="[${DATASET}_${MODEL}_shared,b_um_1_a1]"
fi
xlm job_name=${DATASET}_${MODEL}_${BACKBONE}_${aux_size} job_type=train experiment=${EXPERIMENT} per_device_batch_size=64 trainer_strategy=single_device trainer.devices=1 trainer.num_nodes=1 ++trainer.precision=bf16-mixed compile=False 
```

# Acknowledgments
The code uses [xlm-core](https://github.com/dhruvdcoder/xlm-core) as the rapid experiment framework and builds on [FlexMDM](https://github.com/brianlck/FlexMDM). The code for data pipeline for the graph traversal experiments is from [ILM](https://github.com/dhruvdcoder/ILM) and for the molecule generation experiments is adapted from [GenMol](https://github.com/NVIDIA-Digital-Bio/genmol).

# TODO (add citation)

