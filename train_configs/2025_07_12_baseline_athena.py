import os
from mrunner.helpers.specification_helper import create_experiments_helper

# Extract experiment name from filename
name = globals().get("script", "experiment")[:-3] if "script" in globals() else "verl_ppo_crafter"

# Base configuration (Hydra-style arguments converted to dict)
config = {
    # Model and Path settings
    "model_path": "Mohxx/helios-1.5B-sft",
    "micro_batch_size": 16,
    
    # RL Hyperparameters
    "max_prompt_length": 3000,
    "max_response_length": 512,
    "train_batch_size": 512,
    "ppo_mini_batch_size": 128,
    "kl_coef": 0.001,
    
    # Environment settings
    "env_name": "crafter",
    "n_rollouts": 16,
    "format_penalty": 0.1,
    
    # Trainer / Infrastructure
    "project_name": "plan-crl",
    "total_epochs": 100,
}

# Grid search parameters
params_grid = [
    {
        "seed": [0],
        # Example: you can sweep over batch sizes or KL coeffs here
        # "train_batch_size": [256, 512],
    },
]

# Slurm / Resource configuration
# This replaces your #SBATCH directives
slurm_config = {
    "nodes": 1,
    "ntasks-per-node": 1,
    "cpus-per-task": 32,
    "mem": "256G",
    "time": "36:29:58",
    "gres": "gpu:3",
    "partition": "plgrid-gpu-gh200",
    "account": "plgbro4ppo-gpu-gh200",
}

# Commands to run before the python script (Modules and Env)
pre_exec_cmds = [
    "ml purge",
    "ml ML-bundle/24.06a",
    "ml GCCcore/13.3.0",
    "ml CUDA/12.8.0",
    "source .venv4/bin/activate",
    "export HYDRA_FULL_ERROR=1",
    "export PYTHONUNBUFFERED=1",
    "export WANDB_ENTITY='ideas-ncbr'",
    "export WANDB_MODE='offline'",
]

experiments_list = create_experiments_helper(
    experiment_name=name,
    project_name="plan-crl",
    with_neptune=False,
    script="python3 -m verl.trainer.main_ppo", # Your main entry point
    python_path=".",
    tags=[name],
    env={
        "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
        "HF_HOME": "$SCRATCH/huggingface",
    },
    base_config=config,
    params_grid=params_grid,
    slurm_config=slurm_config,
    pre_exec=pre_exec_cmds,
)