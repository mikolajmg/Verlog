import os

from mrunner.helpers.specification_helper import create_experiments_helper

name = globals()["script"][:-3]

scratch = "/net/scratch/hscra/plgrid/plgmikolajg"

# params for all exps
config = {
    # Model and Path 
    "entry_point": "verl.trainer.main_ppo",
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

experiments_list = create_experiments_helper(
    experiment_name=name,
    project_name="plan-crl",
    with_neptune=False,
    script="python3 mrunner_run.py",
    python_path=".",
    tags=[name],
    env={
        "WANDB_API_KEY": os.environ["WANDB_API_KEY"],
        #"HF_TOKEN": os.environ["HF_TOKEN"],
        "HF_HOME": "$SCRATCH/huggingface",
        "WANDB_ENTITY":"ideas-ncbr",
        "WANDB_MODE": "offline",
    },
    base_config=config,
    params_grid=params_grid,
    mrunner_ignore=".mrunnerignore",
)
