import os
from dotenv import load_dotenv
import posixpath
# This looks for the .env file and loads it into os.environ
load_dotenv()
from mrunner.helpers.specification_helper import create_experiments_helper

name = globals()["script"][:-3]

scratch = "/net/scratch/hscra/plgrid/plgmikolajg"

# params for all exps
model_name = "Mohxx/helios-1.5B-sft" # or "Qwen/Qwen2.5-1.5B-Instruct"
micro_batch_size = 16

config = {
    # --- DATA & BATCH SIZES ---
    "entry_point": "verl.trainer.main_ppo",
    "data.max_prompt_length": 3000,
    "data.max_response_length": 512,
    "data.train_batch_size": 512,
    "actor_rollout_ref.actor.ppo_mini_batch_size": 128,
    
    # Micro batch sizes (Used in 4 places in your sbatch script)
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": micro_batch_size,
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": micro_batch_size,
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": micro_batch_size,
    "critic.ppo_micro_batch_size_per_gpu": micro_batch_size,

    # --- MODEL PATHS ---
    "actor_rollout_ref.model.path": model_name,
    "critic.model.path": model_name,

    # --- INFRASTRUCTURE ---
    "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
    "actor_rollout_ref.rollout.gpu_memory_utilization": 0.7,
    "trainer.n_gpus_per_node": 1,      # Matching your mrunner gpu:1 setting
    "trainer.n_cpus_per_node": 16,     # Matching your mrunner cpu:16 setting
    "trainer.nnodes": 1,
    
    # --- ALGORITHM ---
    "algorithm.step_gamma": 0.99,
    "algorithm.step_lam": 0.95,
    "algorithm.use_kl_in_reward": True,
    "algorithm.kl_ctrl.kl_coef": 0.001,
    
    # --- CRITIC ---
    "critic.highlight_first": True,
    "critic.highlight_ratio": 3.0,

    # --- ENVIRONMENT ---
    "envs.n_rollouts": 16,
    "envs.env_name": "crafter",
    "envs.task": "default",
    "envs.format_penalty": 0.1,
    "envs.binary_reward": False,
    "envs.captioner.type": "cot",
    "envs.captioner.max_text_history": 1,
    "envs.captioner.max_cot_history": 1,
    
    # --- TRAINER ---
    "trainer.project_name": "plan-crl",
    "trainer.experiment_name": name,
    "trainer.val_before_train": True,
    "trainer.critic_warmup": 20,
    "trainer.critic_warmup_step": 20,
    "trainer.logger": ["console", "wandb"],
    "trainer.save_freq": 20,
    "trainer.test_freq": 5,
    "trainer.render": False,
    "trainer.total_epochs": 100,

    # --- DISABLE UNUSED COMPONENTS ---
    "support_model.enable": False,
    "support_model.judge.enable": False,
    "reward_model.enable": False,
    "support_model.planner.enable": False,
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
        "HF_HOME": posixpath.join(scratch,"huggingface"),
        "WANDB_ENTITY":"ideas-ncbr",
        "WANDB_MODE": "offline",
    },
    base_config=config,
    params_grid=params_grid,
    mrunner_ignore=".mrunnerignore",
)
