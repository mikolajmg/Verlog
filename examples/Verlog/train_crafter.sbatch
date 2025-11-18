#!/bin/bash -l
#SBATCH --error=slurm/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:29:58
#SBATCH --gres=gpu:2
#SBATCH --account=plgbro4ppo-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=slurm/slurm-%j.out

# 1. Load ONLY the ML-bundle. It provides CUDA, PyTorch, etc.
# DO NOT add manual 'export' or other 'ml' commands.
ml purge
ml ML-bundle/24.06a
ml GCCcore/13.3.0
ml CUDA/12.8.0

# 2. Activate your environment
source .venv/bin/activate
#pip uninstall -y vllm
#pip3 install vllm-0.9.2.dev0+gb6553be1b.d20250704-cp311-cp311-linux_aarch64.whl
# 3. Install helper packages
#pip uninstall -y  vllm 
#pip install torch==2.8.0+cu124  --index-url https://download.pytorch.org/whl/cu124
#pip install vllm --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu124
cd ../Verlog

pip install -e --no-cache-dir .

#pip install --no-cache-dir torch==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
# python3 -c "from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis"

export HYDRA_FULL_ERROR=1



# 7. List installed packages to confirm
pip list

# ---
# Run your script
# ---

model=Qwen/Qwen2.5-0.5B-Instruct
micro_batch_size=8

export DISABLE_FLASH_ATTN=1

PYTHONUNBUFFERED=1 CUDA_LAUNCH_BLOCKING=1 python3 -m verl.trainer.main_ppo \
 data.max_prompt_length=4096 \
 data.max_response_length=512 \
 data.train_batch_size=64 \
 actor_rollout_ref.actor.ppo_mini_batch_size=32 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$micro_batch_size \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$micro_batch_size \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$micro_batch_size \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
 actor_rollout_ref.model.path=$model \
 critic.model.path=$model \
 critic.ppo_micro_batch_size_per_gpu=$micro_batch_size \
 critic.highlight_first=True \
 critic.highlight_ratio=3.0 \
 algorithm.step_gamma=0.99 \
 algorithm.step_lam=0.95 \
 algorithm.use_kl_in_reward=True \
 algorithm.kl_ctrl.kl_coef=0.001 \
 envs.n_rollouts=16 \
 envs.env_name=crafter \
 envs.task=default \
 envs.format_penalty=0.1 \
 envs.binary_reward=False \
 envs.captioner.type=cot \
 envs.captioner.max_text_history=1 \
 envs.captioner.max_cot_history=1 \
 trainer.project_name=zero \
 trainer.experiment_name=test \
 trainer.val_before_train=False \
 trainer.critic_warmup=20 \
 trainer.critic_warmup_step=20 \
 trainer.n_gpus_per_node=2 \
 trainer.n_cpus_per_node=4 \
 trainer.nnodes=1 \
 trainer.logger=["console"]\
 trainer.save_freq=1000 \
 trainer.test_freq=1000 \
 trainer.render=False \
 trainer.total_epochs=300 2>&1 | tee verl_demo.log