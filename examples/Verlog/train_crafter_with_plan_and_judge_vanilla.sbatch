#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=36:29:58
#SBATCH --gres=gpu:3
#SBATCH --account=plgbro4ppo-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=slurm/train_crafter_with_plan_and_judge_vanilla-%j.out


ml purge
ml ML-bundle/24.06a
ml GCCcore/13.3.0
#ml CUDA/12.8.0
ml CUDA/12.8.0

source .venv4/bin/activate
# pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
# pip install vllm==0.12.0
# pip install triton==3.5.0
export HYDRA_FULL_ERROR=1
#pip install bitsandbytes
#pip install unsloth
python3 -c "import torch; print(f'Torch: {torch.__version__}'); import flash_attn; print('Flash Attention load success!')"
#pip install flashinfer-python

pip list

set -e
export WANDB_ENTITY="ideas-ncbr"
export WANDB_MODE="offline"
model=Qwen/Qwen2.5-1.5B-Instruct #Mohxx/helios-1.5B-sft # Qwen/Qwen2.5-1.5B-Instruct
micro_batch_size=16
experiment_name="train_crafter_with_plan_and_judge_1.5B_3"
#export DISABLE_FLASH_ATTN=1
#reward_model.strategy="bridge" \
#CUDA_LAUNCH_BLOCKING=1
PYTHONUNBUFFERED=1  python3 -m verl.trainer.main_ppo \
 data.max_prompt_length=3000  \
 data.max_response_length=512  \
 data.train_batch_size=512 \
 actor_rollout_ref.actor.ppo_mini_batch_size=128  \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$micro_batch_size \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$micro_batch_size \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$micro_batch_size \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
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
 trainer.project_name="plan-crl" \
 trainer.experiment_name=$experiment_name \
 trainer.val_before_train=True \
 trainer.critic_warmup=20 \
 trainer.critic_warmup_step=20 \
 trainer.n_gpus_per_node=$SLURM_GPUS_ON_NODE \
 trainer.n_cpus_per_node=28 \
 trainer.nnodes=1 \
 trainer.logger=["console","wandb"]\
 trainer.save_freq=20 \
 trainer.test_freq=5 \
 trainer.render=False \
 support_model.judge.reward_scale=1.0 \
 support_model.judge.enable=True \
 support_model.enable=True \
 reward_model.enable=True \
 reward_model.strategy="bridge" \
 reward_model.reward_manager="bridge" \
 support_model.planner.enable=True \
 trainer.total_epochs=100 2>&1 | tee verl_demo.log