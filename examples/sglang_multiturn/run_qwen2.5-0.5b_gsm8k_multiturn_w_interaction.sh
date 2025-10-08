#!/bin/bash
# run on 8xH100
# make sure your current working directory is the root of the project
set -x
ulimit -n 65535
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
OFFLOAD=${OFFLOAD:-False}
HF_MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"
export VLLM_USE_V1=1

# Calculate derived values
PPO_MINI_BATCH_SIZE=$((TRAIN_BATCH_SIZE / 2))
LOG_PROB_MICRO_BATCH=$((MICRO_BATCH_SIZE * 8))

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='gsm8k_multiturn_grpo_w_interaction' \
    algorithm.adv_estimator=gae \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$HF_MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    +actor_rollout_ref.model.enable_activation_offloading=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=$OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OFFLOAD \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.agent.num_workers=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH \
    actor_rollout_ref.ref.fsdp_config.param_offload=$OFFLOAD \
    critic.optim.lr=1e-5 \
    critic.model.path=$HF_MODEL_PATH \
    critic.model.enable_gradient_checkpointing=False \
    critic.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    algorithm.use_kl_in_reward=False \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='gsm8k_async_rl' \
    trainer.experiment_name='test_throughput' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    data.train_files=$HOME/data/gsm8k_verl_sgl_multi_turn_w_interaction/train.parquet \
    data.val_files=$HOME/data/gsm8k_verl_sgl_multi_turn_w_interaction/test.parquet \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/interaction_config/gsm8k_interaction_config.yaml" \
    trainer.total_epochs=15 "$@"