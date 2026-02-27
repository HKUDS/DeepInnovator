
# set -x

which python
export HYDRA_FULL_ERROR=1
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
ulimit -n 65535
cd ./verl-main


# export VLLM_ATTENTION_BACKEND=XFORMERS

export WANDB_BASE_URL=""
export WANDB_API_KEY=""


export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" # 
# export CUDA_LAUNCH_BLOCKING=1
ray stop
rm -rf /tmp/ray/*
export RAY_num_server_call_thread=1
export HYDRA_FULL_ERROR=1
# export VLLM_ATTENTION_BACKEND=XFORMERS
export VERL_LOGGING_LEVEL=DEBUG
##

PROJECT_DIR="$(pwd)"
export VLLM_USE_V1=1

RESUME_PATH="${1:-}"

if [ -z "$RESUME_PATH" ]; then
    RESUME_PATH=null
fi


MODEL_DIR="./qwen2.5-14b-it"
PROJECT_DIR="$(pwd)"

AGENTLOOP_CONFIG_PATH="$PROJECT_DIR/recipe/DeepInnovator/config/agent.yaml"

WANDB_PROJECT_NAME="DeepInnovator"

WANDB_EXPERIMENT_NAME="DeepInnovator"
DATASET_DIR="DATASET_DIR"

TRAIN_BATCH_SIZE=16
PPO_MINI_BATCH_SIZE=4
N_ROLLOUTS=8


RewardFunctionPath="$PROJECT_DIR/recipe/DeepInnovator/reward_function.py"
RewardFunctionName="conversation_level_reward_func"
RewardConfigPath="$PROJECT_DIR/recipe/DeepInnovator/config/reward_config.yaml"
# RewardFunctionName="reward_func"




python3 -m verl.trainer.main_ppo \
    trainer.val_before_train=False \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATASET_DIR/rl_train.parquet \
    data.val_files=$DATASET_DIR/rl_validation.parquet \
    reward_model.reward_manager=DeepInnovator \
    +reward_model.reward_kwargs.reward_kwargs_path=$RewardConfigPath \
    reward_model.use_reward_loop=False \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=8192 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_DIR \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=$N_ROLLOUTS \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=5 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=6 \
    actor_rollout_ref.rollout.multi_turn.num_repeat_rollouts=3 \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=$AGENTLOOP_CONFIG_PATH \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=$WANDB_PROJECT_NAME \
    trainer.experiment_name=$WANDB_EXPERIMENT_NAME \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=3 \
    custom_reward_function.path=$RewardFunctionPath \
    custom_reward_function.name=$RewardFunctionName \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="$PROJECT_DIR/recipe/DeepInnovator/config/DeepInnovator_interaction_config.yaml" \
    trainer.resume_from_path=$RESUME_PATH 2>&1 | tee ./recipe/DeepInnovator/DeepInnovator_train.log
