#!/usr/bin/env bash
# 4D Tucker-LoRA inference script
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export CUDA_VISIBLE_DEVICES=0,1

MASTER_PORT=$((RANDOM % 101 + 20000))

# Base model path
BASE_MODEL="/data3/lg/StreamVLN/model_zoo/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln"

# Tucker-LoRA model path
TUCKER_LORA_PATH="/data3/lg/StreamVLN/checkpoints/tucker_4d_continual_qwen_1_5_9_14/task_19"

# Optional: specify Tucker-LoRA weight directory
TUCKER_DIR="${TUCKER_LORA_PATH}/tucker_lora"

# Config yaml file
HABITAT_CONFIG_PATH="config/vln_r2r_night_scene.yaml" 

# ========== Hard routing configuration ==========
# Enable hard routing (set to true to enable)
USE_HARD_ROUTING=true

# Scene1
# Manually specify scene and environment indices
SCENE_IDX=0
ENV_IDX=1

# ========== Build routing arguments ==========
ROUTING_ARGS=""
if [ "$USE_HARD_ROUTING" = true ]; then
    ROUTING_ARGS="--use_hard_routing"
    
    if [ ! -z "$TASK_ID" ]; then
        # Use task ID for automatic routing
        ROUTING_ARGS="${ROUTING_ARGS} --auto_route_by_task $TASK_ID"
        echo "Using auto-routing for Task $TASK_ID"
    else
        # Use manually specified routing
        ROUTING_ARGS="${ROUTING_ARGS} --scene_idx $SCENE_IDX --env_idx $ENV_IDX"
        echo "Using manual routing: Scene $SCENE_IDX, Environment $ENV_IDX"
    fi
else
    echo "Hard routing disabled - using all Tucker parameters"
fi

# ========== Run evaluation ==========
torchrun \
  --nproc_per_node=2 \
  --master_port=$MASTER_PORT \
  streamvln/streamvln_eval.py \
  --model_path $TUCKER_LORA_PATH \
  --base_model_path $BASE_MODEL \
  --eval_split 'val_seen_5LpN3gDmAk7' \
  --output_path "./results" \
  --tucker_dir $TUCKER_DIR \
  --habitat_config_path $HABITAT_CONFIG_PATH \
  --save_video \
  $ROUTING_ARGS