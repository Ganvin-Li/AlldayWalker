#!/usr/bin/env bash
# train_tucker_4d_continual.sh
# 4D Tucker-LoRA continual learning training script

# Basic configuration
BASE_VIDEO_FOLDER="task"  # Base path
BASE_MODEL="model_zoo/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln/"
PROMPT_VERSION="qwen_1_5"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 4D Tucker-LoRA parameters
USE_TUCKER_4D=true
TUCKER_SCENE_NUM=5
TUCKER_ENV_NUM=4
TUCKER_RANKS_4D="16,16,32,32"  # (r1, r2, r3, r4)
TUCKER_INIT_SCALE=0.02
LORA_ALPHA=32

# Continual learning parameters
CONTINUAL_LEARNING=true
NUM_TASKS=20
EWC_LAMBDA=5000
EWC_MODE="online"
USE_ORTHOGONAL_REG=true
ORTHO_REG_WEIGHT=0.01
STABILITY_WEIGHT=1.0

# Environment variables
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export WANDB_MODE=offline

# Multi-GPU communication configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-20000}

# Base run name
BASE_RUN_NAME="tucker_4d_continual_${PROMPT_VERSION}_9_14"
BASE_OUTPUT_DIR="checkpoints/${BASE_RUN_NAME}"

# Create routing configuration file
ROUTE_CONFIG_FILE="${BASE_OUTPUT_DIR}/route_config.json"
mkdir -p ${BASE_OUTPUT_DIR}

# Fisher matrix save path
FISHER_PATH="${BASE_OUTPUT_DIR}/fisher_matrices"
mkdir -p ${FISHER_PATH}

# Training loop for all tasks
for TASK_ID in $(seq 0 $((NUM_TASKS - 1))); do

    TASK_VIDEO_FOLDER="${BASE_VIDEO_FOLDER}/Task_$((TASK_ID + 1))"
    
    # Use different port for each task to avoid conflicts
    MASTER_PORT=$((MASTER_PORT_BASE + TASK_ID))
    
    # Task-specific output directory
    TASK_OUTPUT_DIR="${BASE_OUTPUT_DIR}/task_${TASK_ID}"
    
    # Task-specific run name
    RUN_NAME="${BASE_RUN_NAME}_task_${TASK_ID}"
    
    # Set model loading path
    if [ ${TASK_ID} -eq 0 ]; then
        PRETRAINED_CHECKPOINT=""
    else
        PREV_TASK_ID=$((TASK_ID - 1))
        PREV_CHECKPOINT="${BASE_OUTPUT_DIR}/task_${PREV_TASK_ID}"
        PRETRAINED_CHECKPOINT="--pretrained_checkpoint_path ${PREV_CHECKPOINT}"
        echo "Will load Tucker weights from: ${PREV_CHECKPOINT}"
    fi

    # Start training
    torchrun \
      --nnodes 1 \
      --nproc_per_node ${NUM_GPUS} \
      --rdzv_id ${RUN_NAME} \
      --rdzv_backend c10d \
      --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
      streamvln/streamvln_train.py \
        --deepspeed scripts/zero2.json \
        --model_name_or_path ${BASE_MODEL} \
        ${PRETRAINED_CHECKPOINT} \
        --use_lora true \
        --use_tucker_4d ${USE_TUCKER_4D} \
        --tucker_scene_num ${TUCKER_SCENE_NUM} \
        --tucker_env_num ${TUCKER_ENV_NUM} \
        --tucker_ranks_4d "${TUCKER_RANKS_4D}" \
        --tucker_init_scale ${TUCKER_INIT_SCALE} \
        --lora_alpha ${LORA_ALPHA} \
        --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
        --continual_learning ${CONTINUAL_LEARNING} \
        --num_tasks ${NUM_TASKS} \
        --current_task_id ${TASK_ID} \
        --ewc_lambda ${EWC_LAMBDA} \
        --ewc_mode ${EWC_MODE} \
        --use_orthogonal_reg ${USE_ORTHOGONAL_REG} \
        --ortho_reg_weight ${ORTHO_REG_WEIGHT} \
        --stability_weight ${STABILITY_WEIGHT} \
        --route_config_file ${ROUTE_CONFIG_FILE} \
        --fisher_path ${FISHER_PATH} \
        --version ${PROMPT_VERSION} \
        --video_folder "${TASK_VIDEO_FOLDER}" \
        --group_by_task False \
        --num_history 8 \
        --num_future_steps 4 \
        --num_frames 32 \
        --data_augmentation True \
        --mm_tunable_parts="mm_mlp_adapter,mm_language_model" \
        --vision_tower ${VISION_MODEL_VERSION} \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio anyres_max_9 \
        --image_grid_pinpoints "(1x1),...,(6x6)" \
        --bf16 True \
        --run_name ${RUN_NAME} \
        --output_dir ${TASK_OUTPUT_DIR} \
        --num_train_epochs 15 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 2 \
        --evaluation_strategy "no" \
        --save_strategy "no" \
        --save_total_limit 0 \
        --learning_rate 2e-5 \
        --mm_vision_tower_lr 5e-6 \
        --weight_decay 0. \
        --warmup_ratio 0.075 \
        --lr_scheduler_type "cosine_with_min_lr" \
        --lr_scheduler_kwargs '{"min_lr": 1.85e-05}' \
        --logging_steps 10 \
        --tf32 True \
        --model_max_length 32768 \
        --gradient_checkpointing True \
        --dataloader_num_workers 8 \
        --lazy_preprocess True \
        --torch_compile True \
        --torch_compile_backend "inductor" \
        --dataloader_drop_last True \
        --report_to wandb
done