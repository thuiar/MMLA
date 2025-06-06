#!/bin/bash

# Environment Variables
ARG_WORLD_SIZE=${1:-1}
ARG_NPROC_PER_NODE=${2:-4}
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16666
ARG_RANK=0

# Multiple conditions
if [ ! -n "$WORLD_SIZE" ] || [ ! -n "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
if [ ! -n "$MASTER_ADDR" ] || [ ! -n "$MASTER_PORT" ] || [ ! -n "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
    RANK=$ARG_RANK
fi

echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"

# Training Arguments
GLOBAL_BATCH_SIZE=256
LOCAL_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=$[$GLOBAL_BATCH_SIZE/($WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE)]

# Log Arguments
export TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=videollama2
RUN_NAME=vllava_settings
DATA_DIR=test_data
OUTP_DIR=work_dirs

torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE  \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank $RANK \
    videollama2/train_flash_attn.py \
    --deepspeed scripts/zero3.json \
    --model_type videollama2 \
    --model_path /home/sharing/disk1/llm_checkpoints/Mistral-7B-Instruct-v0.2 \
    --vision_tower /home/sharing/disk1/llm_checkpoints/openaiclip-vit-large-patch14-336 \
    --mm_projector_type stc_connector \
    --tune_mm_mlp_adapter True \
    --data_path   ${DATA_DIR}/custom.json \
    --data_folder ${DATA_DIR} \
    --mm_vision_select_layer -2 \
    --num_frames 8 \
    --bf16 False \
    --tf32 False \
    --fp16 True \
    --output_dir ${OUTP_DIR}/${WANDB_PROJECT}/pretrain_${RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 99 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --run_name $RUN_NAME \
