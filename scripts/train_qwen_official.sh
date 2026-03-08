#!/bin/bash
# Training script for Qwen3-VL multiclass grounding using official Qwen repo

export CUDA_VISIBLE_DEVICES=0,2
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=$(shuf -i 20000-29999 -n 1)
NPROC_PER_NODE=2

MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct"
OUTPUT_DIR="/efs/user_folders/dnshalam/projects/qwen3-vl-grounding/checkpoints_multiclass"
CACHE_DIR="/home/dnshalam/.cache/huggingface"
DATASETS="lvis_multiclass%100"

cd /efs/user_folders/dnshalam/projects/qwen3-vl-grounding/qwen3-vl-official/qwen-vl-finetune

torchrun --nproc_per_node=$NPROC_PER_NODE \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         qwenvl/train/train_qwen.py \
         --model_name_or_path $MODEL_PATH \
         --tune_mm_llm True \
         --tune_mm_vision False \
         --tune_mm_mlp False \
         --dataset_use $DATASETS \
         --output_dir $OUTPUT_DIR \
         --cache_dir $CACHE_DIR \
         --bf16 \
         --per_device_train_batch_size 2 \
         --gradient_accumulation_steps 8 \
         --learning_rate 2e-7 \
         --mm_projector_lr 1e-5 \
         --vision_tower_lr 1e-6 \
         --optim adamw_torch \
         --model_max_length 4096 \
         --data_flatten True \
         --max_pixels 576\*28\*28 \
         --min_pixels 16\*28\*28 \
         --num_train_epochs 3 \
         --warmup_ratio 0.03 \
         --lr_scheduler_type cosine \
         --weight_decay 0.01 \
         --logging_steps 10 \
         --save_steps 500 \
         --save_total_limit 2 \
         --lora_enable True \
         --lora_r 8 \
         --lora_alpha 16 \
         --lora_dropout 0.0 \
         --deepspeed scripts/zero2.json
