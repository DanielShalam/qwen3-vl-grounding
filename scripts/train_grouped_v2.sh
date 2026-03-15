#!/bin/bash
# Train Qwen3-VL on grouped single-class grounding data (v2)

MASTER_ADDR="127.0.0.1"
MASTER_PORT=$(shuf -i 20000-29999 -n 1)
NPROC_PER_NODE=4

MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct"
OUTPUT_DIR="./checkpoints_grouped_v2"
CACHE_DIR="./cache"
DATASETS="lvis_grouped_v2"

cd /efs/user_folders/dnshalam/projects/qwen3-vl-grounding

/efs/user_folders/dnshalam/envs/qwen3-vl/bin/torchrun --nproc_per_node=$NPROC_PER_NODE \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         qwen3-vl-official/qwen-vl-finetune/qwenvl/train/train_qwen.py \
         --model_name_or_path $MODEL_PATH \
         --dataset_use $DATASETS \
         --tune_mm_llm True \
         --tune_mm_vision False \
         --tune_mm_mlp False \
         --output_dir $OUTPUT_DIR \
         --cache_dir $CACHE_DIR \
         --bf16 \
         --per_device_train_batch_size 2 \
         --gradient_accumulation_steps 8 \
         --learning_rate 1e-6 \
         --optim adamw_torch \
         --model_max_length 4096 \
         --data_flatten True \
         --max_pixels $((576*28*28)) \
         --min_pixels $((16*28*28)) \
         --num_train_epochs 3 \
         --warmup_ratio 0.03 \
         --lr_scheduler_type "cosine" \
         --weight_decay 0.01 \
         --logging_steps 10 \
         --save_steps 1000 \
         --save_total_limit 3 \
         --lora_enable True \
         --lora_r 8 \
         --lora_alpha 16 \
         --lora_dropout 0.0 \
         --deepspeed qwen3-vl-official/qwen-vl-finetune/scripts/zero3.json
