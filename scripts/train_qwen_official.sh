#!/bin/bash
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}

export CUDA_VISIBLE_DEVICES=2,5

cd /efs/user_folders/dnshalam/projects/qwen3-vl-grounding/qwen3-vl-official/qwen-vl-finetune

args="
--deepspeed scripts/zero3.json \
--model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
--dataset_use lvis_multiclass%100 \
--data_flatten True \
--tune_mm_vision False \
--tune_mm_mlp True \
--tune_mm_llm True \
--bf16 \
--output_dir /efs/user_folders/dnshalam/projects/qwen3-vl-grounding/checkpoints_multiclass \
--num_train_epochs 3 \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 4 \
--max_pixels 451584 \
--min_pixels 12544 \
--learning_rate 2e-7 \
--weight_decay 0 \
--warmup_ratio 0.03 \
--max_grad_norm 1 \
--lr_scheduler_type cosine \
--logging_steps 10 \
--save_steps 500 \
--save_total_limit 2 \
--model_max_length 8192 \
--gradient_checkpointing True \
--dataloader_num_workers 4 \
--lora_enable True \
--lora_r 8 \
--lora_alpha 16 \
--lora_dropout 0.0 \
--eval_strategy no \
--save_strategy steps"

torchrun --nproc_per_node=2 \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         qwenvl/train/train_qwen.py ${args}
