#!/bin/bash

# run "accelerate config" first!
# 50 epoch / 10h

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True accelerate launch train_CLIP.py \
    --language_model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --version v0 \
    --local_loss False \
    --gather_loss True \
    --output_dir ./LaMed/output/CLIP-aug-qwen2.5-test \
    --num_train_epochs 50 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "steps" \
    --eval_accumulation_steps 8 \
    --eval_steps 1 \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 16 \
    --learning_rate 5e-5 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 16 \
    --gradient_checkpointing False \
    --dataloader_pin_memory True\
    --dataloader_num_workers 4 \
    --report_to tensorboard