#!/bin/bash

# run "accelerate config" first!
# 50 epoch / 10h

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True deepspeed --num_gpus=4 train_CLIP.py --deepspeed ./deepspeed_config.json \
    --language_model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --version v0 \
    --local_loss False \
    --gather_loss True \
    --output_dir ./LaMed/output/CLIP-aug-qwen2.5-test \
    --num_train_epochs 100 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_accumulation_steps 1 \
    --eval_steps 160 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 16 \
    --learning_rate 5e-5 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 216 \
    --gradient_checkpointing False \
    --dataloader_pin_memory True\
    --dataloader_num_workers 4 \
    --report_to tensorboard