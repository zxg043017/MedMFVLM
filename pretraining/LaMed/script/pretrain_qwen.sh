#!/bin/bash

# run "accelerate config" first!

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false deepspeed --num_gpus=2 train.py --deepspeed ./deepspeed_config.json \
    --version v0 \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --model_type lamed_qwen2_5 \
    --vision_tower swin \
    --pretrain_vision_model /raid/export/wqruan/M3D/LaMed/output/CoCa-Qwen2.5-VL-7B-tao/checkpoint-600/model.safetensors \
    --tune_mm_mlp_adapter True \
    --bf16 True \
    --output_dir ./LaMed/output/LaMed-Qwen-2.5-7B-VL-instruct-pretrain-0423 \
    --num_train_epochs 50 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "steps" \
    --eval_accumulation_steps 4 \
    --eval_steps 300 \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 16 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 16 \
    --gradient_checkpointing False \
    --dataloader_pin_memory True\
    --dataloader_num_workers 8 \
    --report_to tensorboard \
    --radgraph_enable False