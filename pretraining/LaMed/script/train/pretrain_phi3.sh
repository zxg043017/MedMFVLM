#!/bin/bash

# run "accelerate config" first!

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false accelerate launch train.py \
    --version v0 \
    --model_name_or_path microsoft/Phi-3.5-mini-instruct \
    --model_type lamed_phi3 \
    --vision_tower swin \
    --pretrain_vision_model /raid/export/wqruan/M3D/LaMed/output/CLIP-aug/model.safetensors \
    --tune_mm_mlp_adapter True \
    --bf16 True \
    --output_dir ./LaMed/output/LaMed-Phi-3.5-mini-instruct-pretrain-0507-memory \
    --num_train_epochs 100 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "steps" \
    --eval_accumulation_steps 4 \
    --eval_steps 160 \
    --save_strategy "steps" \
    --save_steps 1000 \
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
    --radgraph_enable False \