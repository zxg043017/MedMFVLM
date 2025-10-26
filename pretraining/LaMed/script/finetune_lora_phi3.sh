#!/bin/bash

# run "accelerate config" first!


PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false accelerate launch train.py \
    --version v0 \
    --tokenizer_name_or_path microsoft/Phi-3.5-mini-instruct \
    --model_name_or_path ./LaMed/output/LaMed-Phi3.5-mini-finetune-0216-radgraph/checkpoint-4200 \
    --model_type phi3 \
    --lora_enable True \
    --vision_tower swin \
    --pretrain_vision_model /raid/export/wqruan/M3D/LaMed/output/CLIP-aug/model.safetensors \
    --pretrain_mm_mlp_adapter /raid/export/wqruan/M3D/LaMed/output/LaMed-Phi-3.5-mini-instruct-pretrain-0215/checkpoint-7000/pytorch_model.bin \
    --output_dir ./LaMed/output/LaMed-Phi3.5-mini-finetune-0306-radgraph \
    --num_train_epochs 50 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "steps" \
    --eval_accumulation_steps 8 \
    --eval_steps 160 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 100 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 16 \
    --gradient_checkpointing False \
    --dataloader_pin_memory True\
    --dataloader_num_workers 8 \
    --report_to tensorboard \
    --radgraph_enable False