#!/bin/bash

# run "accelerate config" first!


PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false deepspeed --master_port 1216 --num_gpus=1 train.py \
    --version v0 \
    --model_name_or_path microsoft/Phi-3.5-mini-instruct \
    --model_type phi3 \
    --lora_enable True \
    --vision_tower swin \
    --pretrain_vision_model /raid/export/wqruan/M3D/LaMed/output/CLIP-aug/model.safetensors \
    --pretrain_mm_mlp_adapter /raid/export/wqruan/M3D/LaMed/output/LaMed-Phi3-4B-pretrain-0208/checkpoint-17000/pytorch_model.bin \
    --output_dir ./LaMed/output/LaMed-Phi3.5-mini-finetune-0213-radgraph-singlegpu \
    --num_train_epochs 50 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "steps" \
    --eval_accumulation_steps 4 \
    --eval_steps 160 \
    --save_strategy "steps" \
    --save_steps 200 \
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
