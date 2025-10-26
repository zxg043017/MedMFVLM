#!/bin/bash

# run "accelerate config" first!


PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false deepspeed --num_gpus=2 train.py --deepspeed ./deepspeed_config.json \
    --version v0 \
    --tokenizer_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --model_type qwen \
    --lora_enable True \
    --vision_tower swin \
    --pretrain_vision_model /raid/export/wqruan/M3D/LaMed/output/CoCa-Qwen2.5-VL-7B-tao/checkpoint-600/model.safetensors \
    --pretrain_mm_mlp_adapter ./LaMed/output/LaMed-Qwen-2.5-7B-VL-instruct-pretrain-0423/checkpoint-4200/pytorch_model.bin \
    --output_dir ./LaMed/output/LaMed-Qwen2.5-VL-finetune-0430 \
    --num_train_epochs 8 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "steps" \
    --eval_accumulation_steps 16 \
    --eval_steps 50 \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 16 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.2 \
    --lr_scheduler_type "cosine" \
    --logging_steps 16 \
    --gradient_checkpointing False \
    --dataloader_pin_memory True\
    --dataloader_num_workers 8 \
    --report_to tensorboard \
    --radgraph_enable False