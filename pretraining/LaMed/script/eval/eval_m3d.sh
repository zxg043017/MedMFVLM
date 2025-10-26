#!/bin/bash

# run "accelerate config" first!


PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false accelerate launch eval_caption.py \
    --output_dir ./LaMed/result/LaMed-finetune-0227/eval_caption_m3d/ \
    --cap_data_path /raid/export/wqruan/m3d_tao/tao_train_llm.json \
    --model_name_or_path GoodBaiBai88/M3D-LaMed-Phi-3-4B \
    --max_length 512 \
    --max_new_tokens 256 \
    --data_root /raid/export/wqruan/m3d_tao \
    --proj_out_num 256 \
    --vision_tower vit \
    --per_device_eval_batch_size 1