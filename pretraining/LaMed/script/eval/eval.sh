#!/bin/bash

# run "accelerate config" first!


PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false accelerate launch eval_caption.py \
    --output_dir ./LaMed/result/LaMed-finetune-0513/eval_caption/ \
    --cap_data_path /raid/export/wqruan/m3d_tao/tao_train_llm.json \
    --model_name_or_path ./LaMed/output/LaMed-Phi3.5-mini-finetune-0511-radgraph-memory