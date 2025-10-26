#!/bin/bash

# run "accelerate config" first!

# python eval_caption.py \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false accelerate launch eval_caption.py \
    --output_dir ./LaMed/result/LaMed-finetune-0426/eval_caption/ \
    --tokenizer_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --model_name_or_path ./LaMed/output/LaMed-Qwen2.5-VL-finetune-0426 \
    --cap_data_path /raid/export/wqruan/mri_data/tao_train_llm_test.json
    # --cap_data_path /raid/export/wqruan/mri_data/tao_train_llm.json