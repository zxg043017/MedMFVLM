PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false accelerate launch eval_llava_med.py \
    --output_dir ./LaMed/result/LaMed-finetune-0227/eval_caption_llava_med/