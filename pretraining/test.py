import torch
from LaMed.src.model.language_model import LamedLlamaForCausalLM, LamedPhi3ForCausalLM, LamedQwen2_5ForCausalLM


model_name_or_path = "Qwen/Qwen2.5-VL-7B-Instruct"
model = LamedQwen2_5ForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.bfloat16,
            )