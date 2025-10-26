from typing import Optional
import transformers
from transformers import Trainer
from dataclasses import dataclass, field
from LaMed.src.dataset.multi_dataset import ITRDataset
# from LaMed.src.model.CLIP import M3DCLIP, M3DCLIPConfig
from LaMed.src.model.Swin_CLIP import M3DCLIP, M3DCLIPConfig
from transformers import BertTokenizer, AutoModelForCausalLM, AutoTokenizer
import torch
from safetensors.torch import load_file
import os
# import transformers.modeling_utils import unwrap_model
from numpy import inf
from accelerate import Accelerator, DistributedType
from torch.distributed import is_initialized, get_rank
from torch.utils.data import DataLoader, Dataset
import json
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as TF
from transformers import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_cosine_schedule_with_warmup
import torch.distributed as dist
# import torch.nn.utils.convert_parameters as convert_parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.set_per_process_memory_fraction(0.5, 0)  # 设置第一个GPU使用50%的显存
@dataclass
class ModelArguments:
    version: Optional[str] = field(default="v0")
    # language_model_name_or_path: str = field(default="/share/home/jiangmeirui/data/upload/weights/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
    language_model_name_or_path: str = field(default="google-bert/bert-base-uncased")
    # /share/home/jiangmeirui/data/M3D/bert-base-chinese
    gather_loss: bool = field(default=True, metadata={"help": "Gather all distributed batch data of multiple GPUs and calculate contrastive loss together."})
    local_loss: bool = field(default=False)

    pretrained_model: str = field(default=None)
    pretrained_optimizer: str = field(default=None)
    pretrained_scheduler: str = field(default=None)
    in_channels: int = field(default=1)
    img_size: tuple = field(default=(128, 128, 128))
    # patch_size: tuple = field(default=(16, 16, 4))

    hidden_size: int = field(default=768)
    # mlp_dim: int = field(default=3072)
    # num_layers: int = field(default=12)
    # num_heads: int = field(default=12)
    # pos_embed: str = field(default="perceptron")
    # dropout_rate: float = field(default=0.0)
    # spatial_dims: int = field(default=3)
    max_text_len: int = field(default=512)
    # vocab_size: int = field(default=30522)
    
# /share/home/jiangmeirui/data/npz_data
# /share/home/jiangmeirui/data/ori_img_seg
@dataclass
class DataArguments:
    # data_root: str = field(default="/share/home/jiangmeirui/data/npz_data", metadata={"help": "Root directory for all data."})
    # # caption data
    # cap_data_path: str = field(default="/share/home/jiangmeirui/data/M3D/Full_revise_samples_translated.json", metadata={"help": "Path to caption data."})
    # data_root: str = field(default="/raid/export/wqruan/tao", metadata={"help": "Root directory for all data."})
    # cap_data_path: str = field(default="/raid/export/wqruan/tao/tao_train.json", metadata={"help": "Path to caption data."})
    data_root: str = field(default="/raid/export/wqruan/m3d_tao", metadata={"help": "Root directory for all data."})
    cap_data_path: str = field(default="/raid/export/wqruan/m3d_tao/tao_train_clip.json", metadata={"help": "Path to caption data."})
    max_length: int = field(default=512)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)

    # ddp_backend: str = "nccl"
    # ddp_find_unused_parameters: bool = True

    # config in bash file
    fp16: bool = True
    # bf16: bool = True
    output_dir: str = "./LaMed/output/CLIP-aug"
    num_train_epochs: int = 160
    per_device_train_batch_size: int = 2 #32
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    evaluation_strategy: str = "steps"
    eval_accumulation_steps: int = 1
    eval_steps: float = 160 # 0.04
    save_strategy: str = "steps"
    save_steps: int = 1000
    save_total_limit: int = 16
    learning_rate: float = 5e-5 #1e-4
    weight_decay: float = 0.1
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    logging_steps: float = 16 # 0.001
    gradient_checkpointing: bool = False # train fast
    dataloader_pin_memory: bool = True # fast
    dataloader_num_workers: int = 8
    report_to: str = "tensorboard"
    deepspeed: str = "deepspeed_config.json"
    


def compute_metrics(eval_pred):
    preds = eval_pred.predictions
    labels = eval_pred.label_ids
    correct = (preds == labels).sum()
    total = labels.size
    acc = correct / total
    return {"accuracy": acc}

def preprocess_logits_for_metrics(logits, labels):
    preds = torch.argmax(logits, dim=-1)
    return preds

@dataclass
class DataCollator:
    def __init__(self, gather_all):
        self.gather_all = gather_all

    def __call__(self, batch: list) -> dict:
        images, texts, input_ids, attention_mask = tuple(
            [b[key] for b in batch] for key in ('image', 'text', 'input_id', 'attention_mask'))

        images = torch.cat([_.unsqueeze(0) for _ in images], dim=0)
        input_ids = torch.cat([_.unsqueeze(0) for _ in input_ids], dim=0)
        attention_mask = torch.cat([_.unsqueeze(0) for _ in attention_mask], dim=0)

        batch_size = images.shape[0]
        if self.gather_all:
            world_size = torch.distributed.get_world_size()
            batch_size *= world_size

        labels = torch.arange(batch_size, device=images.device, dtype=torch.long)

        return_dict = dict(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        return return_dict


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    tokenizer = AutoTokenizer.from_pretrained(model_args.language_model_name_or_path)
    config = M3DCLIPConfig.from_dict(vars(model_args))
    model = M3DCLIP(config).to(torch.float16)
    
    train_dataset = ITRDataset(data_args, tokenizer, mode='train')
    eval_dataset = ITRDataset(data_args, tokenizer, mode='validation')
    # print(len(train_dataset))
    # exit()

    if model_args.gather_loss and not model_args.local_loss:
        gather_all = True
    else:
        gather_all = False
    data_collator = DataCollator(gather_all)
  
    trainer = Trainer(
                        model=model,
                        args=training_args,
                        data_collator=data_collator,
                        train_dataset=train_dataset,
                        eval_dataset=eval_dataset,
                        compute_metrics=compute_metrics,
                        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                      )
    
    accelerator = Accelerator()
    trainer.accelerator = accelerator

    if model_args.pretrained_model:
        trainer.train(resume_from_checkpoint=model_args.pretrained_model)
    else:
        trainer.train()
        
    model.config.save_pretrained(os.path.join(training_args.output_dir,'model_config'), safe_serialization=False)
    # print("after config.save_pretrain")
    model.save_pretrained(os.path.join(training_args.output_dir, 'model_pretrained'), safe_serialization=False)
    tokenizer.save_pretrained(os.path.join(training_args.output_dir, 'tokenizer_pretrained'), safe_serialization=False)
    state_dict = model.state_dict()
    torch.save(state_dict, os.path.join(training_args.output_dir, 'model_params.bin'))

if __name__ == "__main__":
    main()
