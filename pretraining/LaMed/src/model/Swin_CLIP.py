import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig, BertModel, AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
from LaMed.src.model.multimodal_encoder.vit import ViT
from LaMed.src.model.SwinUNetrClassification import SwinUNETR_Encoder
from LaMed.src.utils.dist_utils import gather_features


class M3DCLIPConfig(PretrainedConfig):
    model_type = "m3d_clip"

    def __init__(
        self,
        language_model_name_or_path: str = "/research/d1/rshr/xgzhou/code/M3D/M3D/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        local_loss: bool = False,
        gather_loss: bool = True,
        in_channels: int = 1,
        img_size: tuple = (128, 128, 128),
        # patch_size: tuple = (16, 16, 4),
        hidden_size: int = 768,
        vicunna_hidden_size: int = 4096,
        # mlp_dim: int = 3072,
        # num_layers: int = 12,
        # num_heads: int = 12,
        # pos_embed: str = "perceptron",
        # dropout_rate: float = 0,
        # spatial_dims: int = 3,
        max_text_len: int = 128,
        vocab_size: int = 30522,
        **kwargs,
    ):
        self.language_model_name_or_path = language_model_name_or_path
        self.in_channels = in_channels
        self.img_size = img_size
        # self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.vicunna_hidden_size = vicunna_hidden_size
        # self.mlp_dim = mlp_dim
        # self.num_layers = num_layers
        # self.num_heads = num_heads
        # self.pos_embed = pos_embed
        # self.dropout_rate = dropout_rate
        # self.spatial_dims = spatial_dims
        self.local_loss = local_loss
        self.gather_loss = gather_loss
        self.max_text_len = max_text_len
        self.vocab_size = vocab_size
        super().__init__(**kwargs)




class M3DCLIP(PreTrainedModel):
    config_class = M3DCLIPConfig

    def __init__(self, config):
        super().__init__(config)

        self.vision_encoder = SwinUNETR_Encoder(img_size=config.img_size, in_channels=config.in_channels, out_channels=4, feature_size=48)
        self.GAP = nn.Sequential(
                nn.GroupNorm(16, 768), #TBD
                nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool3d((1,1,1)),
                # nn.Conv3d(768, 256, kernel_size=1, stride=1, padding=0),
                nn.Flatten()
        )

        # self.language_encoder = BertModel.from_pretrained(config.language_model_name_or_path)
        # self.language_encoder = AutoModelForCausalLM.from_pretrained(config.language_model_name_or_path)
        self.language_encoder = AutoModel.from_pretrained(config.language_model_name_or_path)
        # self.language_encoder = AutoModel.from_pretrained("google-bert/bert-base-uncased")
        self.mm_vision_proj = nn.Linear(config.hidden_size, config.hidden_size)
        #self.mm_language_proj = nn.Linear(config.vicunna_hidden_size, config.hidden_size)
        self.mm_language_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.local_loss = config.local_loss
        self.gather_loss = config.gather_loss

    def encode_image(self, image):
        # print("In model:", image.dtype)
        # print("image", image.shape)
        
        image_feats = self.vision_encoder(image) # ViT [1, 2049, 768] SwinUNetr [1, 768, 3, 3, 3]
        # print(self.vision_encoder)
        print(1, image_feats.shape)
        # exit()
        image_feats = self.GAP(image_feats)
        print(2, image_feats.shape)
        image_feats = self.mm_vision_proj(image_feats)
        print(3, image_feats.shape)
        image_feats = F.normalize(image_feats, dim=-1)
        print(4, image_feats.shape)
        exit()
        return image_feats # [1, 768]

    def encode_text(self, input_id, attention_mask):
        text_feats = self.language_encoder(input_id, attention_mask=attention_mask)["last_hidden_state"]
        # print(text_feats.shape)
        # exit()
        text_feats = self.mm_language_proj(text_feats)
        text_feats = F.normalize(text_feats, dim=-1)

        return text_feats
    
    def load_params(self, model_dict):
        store_dict = self.vision_encoder.state_dict()
        for key in model_dict.keys():
            if "swinViT" in key or "encoder" in key:
                store_dict[key.replace("module.backbone.", "")] = model_dict[key]
        self.vision_encoder.load_state_dict(store_dict)
        
        print('Use pretrained weights')


    def forward(self, images, input_ids, attention_mask, labels, **kwargs):
        #print("zxg:images",images.shape)
        #print("zxg:input_ids",input_ids.shape)
        #print("zxg:attention_mask",attention_mask.shape)
        # image_features = self.encode_image(images)[:, 0]
        image_features = self.encode_image(images)
        text_features = self.encode_text(input_ids, attention_mask)[:, 0]


        if self.gather_loss:
            all_image_features, all_text_features = gather_features(image_features, text_features)
            if self.local_loss:
                logits_per_image = self.logit_scale * image_features @ all_text_features.T
                logits_per_text = self.logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = self.logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = self.logit_scale * image_features @ text_features.T
            logits_per_text = self.logit_scale * text_features @ image_features.T

        loss = (
                           F.cross_entropy(logits_per_image, labels) +
                           F.cross_entropy(logits_per_text, labels)
                   ) / 2

        ret = {
            "loss": loss,
            "logits": (logits_per_image + logits_per_text) / 2.0,
        }

        return ret

AutoConfig.register("m3d_clip", M3DCLIPConfig)
AutoModel.register(M3DCLIPConfig, M3DCLIP)


