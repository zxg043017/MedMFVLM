from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_mm_projector
from .segmentation_module.builder import build_segmentation_module
from LaMed.src.model.loss import BCELoss, BinaryDiceLoss
from safetensors.torch import load_file
from collections import OrderedDict


class LamedMetaModel:
    def __init__(self, config):
        super(LamedMetaModel, self).__init__(config)

        self.config = config
        self.seg_enable = False

        if hasattr(config, "vision_tower"):
            self.vision_tower = build_vision_tower(config)
            self.mm_projector = build_mm_projector(config)

        if hasattr(config, "segmentation_module") and config.segmentation_module is not None:
            self.seg_enable = True
            self.seg_module = build_segmentation_module(config)

            self.seg_projector = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(config.hidden_size, config.mm_hidden_size),
                nn.Dropout(0.1),
            )

            self.dice_loss = BinaryDiceLoss()
            self.bce_loss = BCELoss()

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        return vision_tower
    
    def switch_vision_tower(self):
        # print(self.config)
        # exit()
        self.vision_tower = build_vision_tower(self.config)

    def initialize_vision_modules(self, model_args):
        self.config.image_channel = model_args.image_channel
        self.config.image_size = model_args.image_size
        self.config.patch_size = model_args.patch_size

        self.config.vision_tower = model_args.vision_tower
        self.config.vision_select_layer = model_args.vision_select_layer
        self.config.vision_select_feature = model_args.vision_select_feature

        self.config.mm_projector_type = model_args.mm_projector_type
        self.config.proj_layer_type = model_args.proj_layer_type
        self.config.proj_layer_num = model_args.proj_layer_num
        self.config.proj_pooling_type = model_args.proj_pooling_type
        self.config.proj_pooling_size = model_args.proj_pooling_size

        # vision tower
        if self.get_vision_tower() is None:
            self.vision_tower = build_vision_tower(self.config)
            # If you have a more robust vision encoder, try freezing the vision tower by requires_grad_(False)
            self.vision_tower.requires_grad_(not model_args.freeze_vision_tower)

        def filter_state_dict(pretrained_state_dict, custom_model):
            custom_state_dict = custom_model.state_dict()
            filtered_state_dict = OrderedDict()

            # for k, v in custom_state_dict.items():
            #     print(k)
            # print("===" * 20)
            # for k, v in pretrained_state_dict.items():
            #     print(k)
            # print("===" * 20)
            # exit()
            for k, v in custom_state_dict.items():
                # 去掉预训练模型中的特定前缀
                # 跟模型中的名字有关，不一定是 img_encoder
                key = 'vision_encoder.' + k
                
                if key in pretrained_state_dict and pretrained_state_dict[key].shape == v.shape:
                    filtered_state_dict[k] = v
                else:
                    print(f"Skipping parameter {k} due to mismatch or absence.")
            # for key in filtered_state_dict.keys():
            #     print(key)
            # exit()
            return filtered_state_dict

        # TODO
        if model_args.pretrain_vision_model is not None:
            # vision_model_weights = torch.load(model_args.pretrain_vision_model, map_location='cpu')
            vision_model_weights = load_file(model_args.pretrain_vision_model)
            # for name, param in self.vision_tower.vision_tower.named_parameters():
            #     print(f"{name}: {param.size()}")
            # for key in vision_model_weights.keys():
            #     print(key)
            # 遍历预训练模型的 state_dict 并根据映射字典加载匹配的参数
            filtered_state_dict = filter_state_dict(vision_model_weights, self.vision_tower.vision_tower)

            # 将匹配到的参数加载到自定义模型中
            # print(self.vision_tower.vision_tower)
            # exit()
            self.vision_tower.vision_tower.load_state_dict(filtered_state_dict, strict=True)
            # print("Parameters and their sizes:")
            
            # print(type(vision_model_weights))
            # for key in vision_model_weights.keys():
            #     print(key)
            # print(type(self.vision_tower.vision_tower.state_dict().keys())))
            # for p in self.vision_tower.vision_tower.state_dict().keys():
            #     print(p)
            # exit()
            # self.vision_tower.vision_tower.load_state_dict(vision_model_weights, strict=True)

        self.config.mm_hidden_size = self.vision_tower.hidden_size

        # mm_projector
        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_mm_projector(self.config)
        if model_args.pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'), strict=True)
        # self.mm_projector.pooling_size = 2
        # print(self.mm_projector)
        # exit()

    def initialize_seg_modules(self, model_args):
        self.config.segmentation_module = model_args.segmentation_module

        # segmentation_module
        if getattr(self, 'seg_module', None) is None:
            self.seg_module = build_segmentation_module(self.config)
            self.seg_projector = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.config.hidden_size, self.config.mm_hidden_size),
                nn.Dropout(0.1),
            )
            self.seg_enable = True

        if model_args.pretrain_seg_module is not None:
            seg_module_weights = torch.load(model_args.pretrain_seg_module, map_location='cpu')
            new_state_dict = {}
            for key, value in seg_module_weights.items():
                if key.startswith('model.text_encoder.') or key.startswith('text_encoder.'):
                    continue
                if key.startswith('model.'):
                    new_key = key[len('model.'):]
                    new_state_dict[new_key] = value
            self.seg_module.load_state_dict(new_state_dict, strict=True)

        self.dice_loss = BinaryDiceLoss()
        self.bce_loss = BCELoss()

class LamedMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        # print(f"image shape: {images.shape}")
        # print(images.dtype)
        image_features = self.get_model().get_vision_tower()(images)
        # print(f"feature shape: {image_features.shape}")
        # exit()
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_inputs_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, is_eval=False, patch_id=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        else:
            images = images.to(dtype=torch.bfloat16)
            image_features = self.encode_images(images)
            inputs_embeds = self.get_model().embed_tokens(input_ids)
            if is_eval:
                # print(image_features.shape)
                # print(inputs_embeds.shape)
                # exit()
                patch_positions = torch.where(input_ids==patch_id)
                start_idx = patch_positions[1][0]
                inputs_embeds = torch.cat((inputs_embeds[:, :start_idx + 1, :], image_features, inputs_embeds[:, (image_features.shape[1] + start_idx + 1):, :]), dim=1)
                # print(f"position: {position}")
                # exit()
                # for idx in range(len(patch_positions[1])):  # 遍历所有patch位置
                #     position = patch_positions[1][idx]
                #     feature_idx = idx % image_features.shape[1]  # 循环使用image features
                #     # print(position, feature_idx)
                #     inputs_embeds[0, position, :] = image_features[0, feature_idx, :]
                
                # print(position.shape)
                # print(inputs_embeds[0, :, 0])
                # exit()
            else:
                inputs_embeds = torch.cat(
                    (inputs_embeds[:, :1, :], image_features, inputs_embeds[:, (image_features.shape[1] + 1):, :]), dim=1)
        return None, position_ids, attention_mask, past_key_values, inputs_embeds, labels
    

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        num_new_tokens = model_args.num_new_tokens
        self.tokenizer = tokenizer
        # print(len(tokenizer))
        # TODO: modifed here
        self.resize_token_embeddings(len(tokenizer))
        # print(f"len(tokenizer): {len(tokenizer)}")
        # exit()

        if num_new_tokens > 0:
            input_embeddings = self.get_input_embeddings().weight.data
            output_embeddings = self.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
            else:
                # we add 4 new tokens
                # if new tokens need input, please train input_embeddings
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                # if new tokens need predict, please train output_embeddings
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = True

        if model_args.pretrain_mm_mlp_adapter:
            mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
            embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']

            if input_embeddings.shape == embed_tokens_weight.shape:
                input_embeddings = embed_tokens_weight
            elif embed_tokens_weight.shape[0] == num_new_tokens:
                input_embeddings[-num_new_tokens:] = embed_tokens_weight
            else:
                raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Number of new tokens: {num_new_tokens}.")