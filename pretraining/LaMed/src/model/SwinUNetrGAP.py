import torch
# from model.CLIP import M3DCLIP, M3DCLIPConfig
import torch.nn as nn
from model.SwinUNetrClassification import SwinUNETR_Encoder
import torch.nn.functional as F

class SwinUNertr_GAP(nn.Module):
    def __init__(self, input_size, n_class) -> None:    ## change to 3
        super(SwinUNertr_GAP, self).__init__()
        
        self.vision_encoder = SwinUNETR_Encoder(img_size=input_size, in_channels=1, out_channels=4, feature_size=48)  #.to(torch.float32)
        
        self.GAP = nn.Sequential(
                nn.GroupNorm(16, 768),
                nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool3d((1,1,1)), 
                nn.Conv3d(768, 256, kernel_size=1, stride=1, padding=0),
                nn.Flatten()
        )

        self.cls_head = nn.Sequential(
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=n_class)
        )
    
    def load_pretrain(self, model_dict):
        model_state_dict1 = self.vision_encoder.state_dict()
        store_dict1 = {}
        for key in model_dict.keys():
            if 'vision_encoder' in key and model_dict[key].size() == model_state_dict1[key.replace("vision_encoder.", "")].size():
                store_dict1[key.replace("vision_encoder.", "")] = model_dict[key]
       
        self.vision_encoder.load_state_dict(store_dict1)
        
        print("Load Weights Successfully")
    
    def forward(self, image):
        image_feats = self.vision_encoder(image)
        image_feats = self.GAP(image_feats)
        out = self.cls_head(image_feats)
        
        return out

