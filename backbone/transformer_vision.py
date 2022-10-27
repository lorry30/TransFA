import math
import os.path
from timm import models
import torch
import torch.nn as nn
import timm
__all__ = ['vision_transformer', 'swin_transformer']

class fc_block(nn.Module):
    def __init__(self, inplanes, planes, drop_rate=0.15):
        super(fc_block, self).__init__()
        self.fc = nn.Linear(inplanes, planes)
        self.bn = nn.BatchNorm1d(planes)
        if drop_rate > 0:
            self.dropout = nn.Dropout(drop_rate)
        self.relu = nn.ReLU(inplace=True)
        self.drop_rate = drop_rate

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        if self.drop_rate > 0:
            x = self.dropout(x)
        x = self.relu(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, num_attributes=40):
        super(VisionTransformer, self).__init__()
        dim = 1
        num_classes = 1000
        """
        put the ViT into the sequential
        output of self.vit is feature with size[1, 768]
        """
        self.vit = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.vit(x)
        x = self.mlp_head(x)
        return x

class SwinTransformer(nn.Module):
    def __init__(self, num_attributes=40):
        super(SwinTransformer, self).__init__()

        """
        put the Swin_ViT into the sequential
        output of self.vit is feature with size[1, 1024]
        """
        self.swin_transformer = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0)
        # it is useless, this block will be discarded when generating the model
        dim = 1
        num_classes = 1000
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.swin_transformer(x)
        x = self.mlp_head(x)
        return x

class SwinTransformerModifyV1(nn.Module):
    def __init__(self, num_attributes=40):
        super(SwinTransformerModifyV1, self).__init__()
        self.layer01, self.layer02 = restruct(3)
        dim = 1
        num_classes = 10
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):    
        feature_map = self.layer01(x)      
        x = self.layer02(feature_map)      
        x = self.mlp_head(x)
        return x, feature_map

def vision_transformer(**kwargs):
    """
    Constructs a vision transformer model
    """
    return VisionTransformer(**kwargs)

def swin_transformer(**kwargs):
    """
    Constructs a vision transformer model
    """
    return SwinTransformer(**kwargs)

def restruct(feature_layer):
    swin_transformer = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0)
    before_layer = nn.Sequential() 
    after_layer = nn.Sequential()
    flag = False 
    for name, module in swin_transformer.named_children():
        if name == "layers":
            i = 0
            layers = module
            for name, module in layers.named_children():
                if i <= feature_layer:
                    before_layer.add_module(name, module)
                else:
                    after_layer.add_module(name, module)
                i += 1
            flag = True
        elif not flag:
            before_layer.add_module(name, module)
        else:
            after_layer.add_module(name, module)
    return before_layer, after_layer

if __name__ == '__main__':
    swin_transformer = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0)
    print(swin_transformer)