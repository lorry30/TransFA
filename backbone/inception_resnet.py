import math
import os.path
import torch
import torch.nn as nn
import timm
from facenet_pytorch import InceptionResnetV1
__all__ = ['inception_resnet_v2', 'inception_resnet_v1']

class InceptionResnetV2(nn.Module):
    def __init__(self, num_attributes=40):
        super(InceptionResnetV2, self).__init__()
        """
        put the inception_resnet_v2 into the sequential
        output of self.inception_resnet_v2 is feature with size[1, 1536]
        """
        self.inception_resnet_v2  = timm.create_model('inception_resnet_v2', pretrained=True, num_classes=0)

        # it is useless, this block will be discarded when generating the model
        dim = 1
        num_classes = 1000
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        # transformer模型
        x = self.inception_resnet_v2(x)
        x = self.mlp_head(x)
        return x

class InceptionResnet(nn.Module):
    def __init__(self, num_attributes=40):
        super(InceptionResnet, self).__init__()
        """
        put the inception_resnet_v2 into the sequential
        output of self.inception_resnet_v2 is feature with size[1, 1536]
        """
        self.inception_resnet_v1  = InceptionResnetV1(pretrained='vggface2')

        # it is useless, this block will be discarded when generating the model
        dim = 1
        num_classes = 1000
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        # transformer模型
        x = self.inception_resnet_v1(x)
        x = self.mlp_head(x)
        return x

def inception_resnet_v2(**kwargs):
    """
    Constructs a inception_resnet_v2 model
    """
    return InceptionResnetV2(**kwargs)

def inception_resnet_v1(**kwargs):
    """
    Constructs a inception_resnet_v2 model
    """
    return InceptionResnet(**kwargs)

if __name__ == '__main__':
    # net = InceptionResnetV2()
    # For a model pretrained on VGGFace2
    net = InceptionResnetV1(pretrained='vggface2').eval()
    # num of data * the channel of image * width * length
    data = torch.randn(2, 3, 224, 224)
    preds = net(data)  # (1, 1000)
    print(preds.size())


    
