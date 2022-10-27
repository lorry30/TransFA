from __future__ import print_function, division
import torch
import torch.nn as nn
from torchvision import transforms, models
# we can add our model into the backbone then import them
# from backbone.GC_resnet import *
from backbone.transformer_vision import *
from backbone.SE_resnet import * 
from backbone.resnet_sge import * 
from backbone.resnet_sk import * 
from backbone.shuffle_netv2 import *
from backbone.resnet_cbam import *
from backbone.vgg_net import *
from backbone.inception_resnet import *
import config as cfg

# you can add more models as you need.
__SUPPORT_MODEL__ = ["Resnet18", "Resnet101", "densenet121", "se_resnet101", "se_resnet50", "Resnet50", 
"vision_transformer", "vision_transformer_dmtl", "swin_transformer", "vgg16", "inception_resnet_v2", "inception_resnet_v1"]

"""
Adopt the pretrained resnet model to extract feature of the feature
"""
class FeatureExtraction(nn.Module):
    def __init__(self, pretrained, model_type = "Resnet18"):
        super(FeatureExtraction, self).__init__()
        # 从torchvision model中下载预训练模型
        if model_type == "Resnet18":
            self.model = models.resnet18(pretrained=pretrained)   
        if model_type == "Resnet101":
            self.model = models.resnet101(pretrained=pretrained)
        elif model_type == "Resnet152":
            self.model = models.resnet152(pretrained=pretrained)
        elif model_type == "Resnet50":
            self.model = models.resnet50(pretrained=pretrained)
        elif model_type == "densenet121":
            self.model = models.densenet121(pretrained=pretrained)
        elif model_type == "gc_resnet101":
            self.model = gc_resnet101(2)
        elif model_type == "gc_resnet50":
            self.model = gc_resnet50(2, pretrained=pretrained)
        elif model_type == 'se_resnet101':
            self.model = se_resnet101(2)
        elif model_type == "se_resnet50":
            self.model = se_resnet50(2, pretrained=pretrained)
        elif model_type == 'sge_resnet101':
            self.model = sge_resnet101(pretrained=pretrained)
        elif model_type == "sge_resnet50":
            self.model = sge_resnet50(pretrained=pretrained)
        elif model_type == "sk_resnet101":
            self.model = sk_resnet101(pretrained=pretrained)
        elif model_type == "sk_resnet50":
            self.model = sk_resnet50(pretrained=pretrained)
        elif model_type == "shuffle_netv2":
            self.model = shufflenetv2_1x(pretrained=pretrained)
        elif model_type == "cbam_resnet101":
            self.model = cbam_resnet101(pretrained=pretrained)
        elif model_type == "cbam_resnet50":
            self.model = cbam_resnet50(pretrained=pretrained)
        elif model_type == "vision_transformer" or model_type == "vision_transformer_dmtl":
            self.model = vision_transformer()
        elif model_type == "swin_transformer":
            self.model = swin_transformer()
        elif model_type == "vgg16":
            self.model = vgg16()
        elif model_type == "inception_resnet_v2":
            self.model = inception_resnet_v2()
        elif model_type == "inception_resnet_v1":
            self.model = inception_resnet_v1()
        self.model = nn.Sequential(*list(self.model.children())[:-1])
    def forward(self, image):
        return self.model(image)

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

class FeatureClassfierModifyV5(nn.Module):
    def __init__(self, selected_attrs,input_dim=512, output_dim = 1):
        super(FeatureClassfierModifyV5, self).__init__()
        self.attrs_num = len(selected_attrs)
        # local loss: 7 subnetwork
        self.subnetwork = 7 # our method
        self.attribute_number_subnetwork = [9, 10, 5, 2, 9, 3, 2] # our method
        self.accumulate_number_subnetwork = [0, 9, 19, 24, 26, 35, 38, 40]
        self.subnetwork_hidden_layer1 = 2048
        self.subnetwork_hidden_layer2 = 512
        
        # self.subnetwork category
        for i in range(self.subnetwork):
            setattr(self, 'subnetwork1_' + str(i).zfill(2), nn.Sequential(
                nn.Linear(input_dim, self.subnetwork_hidden_layer1), 
                ))
        for i in range(self.subnetwork):
            setattr(self, 'subnetwork2_' + str(i).zfill(2), nn.Sequential(
                nn.ReLU(False), 
                nn.BatchNorm1d(self.subnetwork_hidden_layer1),
                nn.Dropout(p=0.5),
                nn.Linear(self.subnetwork_hidden_layer1, self.subnetwork_hidden_layer2),
                nn.ReLU(True),
                nn.BatchNorm1d(self.subnetwork_hidden_layer2),
                nn.Dropout(p=0.5),
                nn.Linear(self.subnetwork_hidden_layer2, self.attribute_number_subnetwork[i])))
        # sigmoid
        self.sigmoid = nn.Sigmoid()

        # caculate global loss, global identity loss component
        self.identity_network = nn.Sequential(
                nn.Linear(self.subnetwork_hidden_layer1 * self.subnetwork, 4096), 
                nn.ReLU(True), 
                nn.BatchNorm1d(4096),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 2048),
                nn.ReLU(True),
                nn.BatchNorm1d(2048),
                nn.Dropout(p=0.5),
                nn.Linear(2048, cfg.train_identity_num),
        )
        # global contrastive loss
        self.global_contrastive_network = nn.Sequential(
                nn.Linear(self.subnetwork_hidden_layer1 * self.subnetwork, 4096), 
                nn.ReLU(True), 
                nn.BatchNorm1d(4096),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 2048))

        # local loss
        for i in range(self.subnetwork):
            setattr(self, 'local_feature_network' + str(i).zfill(2), nn.Sequential(
                nn.Linear(self.subnetwork_hidden_layer1, 1024),
                nn.ReLU(False), 
                nn.BatchNorm1d(1024),
                nn.Dropout(p=0.5),
                nn.Linear(1024, 512)))

    def forward(self, x):
        # x is the output of the swin_transformer, feature_map is the output of one of it's layer
        x = x.view(x.size(0), -1)  # flatten   
        y = torch.tensor([]).cuda()
        global_feature = torch.tensor([]).cuda()
        local_features = torch.tensor([]).cuda()
        for i in range(self.subnetwork):
            subnetwork1 = getattr(self, 'subnetwork1_' + str(i).zfill(2))
            subnetwork2 = getattr(self, 'subnetwork2_' + str(i).zfill(2))
            local_feature_network = getattr(self, 'local_feature_network' + str(i).zfill(2))
            temp_feature = subnetwork1(x) # [64, 2048]
            local_feature = local_feature_network(temp_feature) # [64, 512]
            local_features = torch.cat((local_features, local_feature),-1)
            global_feature = torch.cat((global_feature, temp_feature), -1)
            y = torch.cat((y, subnetwork2(temp_feature)), -1)
        res = self.sigmoid(y)
        identity_scores = self.identity_network(global_feature) # global identity score
        global_contrastive_features = self.global_contrastive_network(global_feature)# global contrastive feature
        return res, identity_scores, global_contrastive_features, local_features

"""
conbime the extraction and classfier
"""
class FaceAttrModelModify(nn.Module):
    def __init__(self, model_type, pretrained, selected_attrs):
        super(FaceAttrModelModify, self).__init__()
        # decide whether we can build the model
        assert model_type in __SUPPORT_MODEL__
        # featuer extraction
        self.featureExtractor = FeatureExtraction(pretrained, model_type)
        self.featureClassfier = FeatureClassfierModifyV5(selected_attrs, input_dim=768)
    
    def forward(self, image):
        features = self.featureExtractor(image)
        results = self.featureClassfier(features)
        return results