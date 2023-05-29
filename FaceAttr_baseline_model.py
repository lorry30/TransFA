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
        print("Has loaded the model {}".format(model_type))
        self.model = nn.Sequential(*list(self.model.children())[:-1])
    def forward(self, image):
        return self.model(image)

"""
judge the attributes from the result of feature extraction
"""
class FeatureClassfier(nn.Module):
    def __init__(self, selected_attrs,input_dim=512, output_dim = 1):
        super(FeatureClassfier, self).__init__()
        print("Classfier is FeatureClassfier")
        self.attrs_num = len(selected_attrs)
        self.selected_attrs = selected_attrs
        output_dim = len(selected_attrs)
        """build full connect layers for every attribute, """
        self.fc_set = {}

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(128, output_dim),
        )

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        res = self.fc(x)
        res = self.sigmoid(res)
        return res

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

class FeatureClassfierDMTL01(nn.Module):
    """
    This network has 8 subnetwork
    subnetwork, based on DMTL
    subnetwork1, 9att,  index: 4, 14, 15, 19, 21, 26, 27, 32, and 40
    subnetwork2, 10att, index: 6, 7, 8, 9, 10, 11, 29, 33, 34, 36
    subnetwork3, 5att,  index: 2, 3, 5, 17, 24
    subnetwork4, 2att,  index: 13, 28
    subnetwork5, 4att,  index: 20, 30, 31, 35
    subnetwork6, 5att,  index: 1, 12, 22, 23, 37
    subnetwork7, 3att,  index: 16, 18, 25
    subnetwork8, 2att,  index: 38, 39
    """
    def __init__(self, selected_attrs,input_dim=512, output_dim = 1):
        super(FeatureClassfierDMTL01, self).__init__()
        print("Classfier is FeatureClassfierDMTL01")
        self.attrs_num = len(selected_attrs)
        # 8 subnetwork
        self.subnetwork = 8
        self.attribute_number_subnetwork = [9, 10, 5, 2, 4, 5, 3, 2]
        for i in range(self.subnetwork):
            setattr(self, 'subnetwork' + str(i).zfill(2), nn.Sequential(nn.Linear(input_dim, 4096), nn.Linear(4096, 1024)))

        # clssifier has batch normalization!
        for i in range(self.attrs_num):
            setattr(self, 'classifier' + str(i).zfill(2), nn.Sequential(fc_block(1024, 256), nn.Linear(256, 1)))
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        # 128, 2048
        x = x.view(x.size(0), -1)  # flatten

        # subnetwork
        subnetwork_output = []
        for i in range(self.subnetwork):
            subnetwork = getattr(self, 'subnetwork' + str(i).zfill(2))
            subnetwork_output.append(subnetwork(x))

        y = None
        begin = 0
        for index, feature in enumerate(subnetwork_output):
            num = self.attribute_number_subnetwork[index]
            end = begin + num
            for i in range(begin, end):
                classifier = getattr(self, 'classifier' + str(i).zfill(2))
                if y == None:
                    y = classifier(feature)
                else:
                    y = torch.cat((y, classifier(feature)), 1)
            begin = end
        res = self.sigmoid(y)
        return res

class FeatureClassfierLocation01(nn.Module):
    """
    This network has 7 subnetwork
    subnetwork1, 9att,  index: 3, 11, 14, 19, 21, 26, 27, 32, 40
    subnetwork2, 10att, index: 5, 6, 9, 10, 12, 18, 29, 33, 34, 36
    subnetwork3, 5att,  index: 2, 4, 13, 16, 24
    subnetwork4, 2att,  index: 8, 28
    subnetwork5, 9att,  index: 1, 7, 15, 17, 22, 23, 25, 31, 37
    subnetwork6, 3att,  index: 20, 30, 35
    subnetwork7, 2att,  index: 38, 39
    """
    def __init__(self, selected_attrs,input_dim=512, output_dim = 1):
        super(FeatureClassfierLocation01, self).__init__()
        print("Classfier is FeatureClassfierLocation01")
        self.attrs_num = len(selected_attrs)
        # 7 subnetwork
        self.subnetwork = 7
        self.attribute_number_subnetwork = [9, 10, 5, 2, 9, 3, 2]
        for i in range(self.subnetwork):
            setattr(self, 'subnetwork' + str(i).zfill(2), nn.Sequential(nn.Linear(input_dim, 4096), nn.Linear(4096, 1024)))

        # clssifier has batch normalization!
        for i in range(self.attrs_num):
            setattr(self, 'classifier' + str(i).zfill(2), nn.Sequential(fc_block(1024, 256), nn.Linear(256, 1)))
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        # 128, 2048
        x = x.view(x.size(0), -1)  # flatten

        # subnetwork
        subnetwork_output = []
        for i in range(self.subnetwork):
            subnetwork = getattr(self, 'subnetwork' + str(i).zfill(2))
            subnetwork_output.append(subnetwork(x))

        y = None
        begin = 0
        for index, feature in enumerate(subnetwork_output):
            num = self.attribute_number_subnetwork[index]
            end = begin + num
            for i in range(begin, end):
                classifier = getattr(self, 'classifier' + str(i).zfill(2))
                if y == None:
                    y = classifier(feature)
                else:
                    y = torch.cat((y, classifier(feature)), 1)
            begin = end
        res = self.sigmoid(y)
        return res

class FeatureClassfierDMTL02(nn.Module):
    """
    This network has 8 subnetwork, 2 fc layer
    """
    def __init__(self, selected_attrs,input_dim=512, output_dim = 1):
        super(FeatureClassfierDMTL02, self).__init__()
        print("Classfier is FeatureClassfierDMTL02")
        self.attrs_num = len(selected_attrs)
        # 8 subnetwork
        self.subnetwork = 8
        self.attribute_number_subnetwork = [9, 10, 5, 2, 4, 5, 3, 2]
        for i in range(self.subnetwork):
            setattr(self, 'subnetwork' + str(i).zfill(2), nn.Sequential(nn.Linear(input_dim, 2048), nn.Linear(2048, self.attribute_number_subnetwork[i])))
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        
        y = None # subnetwork
        for i in range(self.subnetwork):
            subnetwork = getattr(self, 'subnetwork' + str(i).zfill(2))
            if y == None:
                y = subnetwork(x)
            else:
                y = torch.cat((y, subnetwork(x)), 1)

        res = self.sigmoid(y)
        return res # size[batch_size, 40]

class FeatureClassfierLocation02(nn.Module):
    """
    This network has 7 subnetwork, each subnetwork has 2 fc layer, 1 output node layer without Relu, dropout
    """
    def __init__(self, selected_attrs,input_dim=512, output_dim = 1):
        super(FeatureClassfierLocation02, self).__init__()
        print("Classfier is FeatureClassfierLocation02")
        self.attrs_num = len(selected_attrs)
        # 8 subnetwork
        self.subnetwork = 7
        self.attribute_number_subnetwork = [9, 10, 5, 2, 9, 3, 2]
        for i in range(self.subnetwork):
            setattr(self, 'subnetwork' + str(i).zfill(2), nn.Sequential(
                nn.Linear(input_dim, 2048),
                nn.Linear(2048, 512), 
                nn.Linear(512, self.attribute_number_subnetwork[i])))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        
        y = None # subnetwork
        for i in range(self.subnetwork):
            subnetwork = getattr(self, 'subnetwork' + str(i).zfill(2))
            if y == None:
                y = subnetwork(x)
            else:
                y = torch.cat((y, subnetwork(x)), 1)

        res = self.sigmoid(y)
        return res # size[batch_size, 40]

class FeatureClassfierLocation03(nn.Module):
    """
    This network has 7 subnetwork
    each subnetwork has 2 fc layer, relu activation function, dropout = 0.5, 1 output node layer
    """
    def __init__(self, selected_attrs,input_dim=512, output_dim = 1):
        super(FeatureClassfierLocation03, self).__init__()
        print("Classfier is FeatureClassfierLocation03")
        self.attrs_num = len(selected_attrs)
        # 8 subnetwork
        self.subnetwork = 7
        self.attribute_number_subnetwork = [9, 10, 5, 2, 9, 3, 2]
        for i in range(self.subnetwork):
            setattr(self, 'subnetwork' + str(i).zfill(2), nn.Sequential(
                nn.Linear(input_dim, 2048),
                nn.ReLU(True), 
                nn.Dropout(p=0.5),
                nn.Linear(2048, 512),
                nn.ReLU(True), 
                nn.Dropout(p=0.5),
                nn.Linear(512, self.attribute_number_subnetwork[i])
                ))
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        
        y = None # subnetwork
        for i in range(self.subnetwork):
            subnetwork = getattr(self, 'subnetwork' + str(i).zfill(2))
            if y == None:
                y = subnetwork(x)
            else:
                y = torch.cat((y, subnetwork(x)), 1)

        res = self.sigmoid(y)
        return res # size[batch_size, 40]

class FeatureClassfierLocation04(nn.Module):
    """
    This network has 7 subnetwork
    each subnetwork has 2 fc layer, relu activation function, dropout = 0.5, BN layer, 1 output node layer
    """
    def __init__(self, selected_attrs,input_dim=512, output_dim = 1):
        super(FeatureClassfierLocation04, self).__init__()
        print("Classfier is FeatureClassfierLocation04")
        self.attrs_num = len(selected_attrs)
        # 8 subnetwork
        self.subnetwork = 7
        self.attribute_number_subnetwork = [9, 10, 5, 2, 9, 3, 2]
        self.subnetwork_hidden_layer1 = 2048
        self.subnetwork_hidden_layer2 = 512
        for i in range(self.subnetwork):
            setattr(self, 'subnetwork' + str(i).zfill(2), nn.Sequential(
                nn.Linear(input_dim, self.subnetwork_hidden_layer1),
                nn.ReLU(True), 
                nn.BatchNorm1d(self.subnetwork_hidden_layer1),
                nn.Dropout(p=0.5),
                nn.Linear(self.subnetwork_hidden_layer1, self.subnetwork_hidden_layer2),
                nn.ReLU(True), 
                nn.BatchNorm1d(self.subnetwork_hidden_layer2),
                nn.Dropout(p=0.5),
                nn.Linear(self.subnetwork_hidden_layer2, self.attribute_number_subnetwork[i])
                ))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        
        y = None # subnetwork
        for i in range(self.subnetwork):
            subnetwork = getattr(self, 'subnetwork' + str(i).zfill(2))
            if y == None:
                y = subnetwork(x)
            else:
                y = torch.cat((y, subnetwork(x)), 1)

        res = self.sigmoid(y)
        return res # size[batch_size, 40]

class FeatureClassfierModifyV1(nn.Module):
    """
    This network has 7 subnetwork, each subnetwork has 2 fc layer(each fc layer has drop out and relu function)
    """
    def __init__(self, selected_attrs,input_dim=512, output_dim = 1):
        super(FeatureClassfierModifyV1, self).__init__()
        print("Classfier is FeatureClassfierModifyV1")
        self.attrs_num = len(selected_attrs)
        # 8 subnetwork
        self.subnetwork = 7
        self.subnetwork_hidden_layer1 = 2048
        self.subnetwork_hidden_layer2 = 512
        self.attribute_number_subnetwork = [9, 10, 5, 2, 9, 3, 2]
        for i in range(self.subnetwork):
            setattr(self, 'subnetwork1_' + str(i).zfill(2), nn.Sequential(
                nn.Linear(input_dim, self.subnetwork_hidden_layer1), 
                nn.ReLU(True),
                nn.BatchNorm1d(self.subnetwork_hidden_layer1), 
                nn.Dropout(p=0.5),
                nn.Linear(self.subnetwork_hidden_layer1, self.subnetwork_hidden_layer2),
                nn.ReLU(True),
                nn.BatchNorm1d(self.subnetwork_hidden_layer2), 
                nn.Dropout(p=0.5)))
        for i in range(self.subnetwork):
            setattr(self, 'subnetwork2_' + str(i).zfill(2), nn.Sequential(
                nn.Linear(self.subnetwork_hidden_layer2, self.attribute_number_subnetwork[i])))
        # sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x is the output of the swin_transformer, feature_map is the output of one of it's layer
        x = x.view(x.size(0), -1)  # flatten

        y = torch.tensor([]).cuda()
        global_feature = torch.tensor([]).cuda()

        for i in range(self.subnetwork):
            subnetwork1 = getattr(self, 'subnetwork1_' + str(i).zfill(2))
            subnetwork2 = getattr(self, 'subnetwork2_' + str(i).zfill(2))
            temp_feature = subnetwork1(x)
            global_feature = torch.cat((global_feature, temp_feature), -1)
            y = torch.cat((y, subnetwork2(temp_feature)), -1)

        res = self.sigmoid(y)
        return res, global_feature

class FeatureClassfierModifyV2(nn.Module):
    """
    This network has 7 subnetwork, each subnetwork has 2 fc layer
    """
    def __init__(self, selected_attrs,input_dim=512, output_dim = 1):
        super(FeatureClassfierModifyV2, self).__init__()
        print("Classfier is FeatureClassfierModifyV2")
        self.attrs_num = len(selected_attrs)
        self.subnetwork = 8 # TPAMI
        self.attribute_number_subnetwork = [9, 10, 5, 2, 4, 5, 3, 2] # TPAMI
        self.subnetwork_hidden_layer1 = 2048
        self.subnetwork_hidden_layer2 = 512
        
        # self.subnetwork category
        for i in range(self.subnetwork):
            setattr(self, 'subnetwork1_' + str(i).zfill(2), nn.Sequential(
                nn.Linear(input_dim, self.subnetwork_hidden_layer1), 
                ))
        for i in range(self.subnetwork):
            setattr(self, 'subnetwork2_' + str(i).zfill(2), nn.Sequential(
                nn.ReLU(True), 
                nn.BatchNorm1d(self.subnetwork_hidden_layer1),
                nn.Dropout(p=0.5),
                nn.Linear(self.subnetwork_hidden_layer1, self.subnetwork_hidden_layer2),
                nn.ReLU(True),
                nn.BatchNorm1d(self.subnetwork_hidden_layer2),
                nn.Dropout(p=0.5),
                nn.Linear(self.subnetwork_hidden_layer2, self.attribute_number_subnetwork[i])))
        # sigmoid
        self.sigmoid = nn.Sigmoid()

        # caculate global loss
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

    def forward(self, x):
        # x is the output of the swin_transformer, feature_map is the output of one of it's layer
        x = x.view(x.size(0), -1)  # flatten

        y = torch.tensor([]).cuda()
        global_feature = torch.tensor([]).cuda()
        for i in range(self.subnetwork):
            subnetwork1 = getattr(self, 'subnetwork1_' + str(i).zfill(2))
            subnetwork2 = getattr(self, 'subnetwork2_' + str(i).zfill(2))
            temp_feature = subnetwork1(x)
            global_feature = torch.cat((global_feature, temp_feature), -1)
            y = torch.cat((y, subnetwork2(temp_feature)), -1)

        res = self.sigmoid(y)
        identity_scores = self.identity_network(global_feature)
        return res, identity_scores

class FeatureClassfierModifyV3(nn.Module):
    """
    This network has 7 subnetwork, each subnetwork has 2 fc layer
    """
    def __init__(self, selected_attrs,input_dim=512, output_dim = 1):
        super(FeatureClassfierModifyV3, self).__init__()
        print("Classfier is FeatureClassfierModifyV3")
        self.attrs_num = len(selected_attrs)
        # local loss: 7 subnetwork
        self.subnetwork = 7 # our method
        self.attribute_number_subnetwork = [9, 10, 5, 2, 9, 3, 2] # our method
        self.subnetwork_hidden_layer1 = 2048
        self.subnetwork_hidden_layer2 = 512
        
        # self.subnetwork category
        for i in range(self.subnetwork):
            setattr(self, 'subnetwork1_' + str(i).zfill(2), nn.Sequential(
                nn.Linear(input_dim, self.subnetwork_hidden_layer1), 
                ))
        for i in range(self.subnetwork):
            setattr(self, 'subnetwork2_' + str(i).zfill(2), nn.Sequential(
                nn.ReLU(True), 
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
                nn.Linear(4096, 2048),
        )

    def forward(self, x):
        # x is the output of the swin_transformer, feature_map is the output of one of it's layer
        x = x.view(x.size(0), -1)  # flatten

        y = torch.tensor([]).cuda()
        global_feature = torch.tensor([]).cuda()
        for i in range(self.subnetwork):
            subnetwork1 = getattr(self, 'subnetwork1_' + str(i).zfill(2))
            subnetwork2 = getattr(self, 'subnetwork2_' + str(i).zfill(2))
            temp_feature = subnetwork1(x)
            global_feature = torch.cat((global_feature, temp_feature), -1)
            y = torch.cat((y, subnetwork2(temp_feature)), -1)

        res = self.sigmoid(y)
        identity_scores = self.identity_network(global_feature) # global identity score
        global_contrastive_features = self.global_contrastive_network(global_feature)# global contrastive feature
        return res, identity_scores, global_contrastive_features

class FeatureClassfierModifyV4(nn.Module):
    """
    This network has 7 subnetwork, each subnetwork has 2 fc layer
    和v3相比，将参数数量调得更小
    """
    def __init__(self, selected_attrs,input_dim=512, output_dim = 1):
        super(FeatureClassfierModifyV4, self).__init__()
        print("Classfier is FeatureClassfierModifyV4")
        self.attrs_num = len(selected_attrs)
        # local loss: 7 subnetwork
        self.subnetwork = 7 # our method
        self.attribute_number_subnetwork = [9, 10, 5, 2, 9, 3, 2] # our method
        self.subnetwork_hidden_layer1 = 512
        self.subnetwork_hidden_layer2 = 1024
        
        # self.subnetwork category
        for i in range(self.subnetwork):
            setattr(self, 'subnetwork1_' + str(i).zfill(2), nn.Sequential(
                nn.Linear(input_dim, self.subnetwork_hidden_layer1), 
                ))
        for i in range(self.subnetwork):
            setattr(self, 'subnetwork2_' + str(i).zfill(2), nn.Sequential(
                nn.ReLU(True), 
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
                nn.Linear(self.subnetwork_hidden_layer1 * self.subnetwork, 1024), 
                nn.ReLU(True), 
                nn.BatchNorm1d(1024),
                nn.Dropout(p=0.5),
                nn.Linear(1024, 512),
                nn.ReLU(True),
                nn.BatchNorm1d(512),
                nn.Dropout(p=0.5),
                nn.Linear(512, cfg.train_identity_num),
        )
        # global contrastive loss
        self.global_contrastive_network = nn.Sequential(
                nn.Linear(self.subnetwork_hidden_layer1 * self.subnetwork, 1024), 
                nn.ReLU(True), 
                nn.BatchNorm1d(1024),
                nn.Dropout(p=0.5),
                nn.Linear(1024, 512),
        )


    def forward(self, x):
        # x is the output of the swin_transformer, feature_map is the output of one of it's layer
        x = x.view(x.size(0), -1)  # flatten

        y = torch.tensor([]).cuda()
        global_feature = torch.tensor([]).cuda()
        for i in range(self.subnetwork):
            subnetwork1 = getattr(self, 'subnetwork1_' + str(i).zfill(2))
            subnetwork2 = getattr(self, 'subnetwork2_' + str(i).zfill(2))
            temp_feature = subnetwork1(x)
            global_feature = torch.cat((global_feature, temp_feature), -1)
            y = torch.cat((y, subnetwork2(temp_feature)), -1)

        res = self.sigmoid(y)
        identity_scores = self.identity_network(global_feature) # global identity score
        global_contrastive_features = self.global_contrastive_network(global_feature)# global contrastive feature
        return res, identity_scores, global_contrastive_features

class FeatureClassfierModifyV5(nn.Module):
    """
    This network has 7 subnetwork, each subnetwork has 2 fc layer
    """
    def __init__(self, selected_attrs,input_dim=512, output_dim = 1):
        super(FeatureClassfierModifyV5, self).__init__()
        print("Classfier is FeatureClassfierModifyV5")
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
        for i in range(self.subnetwork): # 7个network用于计算local feature contrastive loss
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

class FeatureClassfierFRV1(nn.Module):
    """
    This network has 7 subnetwork, each subnetwork has 2 fc layer
    V5相比V3,添加了local attribute相关的
    """
    def __init__(self, selected_attrs,input_dim=512, output_dim = 1):
        super(FeatureClassfierFRV1, self).__init__()
        print("Classfier is FeatureClassfierFRV1")
        self.attrs_num = len(selected_attrs)
        # local loss: 7 subnetwork
        self.subnetwork = 7 # our method
        self.attribute_number_subnetwork = [9, 10, 5, 2, 9, 3, 2] # our method
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
                nn.Linear(2048, cfg.train_identity_num), # cfg.train_identity_num = 8192
        )
        # global contrastive loss
        self.global_contrastive_network = nn.Sequential(
                nn.Linear(self.subnetwork_hidden_layer1 * self.subnetwork, 4096), 
                nn.ReLU(True), 
                nn.BatchNorm1d(4096),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 2048))

        # local loss
        for i in range(self.subnetwork): # 7个network用于计算local feature contrastive loss
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
            temp_feature = subnetwork1(x)
            local_feature = local_feature_network(temp_feature)
            local_features = torch.cat((local_features, local_feature), -1)
            global_feature = torch.cat((global_feature, temp_feature), -1)
            y = torch.cat((y, subnetwork2(temp_feature)), -1)

        res = self.sigmoid(y)
        identity_scores = self.identity_network(global_feature) # global identity score
        global_contrastive_features = self.global_contrastive_network(global_feature)# global contrastive feature
        # attribute result; global feature;global contrastive feature; local feature * 7
        return res, identity_scores, global_contrastive_features, local_features # 返回globa feature(8192), 7个local feature(512*7)

"""
conbime the extraction and classfier
"""
class FaceAttrModel(nn.Module):
    def __init__(self, model_type, pretrained, selected_attrs):
        super(FaceAttrModel, self).__init__()
        # decide whether we can build the model
        assert model_type in __SUPPORT_MODEL__
        # featuer extraction
        self.featureExtractor = FeatureExtraction(pretrained, model_type)
        # classifier used for attribute classification
        if model_type == "Resnet18":
            self.featureClassfier = FeatureClassfierLocation04(selected_attrs, input_dim=512)
        elif model_type == "vision_transformer_dmtl":
            self.featureClassfier = FeatureClassfierDMTL01(selected_attrs, input_dim=192)
        elif model_type == "vision_transformer":
            self.featureClassfier = FeatureClassfierLocation02(selected_attrs, input_dim=192)
        elif model_type == "swin_transformer":
            self.featureClassfier = FeatureClassfierLocation04(selected_attrs, input_dim=768)
        elif model_type == "vgg16":
            # the output of the swin_transformer is 4096
            self.featureClassfier = FeatureClassfier(selected_attrs, input_dim=4096)
        elif model_type == "inception_resnet_v2":
            # the output of the swin_transformer is 1536
            self.featureClassfier = FeatureClassfierLocation02(selected_attrs, input_dim=1536)
        elif model_type == "inception_resnet_v1":
            # the output of the swin_transformer is 1536
            self.featureClassfier = FeatureClassfierLocation02(selected_attrs, input_dim=512)
        else:# default is 2048
            self.featureClassfier = FeatureClassfierLocation02(selected_attrs, input_dim=2048)
    
    def forward(self, image):
        features = self.featureExtractor(image)
        results = self.featureClassfier(features)
        return results

class FaceAttrModelModify(nn.Module):
    def __init__(self, model_type, pretrained, selected_attrs):
        super(FaceAttrModelModify, self).__init__()
        # decide whether we can build the model
        assert model_type in __SUPPORT_MODEL__
        # featuer extraction
        self.featureExtractor = FeatureExtraction(pretrained, model_type)
        pytorch_total_params = sum(p.numel() for p in self.featureExtractor.parameters())
        print("featureExtractor: " + str(pytorch_total_params))
        # classifier used for attribute classification
        if model_type == "Resnet18":
            self.featureClassfier = FeatureClassfierModifyV3(selected_attrs, input_dim=512)
        elif model_type == "vision_transformer_dmtl":
            self.featureClassfier = FeatureClassfierModifyV2(selected_attrs, input_dim=192)
        elif model_type == "vision_transformer":
            self.featureClassfier = FeatureClassfierModifyV2(selected_attrs, input_dim=192)
        elif model_type == "swin_transformer":
            self.featureClassfier = FeatureClassfierModifyV5(selected_attrs, input_dim=768)
        elif model_type == "vgg16":
            # the output of the swin_transformer is 4096
            self.featureClassfier = FeatureClassfierModifyV2(selected_attrs, input_dim=4096)
        elif model_type == "inception_resnet_v2":
            # the output of the swin_transformer is 1536
            self.featureClassfier = FeatureClassfierModifyV2(selected_attrs, input_dim=1536)
        elif model_type == "inception_resnet_v1":
            # the output of the swin_transformer is 1536
            self.featureClassfier = FeatureClassfierModifyV2(selected_attrs, input_dim=512)
        else:# default is 2048
            self.featureClassfier = FeatureClassfierModifyV2(selected_attrs, input_dim=2048)

        pytorch_total_params = sum(p.numel() for p in self.featureClassfier.parameters())
        print("featureClassfier: " + str(pytorch_total_params))

    def forward(self, image):
        features = self.featureExtractor(image)
        results = self.featureClassfier(features)
        return results



