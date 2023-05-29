import argparse
from random import sample
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
from FaceAttr_baseline_model import FaceAttrModel, FaceAttrModelModify
import config as cfg
# import yaml
import math
import argparse

# parameter processing
parser = argparse.ArgumentParser(description='Generate embedding from image')
parser.add_argument("--mode", default='local',choices=["global", "local"], type=str)
parser.add_argument("--local_layer", default=0, type=int)
args = parser.parse_args()
print("mode: " + args.mode + ", layer: " + str(args.local_layer))

batch_size = 20
data_dir = "./FR/FR_data/"
if not 100 % batch_size == 0:
    print("batchsize must can be divided by 100!!!!")

# Load Data
data_transforms = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5])
    ])
image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','probe']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                            shuffle=False, num_workers=0) for x in ['gallery','probe']}

def generate_model():
    # create model
    device = torch.device("cuda:" + str(cfg.DEVICE_ID) if torch.cuda.is_available() else "cpu")
    model_type = "swin_transformer"
    pretrained = True
    model = FaceAttrModelModify(model_type, pretrained, cfg.selected_attrs).to(device)
    # load pretrained model
    model_state_dict_path = "/home/gpu/Documents/MachineLearningCode/FaceAttr-Analysis-master/result/model_from_2080/beta0.3_model_best.pth.tar"
    print("=> loading checkpoint '{}'".format(model_state_dict_path))

    checkpoint = torch.load(model_state_dict_path)
    pretrained_dict = checkpoint['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k:v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model = model.eval()  
    model.cuda()  
    return model

def extract_feature(model, dataloader):
    model.eval()
    for idx, sample in enumerate(dataloader):
        image, identity = sample
        image = image.cuda()
        res, identity_scores, global_contrastive_features, local_features = model(image) # local feature:512 * 7
        if args.mode == "global":
            needed_feature = identity_scores
        else:
            needed_feature = local_features[:, args.local_layer * 512: (args.local_layer + 1) * 512]
        n, dimension = needed_feature.size()
        ff = torch.FloatTensor(n,dimension).zero_().cuda()
        ff = ff + needed_feature
        if idx == 0:
            features = torch.FloatTensor(len(dataloader.dataset), ff.shape[1])
        start = idx * batch_size
        end = min((idx+1)*batch_size, len(dataloader.dataset))
        features[ start:end, :] = ff
    return features

def get_id(img_path):
    labels = []
    for path, v in img_path:
        identity = path.split('/')[-2]
        labels.append(int(identity))
    return labels

# generate labels of probe and gallery
gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['probe'].imgs
gallery_label = get_id(gallery_path)
query_label = get_id(query_path)

# load model and change it to test mode
model = generate_model()

# Extract Feature
since = time.time()
with torch.no_grad():
    gallery_feature = extract_feature(model, dataloaders['gallery'])
    probe_feature = extract_feature(model, dataloaders['probe'])
time_elapsed = time.time() - since
print('Complete in {:.0f}m {:.2f}s'.format(
            time_elapsed // 60, time_elapsed % 60))                                    

# Save to Matlab for check
result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'probe_f':probe_feature.numpy(),'probe_label':query_label}
scipy.io.savemat('./FR/FR_result.mat',result)
print("The feature result has been saved in ./FR/FR_result.mat")
