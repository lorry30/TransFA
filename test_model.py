from solver import Solver
import os
import random
import numpy as np
import torch
import pandas as pd
import argparse
from utils import seed_everything
import config as cfg
import shutil
import argparse

# parameter processing
parser = argparse.ArgumentParser(description='FaceAtrr')
parser.add_argument('--model_type', choices=[
                    'Resnet101','Resnet152','Resnet50',
                    'gc_resnet101','gc_resnet50',
                    'se_resnet101','se_resnet50', 
                    'sk_resnet101', 'sk_resnet50',
                    'sge_resnet101','sge_resnet50', 
                    "shuffle_netv2", 'densenet121',
                    "cbam_resnet101","cbam_resnet50", 
                    "vision_transformer", "vision_transformer_dmtl", "swin_transformer",
                    "vgg16", "inception_resnet_v2", "inception_resnet_v1"], 
                    default=cfg.model_type)
parser.add_argument('--batch_size', default=cfg.batch_size, type=int, help='batch_size')
parser.add_argument('--epochs', default=cfg.epochs, type=int, help='epochs')
parser.add_argument('--learning_rate', default=[cfg.learning_rate_large, cfg.learning_rate_small], type=float, help='learning_rate')
parser.add_argument('--momentum', default=cfg.momentum, type=float, help='momentum')
parser.add_argument('--optim_type', choices=['SGD','Adam'], default=cfg.optim_type)
parser.add_argument('--pretrained', action='store_true', default=cfg.pretrained)
parser.add_argument("--loss_type", choices=['BCE_loss', 'focal_loss'], default=cfg.loss_type)
parser.add_argument("--exp_version",type=str, default=cfg.exp_version)
parser.add_argument("--load_model_path", default=cfg.load_model_path, type=str)
parser.add_argument("--weight_alpha", default=cfg.weight_global_identity, type=float)
parser.add_argument("--weight_beta", default=cfg.weight_local_identity, type=float)
args = parser.parse_args()

epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.learning_rate
model_type = args.model_type
optim_type = args.optim_type
momentum = args.momentum
pretrained = args.pretrained
loss_type = args.loss_type
exp_version = args.exp_version
model_path = args.load_model_path
weight = [args.weight_alpha, 1 - args.weight_alpha, args.weight_beta]

def copy_file():
    # copy the code of the process of training 
    copy_file_list = ["main.py","config.py", "FaceAttr_baseline_model.py", "solver.py"]

    destination_dir = "./result/" + cfg.exp_version + "-" + cfg.model_type
    if not os.path.exists(destination_dir):
        os.mkdir(destination_dir)

    destination_dir = destination_dir + "/code/"
    if not os.path.exists(destination_dir):
        os.mkdir(destination_dir)
        
    for file in copy_file_list:
        shutil.copyfile(file, destination_dir + "/" + file)

#--------------- exe ----------------------------- #
if __name__ == "__main__":
    # train
    seed_everything()
    # # too more params to send.... not a good way....use the config.py to improve it
    solver = Solver(epoches=epochs, batch_size=batch_size, learning_rate=learning_rate, model_type=model_type,
                    optim_type=optim_type, momentum=momentum, pretrained=pretrained, loss_type=loss_type,
                    exp_version=exp_version, weight = weight)
    try:
        # # train the model, if the model_path is not none, continue the train
        solver.test_model(model_path=model_path)
    except InterruptedError:
        print("early stop...")
        print("save the model dict....")
        solver.save_model_dict(exp_version+"_"+model_path + "_earlystop.pth")
