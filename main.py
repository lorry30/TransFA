from solver import Solver
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import random
import numpy as np
import pandas as pd
import argparse
from utils import seed_everything
import config as cfg
import shutil
import argparse
# command connect npc: ./npc -server=120.26.57.192:7000 -vkey=15014459253Hwj.. -type=tcp
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
# weight hyper parameters
parser.add_argument("--weight_alpha", default=cfg.weight_global_identity, type=float)
parser.add_argument("--weight_global_contrastive_identity", default=cfg.weight_global_contrastive_identity, type=float)
parser.add_argument("--weight_beta", default=cfg.weight_local_identity, type=float)
parser.add_argument("--weight_lamda", default=cfg.weight_attribute, type=float)
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
# update to cfg file
cfg.epochs = args.epochs
cfg.batch_size = args.batch_size
cfg.learning_rate_large = args.learning_rate
cfg.learning_rate_small = args.learning_rate
cfg.model_type = args.model_type
cfg.optim_type = args.optim_type
cfg.momentum = args.momentum
cfg.pretrained = args.pretrained
cfg.loss_type = args.loss_type
cfg.exp_version = args.exp_version
cfg.load_model_path = args.load_model_path
# weight parameters
cfg.weight_global_identity = args.weight_alpha
cfg.weight_global_contrastive_identity = args.weight_global_contrastive_identity
cfg.weight_local_identity = args.weight_beta
cfg.weight_attribute = args.weight_lamda

weight = [cfg.weight_global_identity, cfg.weight_global_contrastive_identity, cfg.weight_local_identity, cfg.weight_attribute]
print("alpha(weight_global_identity,LossF): " + str(weight[0]) + ", 1 - alpha(weight_global_contrastive_identity,LossC): " + str(weight[1]) + ", beta(weight_local_identity,Lossg): " + str(weight[2]) + ", lamda(weight_attribute,LossA): " + str(weight[3]))

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
    # torch.autograd.set_detect_anomaly(True)
    seed_everything()
    # # too more params to send.... not a good way....use the config.py to improve it
    solver = Solver(epoches=epochs, batch_size=batch_size, learning_rate=learning_rate, model_type=model_type,
                    optim_type=optim_type, momentum=momentum, pretrained=pretrained, loss_type=loss_type,
                    exp_version=exp_version, weight = weight)
    try:
        # # train the model, if the model_path is not none, continue the train
        copy_file()
        solver.fit(model_path=model_path)
    except InterruptedError:
        print("early stop...")
        print("save the model dict....")
        solver.save_model_dict(exp_version+"_"+model_path + "_earlystop.pth")
