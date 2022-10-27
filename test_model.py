from solver import Solver
import numpy as np
import pandas as pd
import argparse
from utils import seed_everything
import config as cfg
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
parser.add_argument('--pretrained', action='store_true', default=cfg.pretrained)
parser.add_argument("--load_model_path", default=cfg.load_model_path, type=str)
parser.add_argument("--image_dir", default=cfg.image_dir, type=str)
args = parser.parse_args()


batch_size = args.batch_size
model_type = args.model_type
pretrained = args.pretrained
model_path = args.load_model_path
cfg.image_dir = args.image_dir

#--------------- exe ----------------------------- #
if __name__ == "__main__":
    # train
    seed_everything()
    solver = Solver(batch_size=batch_size, model_type=model_type, pretrained=pretrained)
    solver.test_model(model_path=model_path)
