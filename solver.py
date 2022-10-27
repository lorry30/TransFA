from __future__ import print_function
from __future__ import division
import torch
from torchvision import transforms
import pandas as pd 
import time
import os
from CelebA import get_loader
import utils
from FaceAttr_baseline_model import FaceAttrModelModify
import config as cfg

class Solver(object):
    def __init__(self, batch_size, model_type, pretrained):
        self.identity_path = cfg.identity_path
        self.batch_size = batch_size
        self.selected_attrs = cfg.selected_attrs
        self.device = torch.device("cuda:" + str(cfg.DEVICE_ID) if torch.cuda.is_available() else "cpu")
        self.image_dir = cfg.image_dir
        self.attr_path = cfg.attr_path
        self.pretrained = pretrained
        self.model_type = model_type
        self.build_model(model_type, pretrained) 
        self.train_loader = None
        self.validate_loader = None
        self.test_loader = None
        self.log_dir = cfg.log_dir
        self.use_tensorboard = cfg.use_tensorboard
        self.attr_threshold = cfg.attr_threshold
        self.LOADED = False
        self.start_time = time.time()
        torch.cuda.set_device(cfg.DEVICE_ID)

    def build_model(self, model_type, pretrained):
        """Here should change the model's structure""" 
        self.model = FaceAttrModelModify(model_type, pretrained, self.selected_attrs).to(self.device)
        
    def set_transform(self, mode):
        transform = []
        transform.append(transforms.Resize(size=(224, 224))) # test no resize operation.
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5]))
        
        transform = transforms.Compose(transform)
        self.transform = transform

    def load_model_dict(self, model_state_dict_path):
        if os.path.isfile(model_state_dict_path):
            pretrained_dict = torch.load(model_state_dict_path)
            # pretrained_dict = checkpoint['state_dict']
            model_dict = self.model.state_dict()
            pretrained_dict = {k:v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
    
    def evaluate(self, mode):
        """
        Mode: validate or test mode
        Return: correct_dict: save the average predicting accuracy of every attribute
        """
        self.model.eval()
        self.set_transform(mode)
        data_loader = None

        # if self.validate_loader == None and mode == "validate":
        if self.validate_loader == None:
            self.validate_loader = get_loader(image_dir = self.image_dir, 
                                    attr_path = self.attr_path, 
                                    identity_path=self.identity_path,
                                    selected_attrs = self.selected_attrs,
                                    mode=mode, batch_size=self.batch_size, transform=self.transform)
        if self.test_loader == None:
            self.test_loader = get_loader(image_dir = self.image_dir, 
                                    identity_path=self.identity_path,
                                    attr_path = self.attr_path, 
                                    selected_attrs = self.selected_attrs,
                                    mode=mode, batch_size=self.batch_size, transform=self.transform)
        data_loader = self.test_loader

        print("{}_dataset size: {}".format(mode,len(data_loader.dataset)))
        
        correct_dict = {}
        for attr in self.selected_attrs:
            correct_dict[attr] = 0

        confusion_matrix_dict = {}
        confusion_matrix_dict['TP'] = [0 for i in range(len(self.selected_attrs))]
        confusion_matrix_dict['TN'] = [0 for i in range(len(self.selected_attrs))]
        confusion_matrix_dict['FP'] = [0 for i in range(len(self.selected_attrs))]
        confusion_matrix_dict['FN'] = [0 for i in range(len(self.selected_attrs))]
        confusion_matrix_dict['precision'] = [0 for i in range(len(self.selected_attrs))]
        confusion_matrix_dict['recall'] = [0 for i in range(len(self.selected_attrs))]
        confusion_matrix_dict['TPR'] = [0 for i in range(len(self.selected_attrs))]
        confusion_matrix_dict['FPR'] = [0 for i in range(len(self.selected_attrs))]
        confusion_matrix_dict['F1'] = [0 for i in range(len(self.selected_attrs))]

        with torch.no_grad():
            for batch_idx, samples in enumerate(data_loader):

                images, labels, identity = samples
                images = images.to(self.device)
                labels = torch.stack(labels).t()
                identity = list(identity)
                outputs, identity_probability, global_features, local_features = self.model(images)
                for i in range(len(outputs)):
                    for j, attr in enumerate(self.selected_attrs):
                        pred = outputs[i].data[j]
                        pred = 1 if pred > self.attr_threshold[j] else 0

                        # record accuracy
                        if pred == labels[i][j]:
                            correct_dict[attr] = correct_dict[attr] + 1

                        if pred == 1 and labels[i][j] == 1:
                            confusion_matrix_dict['TP'][j] += 1  # TP
                        if pred == 1 and labels[i][j] == 0:
                            confusion_matrix_dict['FP'][j] += 1  # FP
                        if pred == 0 and labels[i][j] == 1:
                            confusion_matrix_dict['FN'][j] += 1  # TN  
                        if pred == 0 and labels[i][j] == 0:
                            confusion_matrix_dict['TN'][j] += 1  # FN
                if batch_idx % 50 == 0:
                    print("\r[{}]: Batch_idx : {}/{}, time: {}".format(mode, 
                                batch_idx, int(len(data_loader.dataset)/self.batch_size), 
                                utils.timeSince(self.start_time)), end='')
            i = 0
            for attr in self.selected_attrs:
                correct_dict[attr] = correct_dict[attr] * 100 / len(data_loader.dataset)
                confusion_matrix_dict['precision'][i] = confusion_matrix_dict['TP'][i]/(confusion_matrix_dict['FP'][i] 
                                                        + confusion_matrix_dict['TP'][i] + 1e-6)
                confusion_matrix_dict['recall'][i]= confusion_matrix_dict['TP'][i]/(confusion_matrix_dict['FN'][i] 
                                                    + confusion_matrix_dict['TP'][i] + 1e-6)
                confusion_matrix_dict['TPR'][i]= confusion_matrix_dict['TP'][i]/(confusion_matrix_dict['TP'][i] 
                                                    + confusion_matrix_dict['FN'][i] + 1e-6)
                confusion_matrix_dict['FPR'][i]= confusion_matrix_dict['FP'][i]/(confusion_matrix_dict['FP'][i] 
                                                    + confusion_matrix_dict['TN'][i] + 1e-6)
                confusion_matrix_dict['F1'][i] = 2*confusion_matrix_dict['precision'][i]*confusion_matrix_dict['recall'][i]/(confusion_matrix_dict['precision'][i] + confusion_matrix_dict['recall'][i] + 1e-6)                                                                          
                i += 1
            
            mean_attributes_acc = 0.0
            for k, v in correct_dict.items():
                mean_attributes_acc += v
            mean_attributes_acc /= len(self.selected_attrs)

        return correct_dict, confusion_matrix_dict, mean_attributes_acc

    def test_model(self, model_path):
        # load model
        self.load_model_dict(model_path)
        print("The model has loaded the state dict on {}".format(model_path))
        # test model acc.
        test_acc_dict, confusion_matrix_dict, mean_attributes_acc = self.evaluate("test")
        print(test_acc_dict)
        print(mean_attributes_acc)
