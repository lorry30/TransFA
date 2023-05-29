from __future__ import print_function
from __future__ import division
from cProfile import label
from pyexpat import model
from re import L
from this import d
from threading import local
import torch
import torchvision
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import pandas as pd 

import copy
import time
import json
import os
import shutil

from CelebA import get_loader, get_identity_index_mapping
from utils import Logger
from utils.AutomaticWeightedLoss import AutomaticWeightedLoss
from utils.IdentityLoss import IdentityLossFunc
import torch.nn.functional as F
import utils
from FaceAttr_baseline_model import FaceAttrModel, FaceAttrModelModify
from Module.focal_loss import FocalLoss
from Module.contrastive_loss import ContrastiveLoss
import config as cfg

from tensorboardX import SummaryWriter
import datetime
# from torch.utils.tensorboard import SummaryWriter   

class Solver(object):
    
    def __init__(self, epoches, batch_size, learning_rate, model_type, 
        optim_type, momentum, pretrained, loss_type, exp_version, weight):
        self.epoches = epoches 
        self.weight = weight
        self.start_epoch = 0
        self.batch_size = batch_size
        self.learning_rate_large = learning_rate[0]
        self.learning_rate_small = learning_rate[1]
        self.selected_attrs = cfg.selected_attrs
        self.momentum = momentum
        self.device = torch.device("cuda:" + str(cfg.DEVICE_ID) if torch.cuda.is_available() else "cpu")
        self.image_dir = cfg.image_dir
        self.attr_path = cfg.attr_path
        self.identity_path = cfg.identity_path
        self.pretrained = pretrained
        self.model_type = model_type
        self.face_attribute_model_type = cfg.face_attribute_model_type
        self.build_model(model_type, pretrained)
        # print(self.model)
        self.optim_type = optim_type
        self.create_optim(optim_type)
        self.train_loader = None
        self.validate_loader = None
        self.test_loader = None
        self.identity_index_map = None
        self.log_dir = cfg.log_dir
        self.use_tensorboard = cfg.use_tensorboard
        self.attr_loss_weight = torch.tensor(cfg.attr_loss_weight).to(self.device)
        self.attr_loss_weight_valiable = torch.nn.Parameter(torch.FloatTensor(1, 40), requires_grad=True)
        self.attr_loss_weight_valiable.data.fill_(1)
        # print(self.attr_loss_weight)
        self.attr_threshold = cfg.attr_threshold
        self.model_save_path = None
        self.LOADED = False
        self.start_time = 0
        self.loss_type = loss_type
        self.exp_version = exp_version
        torch.cuda.set_device(cfg.DEVICE_ID)
        self.loss_sum = AutomaticWeightedLoss(1)
        self.identity_loss = None
        self.contrastive_loss1 = ContrastiveLoss(self.batch_size)
        self.contrastive_loss1.to(self.device)
        self.contrastive_loss2 = ContrastiveLoss(self.batch_size)
        self.contrastive_loss2.to(self.device)
        # torch.tensor(, dtype=torch.float32,requires_grad=True).to(self.device)
        # self.datetime =  datetime.datetime.now().strftime('%Y-%m-%d')

    def build_model(self, model_type, pretrained):
        """Here should change the model's structure""" 
        if self.face_attribute_model_type == 0:
            self.model = FaceAttrModel(model_type, pretrained, self.selected_attrs).to(self.device)
        elif self.face_attribute_model_type == 1 or self.face_attribute_model_type == 2:
            self.model = FaceAttrModelModify(model_type, pretrained, self.selected_attrs).to(self.device)
        else:
            raise ValueError("no such a model type, you can try 0 or 1")

    def create_optim(self, optim_type):
        scheduler = None
        if self.face_attribute_model_type == 2:
            large_lr_layers = list(map(id,self.model.featureClassfier.identity_network.parameters()))
            small_lr_layers = filter(lambda p:id(p) not in large_lr_layers,self.model.parameters())
        if optim_type == "Adam":
            if self.face_attribute_model_type == 0 or self.face_attribute_model_type == 1:
                self.optim_ = optim.Adam(self.model.parameters(), lr = self.learning_rate_large, momentum = self.momentum)
            elif self.face_attribute_model_type == 2:
                self.optim_ = optim.Adam([{"params":self.model.featureClassfier.identity_network.parameters(),"lr":self.learning_rate_large},
                                     {"params":small_lr_layers,"lr":self.learning_rate_small}])
            # self.scheduler = optim.lr_scheduler.MultiStepLR(self.optim_, cfg.lr_step_array, gamma=0.1)
            self.scheduler = optim.lr_scheduler.StepLR(self.optim_, step_size=200, gamma=0.9)
        elif optim_type == "SGD":
            if self.face_attribute_model_type == 0 or self.face_attribute_model_type == 1:
                self.optim_ = optim.SGD(self.model.parameters(), lr = self.learning_rate_large, momentum = self.momentum)
            elif self.face_attribute_model_type == 2:
                self.optim_ = optim.SGD([{"params":self.model.featureClassfier.identity_network.parameters(),"lr":self.learning_rate_large},
                                        {"params":small_lr_layers,"lr":self.learning_rate_small}])
            self.scheduler = optim.lr_scheduler.StepLR(self.optim_, step_size=200, gamma=0.9)
        else:
            raise ValueError("no such a "+ optim_type + "optim, you can try Adam or SGD.")

    def set_transform(self, mode):
        transform = []
        if mode == 'train':
            transform.append(transforms.RandomHorizontalFlip())
            transform.append(transforms.RandomRotation(degrees=30))
            # transform.append(RandomBrightness())
            # transform.append(RandomContrast())
            # transform.append(RandomHue())
            # transform.append(RandomSaturation())
        # the advising transforms way in imagenet
        # the input image should be resized as 224 * 224 for resnet.
        transform.append(transforms.Resize(size=(224, 224))) # test no resize operation.
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5]))
        
        transform = transforms.Compose(transform)
        self.transform = transform


    # self define loss function
    def BCE_loss(self, input_, target):
        # cost_matrix = [1 for i in range(len(self.selected_attrs))]
        loss = F.binary_cross_entropy(input_.to(self.device),  
                                    target.type(torch.FloatTensor).to(self.device), 
                                    weight=self.attr_loss_weight.type(torch.FloatTensor).to(self.device))
        return loss
        
    def focal_loss(self, inputs, targets):
        focal_loss_func = FocalLoss()
        focal_loss_func.to(self.device)
        return focal_loss_func(inputs, targets)

    def lcloss(self, identity, global_features):
        # identity_input = torch.tensor([]).to(self.device)
        # identity_target = torch.tensor([]).to(self.device)
        loss_function = nn.MSELoss(reduction='sum')
        loss_function.to(self.device)
        sum_loss = loss_function(torch.tensor([0], dtype=torch.float16, requires_grad=True).to(self.device), 
                                torch.tensor([0], dtype=torch.float16, requires_grad=True).to(self.device))
        for i in range(0, self.batch_size):
            for j in range(i + 1, self.batch_size):
                if identity[i] == identity[j]:
                    sum_loss += loss_function(global_features[i].clone(), global_features[j])

        return sum_loss / self.batch_size / (self.batch_size - 1)

    def Contrastive_Loss(self, identity, global_features, local_features):
        # generate label metric
        identity_label = torch.zeros([self.batch_size, self.batch_size]) # the label metric
        for i in range(0, self.batch_size):
            for j in range(i + 1, self.batch_size):
                if identity[i] == identity[j]: 
                    identity_label[i][j] = 1 # if two samples have the same identity
        identity_label = identity_label.to(self.device)
        # caculate loss
        global_contrastive_loss = self.contrastive_loss1(global_features[0], global_features, identity_label[0])
        local_contrastive_loss = self.contrastive_loss2(local_features[0], local_features, identity_label[0])

        for i in range(1, self.batch_size):
            global_contrastive_loss = global_contrastive_loss + self.contrastive_loss1(global_features[i], global_features, identity_label[i])
            local_contrastive_loss = local_contrastive_loss + self.contrastive_loss2(local_features[i], local_features, identity_label[i])
        return global_contrastive_loss, local_contrastive_loss / 7

    def load_model_dict(self, model_state_dict_path):
        if os.path.isfile(model_state_dict_path):
            print("=> loading checkpoint '{}'".format(model_state_dict_path))
            checkpoint = torch.load(model_state_dict_path)
            if 'state_dict' in checkpoint.keys():
                pretrained_dict = checkpoint['state_dict']
            else:
                pretrained_dict = checkpoint
            model_dict = self.model.state_dict()
            pretrained_dict = {k:v for k, v in pretrained_dict.items() if k in model_dict}
            # if cfg.dataset == "LFWA":
            #     pop_list = ["featureClassfier.identity_network.8.bias", "featureClassfier.identity_network.8.weight"]
            #     for pop_object in pop_list:
            #         pretrained_dict.pop(pop_object)
            model_dict.update(pretrained_dict)
            # self.model.load_state_dict(checkpoint['state_dict'])
            self.model.load_state_dict(model_dict)
            if 'epoch' not in checkpoint.keys():
                checkpoint['epoch'] = 1
            self.start_epoch =  checkpoint['epoch']
            self.model_save_path = model_state_dict_path
            # print("=> loaded checkpoint '{}' (epoch {})"
            #       .format(model_state_dict_path, checkpoint['epoch']))
            # print("Train of modle {} will start from the {} epoch".format(checkpoint['model_type'], checkpoint['epoch']))
        # self.model_save_path = model_state_dict_path

    def save_model_dict(self, model_state_dict_path):
        torch.save(self.model.state_dict(), model_state_dict_path)
        print("The model has saved!")

    def train(self, epoch):
        """
        Return: the average trainging loss value of this epoch
        """
        self.model.train()
        self.set_transform("train")

        # to avoid loading dataset repeatedly
        if self.train_loader == None:
            self.train_loader = get_loader(image_dir = self.image_dir, attr_path = self.attr_path, identity_path=self.identity_path, 
                                            selected_attrs = self.selected_attrs, mode="train", 
                                            batch_size=self.batch_size, transform=self.transform)
            print("train_dataset size: {}".format(len(self.train_loader.dataset)))    

        if self.identity_loss == None:
            self.identity_index_map, self.identity_index_map_validation = get_identity_index_mapping(cfg.image_dir,cfg.attr_path, cfg.identity_path, cfg.selected_attrs, self.transform, mode = 'train')
            print("The length of identity to index mapping is {}".format(len(self.identity_index_map)))
            self.identity_loss = IdentityLossFunc(cfg.batch_size, self.identity_index_map).to(self.device)
            
        temp_loss = 0.0
        temp_global_loss = 0.0
        temp_attribute_loss = 0.0
        temp_global_contrastive_loss = 0.0
        temp_local_contrastive_loss = 0.0

        for batch_idx, samples in enumerate(self.train_loader):
            images, labels, identity = samples

            # preprocess images\labels\identity
            labels = torch.stack(labels).t() # [batchsize, attributes nums]
            images= images.to(self.device)
            identity = list(identity)
            identity_label = []
            for i in range(0, self.batch_size):
                identity_label.append(self.identity_index_map[identity[i]])
            identity_label = torch.tensor(identity_label).to(self.device)

            if self.face_attribute_model_type == 0:
                outputs = self.model(images) # [batchsize, attributes nums]
            else:
                outputs, identity_probability, global_features, local_features = self.model(images) # [batchsize, attributes nums]
                global_features.to(self.device)
                # outputs, global_features = self.model(images) # [batchsize, attributes nums]
            self.optim_.zero_grad()
            if self.loss_type == "BCE_loss":
                attribute_loss = self.BCE_loss(outputs, labels) * self.weight[3]
                identity_loss = self.identity_loss(identity_probability.to(self.device), identity_label.to(self.device)) * self.weight[0]
                global_contrastive_loss, local_contrastive_loss = self.Contrastive_Loss(identity, global_features, local_features)
                global_contrastive_loss = global_contrastive_loss * self.weight[1]
                local_contrastive_loss = local_contrastive_loss * self.weight[2]
                total_loss = attribute_loss + identity_loss.to(self.device)  + global_contrastive_loss + local_contrastive_loss

            elif self.loss_type == "focal_loss":
                attribute_loss = self.focal_loss(outputs, labels)
                # identity_loss = self.identity_loss(identity_probability, identity_label)
                total_loss = attribute_loss

            total_loss.backward()
            self.optim_.step()
            temp_loss += total_loss.item()
            temp_global_loss += identity_loss.item()
            temp_attribute_loss += attribute_loss.item()
            temp_global_contrastive_loss += global_contrastive_loss.item()
            temp_local_contrastive_loss += local_contrastive_loss.item()

            if batch_idx % 50 == 0:
                if self.face_attribute_model_type == 0:                   
                    print("\r[train]Epoch: {}/{}, training batch_idx : {}/{}, time: {}, loss: {}, attribute loss: {}".format(epoch, self.epoches, 
                                    batch_idx, int(len(self.train_loader.dataset)/self.batch_size), 
                                    utils.timeSince(self.start_time), total_loss.item(), attribute_loss.item()), end='')
                else:
                    print("\r[train]Epoch: {}/{}, training batch_idx : {}/{}, time: {}, loss: {}, global loss: {}, attribute loss: {}, global_contrastive_loss:{}, local_contrastive_loss:{}".format(epoch + self.start_epoch, self.epoches + self.start_epoch, 
                                batch_idx, int(len(self.train_loader.dataset)/self.batch_size), 
                                utils.timeSince(self.start_time), total_loss.item(),  identity_loss.item(), attribute_loss.item(), global_contrastive_loss.item(), local_contrastive_loss.item()), end='')
        return temp_loss/(batch_idx + 1), temp_global_loss/(batch_idx + 1), temp_attribute_loss/(batch_idx + 1), temp_global_contrastive_loss/(batch_idx + 1), temp_local_contrastive_loss / (batch_idx + 1)
        
    def evaluate(self, mode):
        """
        Mode: validate or test mode
        Return: correct_dict: save the average predicting accuracy of every attribute
        """
        self.model.eval()
        self.set_transform(mode)
        data_loader = None
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
        if mode == 'validate':
            data_loader = self.validate_loader
        elif mode == 'test':
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

        validation_temp_loss = 0.0
        validation_temp_global_loss = 0.0
        validation_temp_attribute_loss = 0.0
        validation_temp_global_contrastive_loss = 0.0
        validation_temp_local_contrastive_loss = 0.0

        with torch.no_grad():
            for batch_idx, samples in enumerate(data_loader):
                """
                    data_loader:
                    {
                        'image': [batch_size, C, H, W],
                        'label': [batch_size, num_attr]
                    }
                """
                images, labels, identity = samples
                images = images.to(self.device)
                # labels = torch.stack(labels).t().tolist()
                labels = torch.stack(labels).t()
                identity = list(identity)
                identity_label = []
                if self.face_attribute_model_type == 0:
                    outputs = self.model(images)
                else:
                    # outputs, global_features = self.model(images)
                    outputs, identity_probability, global_features, local_features = self.model(images)
                    if mode == 'validate':
                        for i in range(0, self.batch_size):
                            identity_label.append(self.identity_index_map_validation[identity[i]])
                        identity_label = torch.tensor(identity_label).to(self.device)
                        validation_attribute_loss = self.BCE_loss(outputs, labels) * cfg.weight_attribute
                        validation_identity_loss = self.identity_loss(identity_probability.to(self.device), identity_label.to(self.device)) * self.weight[0]
                        validation_global_contrastive_loss, validation_local_contrastive_loss = self.Contrastive_Loss(identity, global_features, local_features)
                        validation_global_contrastive_loss = validation_global_contrastive_loss * self.weight[1]
                        validation_local_contrastive_loss = validation_local_contrastive_loss * self.weight[2]
                        validation_total_loss = validation_attribute_loss + validation_identity_loss.to(self.device)  + validation_global_contrastive_loss + validation_local_contrastive_loss

                        validation_temp_loss += validation_total_loss.item()
                        validation_temp_global_loss += validation_identity_loss.item()
                        validation_temp_attribute_loss += validation_attribute_loss.item()
                        validation_temp_global_contrastive_loss += validation_global_contrastive_loss.item()
                        validation_temp_local_contrastive_loss += validation_local_contrastive_loss.item()

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
            # get the average accuracy
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
        print("")

        validation_loss_dict = {
            'validation_temp_loss' : 0,
            'validation_temp_global_loss' : 0,
            'validation_temp_attribute_loss' : 0,
            'validation_temp_global_contrastive_loss' : 0,
            'validation_temp_local_contrastive_loss' : 0
        }
        return correct_dict, confusion_matrix_dict, mean_attributes_acc, validation_loss_dict


    def fit(self, model_path=""):
        """
        This function is to combine the train and evaluate, finally getting a best model.
        """
        print("-------------------------------")
        print("You method is {}-{}-{}_epoches".format(self.exp_version, self.model_type, self.epoches))
        print("-------------------------------")

        if model_path is not "":
            self.load_model_dict(model_path)
            print("The model has loaded the state dict on {}".format(model_path))
        else:
            self.model_save_path = "./result/" + self.exp_version + '-' +  self.model_type + "/checkpoint.pth.tar"
        train_losses = []
        train_global_losses = []
        train_attribute_losses = []

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        eval_acc_dict = {}
        confusion_matrix_df = None 

        min_loss = 1000
        early_stop_cnt = 0

        for attr in self.selected_attrs:
            eval_acc_dict[attr] = []
        self.start_time = time.time()

        # record logger
        logger = Logger(os.path.join("./result/" + self.exp_version + '-' +  self.model_type + '/', 'log.txt'), title=self.exp_version + '-' +  self.model_type)
        if self.face_attribute_model_type == 0 or self.face_attribute_model_type == 1:
            logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
        else:
            logger.set_names(['Learning Rate Identity Branch', 'Learning Rate BackBone','Train Loss', 'Train Global Loss', 'Train attribute loss', 'Train Global Contrastive Loss','Train Local Contrastive Loss','Valid Acc.'])
        writer = SummaryWriter(os.path.join("./result/" + self.exp_version + '-' +  self.model_type + '/'))

        for epoch in range(self.epoches):
            is_best = False
            print("---------------------------------------------------------")
            running_loss, running_global_loss, running_attribute_loss, running_global_contrastive_loss, running_local_contrastive_loss = self.train(epoch)
            print("{}/{} Epoch:  in training process average loss: {:.4f}, global_loss:{:.4f}, attribute_loss:{:.4f}, global_contrastive_loss:{:.4f}"
                .format(epoch + 1 + self.start_epoch, self.epoches + self.start_epoch, running_loss, running_global_loss,running_attribute_loss, running_global_contrastive_loss))
            print("The running time since the start is : {} ".format(utils.timeSince(self.start_time)))
            average_acc_dict, confusion_matrix_dict, mean_attributes_acc, validation_loss_dict = self.evaluate("validate")
            print("{}/{} Epoch: in evaluating process average accuracy:{}".format(epoch + 1 + self.start_epoch, self.epoches + self.start_epoch, average_acc_dict))
            print("{}/{} Epoch: the mean accuracy is {}".format(epoch + 1 + self.start_epoch, self.epoches + self.start_epoch, mean_attributes_acc))
            print("The running time since the start is : {} ".format(utils.timeSince(self.start_time)))

            self.scheduler.step()
            train_losses.append(running_loss)
            train_global_losses.append(running_global_loss)
            train_attribute_losses.append(running_attribute_loss)
            average_acc = 0.0

            # Record the evaluating accuracy of every attribute at current epoch
            for attr in self.selected_attrs:
                eval_acc_dict[attr].append(average_acc_dict[attr])
                average_acc += average_acc_dict[attr]
            average_acc /= len(self.selected_attrs) # overall accuracy
            
            # record the experiment data every epoch: lr, train loss, valid acc.
            learning_rate = self.scheduler.get_last_lr()
            if self.face_attribute_model_type == 2:
                logger.append([learning_rate[0],learning_rate[1],running_loss, running_global_loss, running_attribute_loss, running_global_contrastive_loss, running_local_contrastive_loss, mean_attributes_acc])
            else:
                logger.append([learning_rate[0], running_loss, running_global_loss, running_attribute_loss, mean_attributes_acc])
            # tensorboardX
            writer.add_scalar('learning rate identity branch', learning_rate[0], epoch + 1 + self.start_epoch)
            if self.face_attribute_model_type == 2:
                writer.add_scalar('learning rate backbone', learning_rate[1], epoch + 1 + self.start_epoch)
            writer.add_scalar('train loss', running_loss, epoch + 1 + self.start_epoch)
            writer.add_scalar('train global losses', running_global_loss, epoch + 1 + self.start_epoch)
            writer.add_scalar('train attribute losses', running_attribute_loss, epoch + 1 + self.start_epoch)
            writer.add_scalar('train global Contrastive Loss', running_global_contrastive_loss, epoch + 1 + self.start_epoch)
            writer.add_scalar('train local Contrastive Loss', running_local_contrastive_loss, epoch + 1 + self.start_epoch)
            writer.add_scalar('validation accuracy', mean_attributes_acc, epoch + 1 + self.start_epoch)
            for key, value in validation_loss_dict.items():
                writer.add_scalar(key, value, epoch + 1 + self.start_epoch)

            # find a better model and save it 
            if average_acc > best_acc:
                #and epoch > self.epoches / 2: # for save time 
                best_acc = average_acc
                best_model_wts = copy.deepcopy(self.model.state_dict())
                is_best = True
                confusion_matrix_df = pd.DataFrame(confusion_matrix_dict, index=self.selected_attrs)

            # save the model every epoch
            self.save_checkpoint({
                'epoch': epoch + 1 + self.start_epoch,
                'model_type': self.model_type,
                'state_dict': self.model.state_dict(),
                'best_prec1': best_acc,
            }, is_best, checkpoint="./result/" + self.exp_version + '-' +  self.model_type)

            # if loss does not update, break the loop
            if min_loss > running_loss:
                min_loss = running_loss
                early_stop_cnt  = 0
            else:
                early_stop_cnt += 1
            if early_stop_cnt > cfg.early_stop:
                print("==============================early stop now! at epoch {}".format(epoch + 1 + self.start_epoch) + "==============================")
                self.epoches = epoch + 1
                break
        
        # save the accuracy in files
        eval_acc_csv = pd.DataFrame(eval_acc_dict, index = [i for i in range(self.epoches)]).T 
        eval_acc_csv.to_csv("./result/" + self.exp_version + '-' +  self.model_type + "/eval_accuracy"+ ".csv")

        # save the loss files
        train_losses_csv = pd.DataFrame(train_losses)
        train_losses_csv.to_csv("./result/" + self.exp_version + '-' +  self.model_type + "/losses" +".csv")

        # load best model weights used for test step, we need to load the best weight
        self.model.load_state_dict(best_model_wts)
        self.LOADED = True    
        print("The model has saved in {}".format(self.model_save_path))

        # test the model with test dataset.
        test_acc_dict, confusion_matrix_dict, mean_attributes_acc, test_loss_dict = self.evaluate("test")
        test_acc_csv = pd.DataFrame(test_acc_dict, index=['accuracy'])
        test_acc_csv.to_csv("./result/" + self.exp_version + '-' + self.model_type + "/test_accuracy" + '.csv')
        test_confusion_matrix_csv = pd.DataFrame(confusion_matrix_dict, index=self.selected_attrs)
        test_confusion_matrix_csv.to_csv("./result/" + self.exp_version + '-' + self.model_type + '/confusion_matrix.csv', index=self.selected_attrs)

        # report dictionary
        report_dict = {}
        report_dict["model"] = self.model_type
        report_dict["version"] = self.exp_version
        report_dict["mean_attributes_accuracy"] = mean_attributes_acc
        report_dict["speed"] = self.test_speed()
        report_dict["dataset_name"] = cfg.attr_path
        report_dict["batch_size"] = self.batch_size
        report_dict["epochs"] = self.epoches + self.start_epoch
        report_dict["momentum"] = self.momentum
        report_dict["optim_type"] = self.optim_type
        report_dict["loss_type"] = self.loss_type
        print(report_dict)

        # save the config of this experiment
        origin_dict = vars(cfg)
        configuration_dict = {}
        for _ in origin_dict.keys():
            if not _.startswith("_"):
                configuration_dict[_] = origin_dict[_]
        report_dict['configuration'] = configuration_dict
        report_json = json.dumps(report_dict, ensure_ascii=False)
        report_file = open("./result/" + self.exp_version + "-" + self.model_type + "/report.json", 'w')
        report_file.write(report_json)
        report_file.close()
        
        # save the experiment description
        self.save_experiment_description("./result/experiments.json", self.exp_version + "-" + self.model_type, report_dict['configuration']["experiment_description"])
        # close the logger
        logger.close()
        # close the tensorboard writer
        writer.close()
       
        # print(self.attr_loss_weight)
        
    def predict(self, image):
        if not self.LOADED:
            # load the best model dict.
            self.model.load_state_dict(torch.load("./" + self.model_save_path))
            self.LOADED = True
        self.model.eval()
        with torch.no_grad():
            self.set_transform("predict")
            output = self.model(self.transform(image))
            pred_dict = {}
            for i, attr in enumerate(self.selected_attrs):
                pred = output.data[i]
                pred = pred if pred > self.attr_threshold[i] else 0
                if pred != 0:
                    pred_dict[attr] = pred
            return pred_dict  # return the predicted positive attributes dict and the probability.


    def test_speed(self, image_num=1, model_path=""):
        if model_path  is not "":
            self.model.load_state_dict(torch.load(model_path))
            print("You load the model params: {}".format(model_path))

        self.model.eval()

        with torch.no_grad():
            self.set_transform(mode="test")
            self.test_loader = get_loader(image_dir = self.image_dir, 
                                    attr_path = self.attr_path,
                                    identity_path=self.identity_path, 
                                    selected_attrs = self.selected_attrs,
                                    mode="test", batch_size=image_num, transform=self.transform)
            
            for idx, samples in enumerate(self.test_loader):
                images, labels, identity = samples
                images = images.to(self.device)
                labels = torch.stack(labels).t().tolist()
                start_time = time.time()
                outputs = self.model(images)
                end_time = time.time()

                if idx == 0:
                    speed = image_num / (end_time - start_time)
                    print("You test {} images. The cost time is {}. The speed is {} images/s.".format(image_num,(end_time - start_time),speed))
                    print("---------------------------------------------------------")
                    return end_time-start_time
                    break
    
    def save_checkpoint(self, state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
        if not os.path.exists(checkpoint):
            os.mkdir(checkpoint)
        filepath = os.path.join(checkpoint, filename)
        torch.save(state, filepath)
        # save the best model 
        if is_best:
            shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))
    
    def save_experiment_description(self, file_path, key, value):
        """
        """
        with open(file_path, 'r',encoding="utf-8") as f:
            json_file = json.load(f)
        json_file[key] = value
        with open(file_path, 'w',encoding="utf-8") as f:
            data_json = json.dumps(json_file , ensure_ascii=False)
            f.write(data_json)

    def test_model(self, model_path):
        # load model
        self.load_model_dict(model_path)
        print("The model has loaded the state dict on {}".format(model_path))
        # test model acc.
        test_acc_dict, confusion_matrix_dict, mean_attributes_acc, nothing = self.evaluate("test")
        print(test_acc_dict)
        print(mean_attributes_acc)
        test_acc_csv = pd.DataFrame(test_acc_dict, index=[' accuracy'])
        test_result_save_path = "./test_result/" + self.exp_version + '-' + self .model_type + "/"
        if not os. path.exists(test_result_save_path):
            os.mkdir(test_result_save_path)
        test_acc_csv.to_csv("./test_result/" + self.exp_version + '-' + self.model_type + "/test_accuracy" + '.CSV')
