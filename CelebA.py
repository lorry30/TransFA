import torch
import torchvision
import pandas as pd
from torch.utils import data
import os
import random
from PIL import Image
import config as cfg
import pdb
from torchvision import datasets, transforms, models


class CelebA(data.Dataset):
    
    # each image is  218*178;
    def __init__(self, attr_file, identity_file, selected_attrs, image_folder,
                transform, mode = "train"):

        #self.read_bbox_file(bbox_file)
        self.attr_file = attr_file
        self.identity_file = identity_file
        self.image_folder = image_folder
        self.transform = transform
        self.selected_attrs = selected_attrs

        self.train_dataset = []
        self.validate_dataset = []
        self.test_dataset = []
        self.identity2outputindex = {}
        self.identity2outputindex_validation = {}
        self.identity_set = set()

        self.idx2attr = {}
        self.attr2idx = {}
        if cfg.dataset == "CelebA":
            self.preprocess_CelebA()
        else:
            self.preprocess_LFWA()
        self.mode = mode
        if mode == "train":
            self.num_images = len(self.train_dataset)
        elif mode == "validate":
            self.num_images = len(self.validate_dataset)
        elif mode == "test":
            self.num_images = len(self.test_dataset)
    
    def __len__(self):
        return self.num_images
    
    def __getitem__(self, index):
        """Return image data (tensor) and labels (dict)"""
        dataset = self.train_dataset
        if self.mode == 'validate':
            dataset = self.validate_dataset
        elif self.mode == "test":
            dataset = self.test_dataset
        filename, label, identity = dataset[index]
        image = Image.open(os.path.join(self.image_folder, filename)) 
        if self.transform != None:
            image = self.transform(image)
        return image, label, identity

    def preprocess_CelebA(self):
        lines = [line.rstrip() for line in open(self.attr_file, 'r')]
        identity_lines = [line.rstrip() for line in open(self.identity_file, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1024)
        random.shuffle(lines)

        count_index = 0
        count_index_validation = 0
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            fileindex = int(filename.split(".")[0])
            values = split[1:]
           
            label = []
            # get the identity of the image
            identity_line = identity_lines[fileindex - 1].split()
            identity = identity_line[-1]

            # save the attributes into a dict
            for attr_name in self.selected_attrs:
            
                idx = self.attr2idx[attr_name]
                val = int(values[idx])
                if val == -1:
                    val = 0
                label.append(val)   

            # split the data by index.
            if fileindex < cfg.train_end_index:
                self.train_dataset.append([filename, label, identity])
                # self.identity_set.add(identity)
                if identity not in self.identity2outputindex.keys():
                    self.identity2outputindex[identity] = count_index
                    count_index += 1    

            elif fileindex  < cfg.validate_end_index:
                self.validate_dataset.append([filename, label, identity])
                if identity not in self.identity2outputindex_validation.keys():
                    self.identity2outputindex_validation[identity] = count_index_validation
                    count_index_validation += 1
                
                # if cfg.dataset == "LFWA": # since LFWA dataset only have train and test set, set the test set and validation set same
                #     self.test_dataset.append([filename, label, identity])
                # self.identity_set.add(identity)

            elif fileindex  < cfg.test_end_index:
                self.test_dataset.append([filename, label, identity])
                # self.identity_set.add(identity)

            elif fileindex >= cfg.test_end_index:
                break 
                
        print('Finished preprocessing the CelebA data set...' + self.attr_file)

    def preprocess_LFWA(self):
        lines = [line.rstrip() for line in open(self.attr_file, 'r')]
        identity_lines = [line.rstrip() for line in open(self.identity_file, 'r')]
        
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        # random.seed(1024)
        # random.shuffle(lines)

        count_index = 0
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]
           
            label = []
            # get the identity of the image
            identity_line = identity_lines[i].split()
            identity = identity_line[-1]

            # save the attributes into a dict
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                val = int(values[idx])
                if val == -1:
                    val = 0
                label.append(val)   

            # split the data by index.
            if i < cfg.train_end_index:
                self.train_dataset.append([filename, label, identity])
                if identity not in self.identity2outputindex.keys():
                    self.identity2outputindex[identity] = count_index
                    count_index += 1    

            elif i  < cfg.validate_end_index:
                self.validate_dataset.append([filename, label, identity])
                self.test_dataset.append([filename, label, identity])

            elif i >= cfg.test_end_index:
                break 
                
        print('Finished preprocessing the LFWA data set...' + self.attr_file)

#     # ---------------------------------------------------------------#
#     # 以下函数写完发现暂时用不到。。。。                                #
#     # ---------------------------------------------------------------#
#     def read_bbox_file(self, bbox_file):
#         with open(bbox_file, 'r') as f:
#             bbox_attr = f.readlines()
#             self.bbox_nums = bbox_attr[0]
#             bbox_cols = bbox_attr[1].split()
#             bbox_attr = bbox_attr[2:]
#             frame = []
#             for attr in bbox_attr:
#                 row_sample = attr.split()
#                 frame.append(row_sample)
#             bbox_frame = pd.DataFrame(frame, columns=bbox_cols)
#             self.bbox_frame = bbox_frame
#             return bbox_frame
#
#     def read_attr_file(self, attr_file):
#         with open(attr_file, 'r') as f:
#             attrs = f.readlines()
#             self.image_num = attrs[0]
#             attr_cols = ["ImageID"]
#             attr_cols = attr_cols + attrs[1]
#             attrs = attrs[2:]
#             frame = []
#             for attr in attrs:
#                 row_sample = attr.split()
#                 frame.append(row_sample)
#             face_atrr_frame = pd.DataFrame(frame, columns = attr_cols)
#             return face_atrr_frame
#
#     def read_landmarks_file(self, landmarks_file):
#         with open(landmarks_file, 'r') as f:
#             landmarks = f.readlines()
#             self.image_num = landmarks[0]
#             attr_cols = ["ImageID"]
#             attr_cols = attr_cols + landmarks[1]
#             landmarks = landmarks[2:]
#             frame = []
#             for landmark in landmarks:
#                 row_sample = landmark.split()
#                 frame.append(row_sample)
#             landmark_frame = pd.DataFrame(frame, columns = attr_cols)
#             return landmark_frame
#
#     def read_partition_file(self, partition_file):
#         with open(partition_file, 'r') as f:
#             info = f.readlines()
#             frame = []
#             for line in info:
#                 frame.append(line.split())
#             partition_frame = pd.DataFrame(frame, columns = ["ImageID", "Type"])
#             return partition_frame
#
#
# def collate_fn(batch_data):
#     """
#     batch_data = [{'image': [batch_size, 3, 224, 224], 'label': [batch_size, num_attr]}]
#     """
#     new_batch = {'image': None, 'label': None}
#
#     new_images = None
#     new_labels = None
#     for idx, batch in enumerate(batch_data):
#         image = batch['image']
#         label = batch['label']
#
#         if idx == 0:
#             new_images = image.unsqueeze(0)
#             new_labels = torch.tensor(label).unsqueeze(0)
#         else:
#             image = image.unsqueeze(0)
#             new_images = torch.cat([new_images, image], dim=0)
#             label = torch.tensor(label).unsqueeze(0)
#             new_labels = torch.cat([new_labels, label], dim=0)
#     new_batch['image'] = new_images
#     new_batch['label'] = new_labels
#
#     return new_batch

# 218 * 178
def get_loader(image_dir, attr_path, identity_path,selected_attrs,
               batch_size, mode='train', num_workers=1, transform = None):
    """Build and return a data loader."""
    dataset = CelebA(attr_path, identity_path, selected_attrs, image_dir,transform, mode)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers,
                                  drop_last = True)
    return data_loader

def get_identity_index_mapping(image_dir, attr_path, identity_path,selected_attrs,
               batch_size, mode='train', num_workers=1, transform = None):
    dataset = CelebA(attr_path, identity_path, selected_attrs, image_dir,transform, mode)
    return dataset.identity2outputindex, dataset.identity2outputindex_validation

def test():
    transform = []
    transform.append(transforms.Resize(size=(224, 224)))
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]))
    transform = transforms.Compose(transform)
    dataset = CelebA(cfg.attr_path, cfg.identity_path, cfg.selected_attrs, cfg.image_dir, transform, mode = 'train')
    
 

if __name__ == "__main__":
    test()
