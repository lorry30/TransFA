import argparse
from cgi import print_environ
from unittest import result
import cv2
import numpy as np
import torch
import timm
from FaceAttr_baseline_model import FaceAttrModel, FaceAttrModelModify
import config as cfg
import os

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default='/home/gpu/Documents/MachineLearningCode/multi-task-face-attribute/data/CelebA/celeba/000001.jpg',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')

    parser.add_argument(
        '--method',
        type=str,
        default='gradcam++',
        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


def reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def read_label():
    label_path = "../multi-task-face-attribute/data/CelebA/list_attr_celeba_location01.txt"
    lines = [line.rstrip() for line in open(label_path, 'r')]
    lines = lines[2:]
    result = []
    for line in lines:
        line = line.split()
        for i in range(len(line)):
            # print(line[i])
            # print(type(line[i]))
            if line[i] == "-1":
                line[i] = "0"
        result.append(line)
    # print(result[0])
    return result

if __name__ == '__main__':
    """ python swinT_example.py -image-path <path_to_image>
    Example usage of using cam-methods on a SwinTransformers network.
    """

    args = get_args()
    # args.use_cuda = False # not gpu
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    # load the model
    # model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
    model_type = "swin_transformer"
    model = FaceAttrModel(model_type, pretrained=False, selected_attrs=cfg.selected_attrs)
    checkpoint = torch.load("./result/v15-2-5-swin_transformer/model_best.pth.tar")
    pretrained_dict = checkpoint['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k:v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    # self.model.load_state_dict(checkpoint['state_dict'])
    model.load_state_dict(model_dict)
    model.eval()

    if args.use_cuda:
        model = model.cuda()
    """
    You need to choose the target layer to compute CAM for. Some common choices are:
        FasterRCNN: model.backbone
        Resnet18 and 50: model.layer4[-1]
        VGG and densenet161: model.features[-1]
        mnasnet1_0: model.layers[-1]
        ViT: model.blocks[-1].norm1
        SwinT: model.layers[-1].blocks[-1].norm1
    """
    target_layers = [model.featureExtractor.model[-1].layers[-1].blocks[-1].norm2]
    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    cam = methods[args.method](model=model,
                               target_layers=target_layers,
                               use_cuda=args.use_cuda,
                               reshape_transform=reshape_transform)
    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 1
    # the index of image and selected attributes
    image_index = range(182638, 202599 + 1, 20)
    # image_index = [185538]
    # attributes_index = range(0, 40)[0, 1, .... 39]
    # [19, 20, 21, 22, 23]
    attributes_index = [38, 39]
    # create path to save cam_attribute image
    for i in attributes_index:
        attribute_save_path = "./cam/"+model_type+"/attribute_%06d/"%(i)
        attribute_positive_path = "./cam_positive/"+model_type+"/attribute_%06d/"%(i)
        attribute_predict_right_path = "./cam/"+model_type+"/right/attribute_%06d/"%(i)
        attribute_predict_wrong_path = "./cam/"+model_type+"/wrong/attribute_%06d/"%(i)
        if not os.path.exists(attribute_save_path):
            os.mkdir(attribute_save_path)
        if not os.path.exists(attribute_positive_path):
            os.mkdir(attribute_positive_path)
        if not os.path.exists(attribute_predict_right_path):
            os.mkdir(attribute_predict_right_path)
        if not os.path.exists(attribute_predict_wrong_path):
            os.mkdir(attribute_predict_wrong_path)

    label = read_label()
    # generate the image
    for index in image_index:
        image_path = "/home/gpu/Documents/MachineLearningCode/multi-task-face-attribute/data/CelebA/celeba/%06d.jpg"%(index)
        
        # preprocess image
        rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
        rgb_img = cv2.resize(rgb_img, (224, 224))
        rgb_img = np.float32(rgb_img) / 255
        input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5])
        # decide positive or negative attribute
        outputs = model(input_tensor.cuda())
        if len(outputs) > 1:
            outputs = outputs[0]
        result_outputs = []
        for output in outputs[0]:
            if output.item() > 0.5:
                result_outputs.append(1)
            else:
                result_outputs.append(0)
        for i in attributes_index:  
            # If None, returns the map for the highest scoring category.
            # Otherwise, targets the requested category.
            target_category = i
            attribute_save_path = "./cam/"+model_type+"/attribute_%06d/"%(i)
            attribute_positive_save_path = "./cam_positive/"+model_type+"/attribute_%06d/"%(i)
            attribute_predict_right_path = "./cam/"+model_type+"/right/attribute_%06d/"%(i)
            attribute_predict_wrong_path = "./cam/"+model_type+"/wrong/attribute_%06d/"%(i)
            grayscale_cam = cam(input_tensor=input_tensor,
                            target_category=target_category,
                            eigen_smooth=args.eigen_smooth,
                            aug_smooth=args.aug_smooth)
            # Here grayscale_cam has only one image in the batch
            grayscale_cam = grayscale_cam[0, :]
            cam_image = show_cam_on_image(rgb_img, grayscale_cam)
            # save postivate sample
            if result_outputs[i] == 1: # postivate sample
                save_positive_path = attribute_positive_save_path + "%s-%06d-%06d-%1d.jpg"%(args.method, index, i, result_outputs[i])
                cv2.imwrite(save_positive_path, cam_image)
            # print(label[index - 1])
            label_result = int(label[index - 1][i + 1])
            # save prediction result
            if result_outputs[i] == label_result: # predict right
                save_right_path = attribute_predict_right_path + "%s-%06d-%06d-%1d-%1d.jpg"%(args.method, index, i, result_outputs[i], label_result)
                cv2.imwrite(save_right_path, cam_image)
            else: # predict wrong
                save_wrong_path =  attribute_predict_wrong_path + "%s-%06d-%06d-%1d-%1d.jpg"%(args.method, index, i, result_outputs[i], label_result)
                cv2.imwrite(save_wrong_path, cam_image)
            save_path = attribute_save_path + "%s-%s-%06d-%06d-%1d.jpg"%(args.method, model_type,index, i, result_outputs[i])
            cv2.imwrite(save_path, cam_image) # all sample
        print("%06d.jpg has been processed!"%(index))