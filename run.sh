# python main.py  --model_type "swin_transformer" \
# --batch_size 100 \
# --epochs 80 \
# --load_model_path "/home/gpu/Documents/MachineLearningCode/FaceAttr-Analysis-master/result/v20-1-1-swin_transformer/checkpoint.pth.tar" \
# --weight_alpha 0.1 \
# --weight_beta 0.3 \
# --weight_lamda 5 \
# --exp_version "v20-2-1"

python main.py  --model_type "swin_transformer" \
--batch_size 50 \
--epochs 100 \
--load_model_path "/mnt/Harddisk/HeWJ/face_attribute_evaluation_20220529/result/pretrained_model/v15-2-5_swin_transformer_model_best.pth.tar" \
--weight_alpha 0 \
--weight_global_contrastive_identity 0.9 \
--weight_beta 0.3 \
--weight_lamda 1 \
--exp_version "v22-2-2"

python main.py  --model_type "swin_transformer" \
--batch_size 50 \
--epochs 100 \
--load_model_path "/mnt/Harddisk/HeWJ/face_attribute_evaluation_20220529/result/pretrained_model/v15-2-5_swin_transformer_model_best.pth.tar" \
--weight_alpha 0.1 \
--weight_global_contrastive_identity 0 \
--weight_beta 0.3 \
--weight_lamda 1 \
--exp_version "v22-3-2"

CUDA_VISIBLE_DEVICES=1 python main.py  --model_type "swin_transformer" \
--batch_size 50 \
--epochs 100 \
--load_model_path "/mnt/Harddisk/HeWJ/face_attribute_evaluation_20220529/result/pretrained_model/v15-2-5_swin_transformer_model_best.pth.tar" \
--weight_alpha 0.1 \
--weight_global_contrastive_identity 0.9 \
--weight_beta 0 \
--weight_lamda 1 \
--exp_version "v22-1-2"

#python main.py  --model_type "swin_transformer" \
#--batch_size 50 \
#--epochs 200 \
#--load_model_path "/home/gpu/Documents/MachineLearningCode/FaceAttr-Analysis-master/result/v20-4-1-swin_transformer/checkpoint.pth.tar" \
#--weight_alpha 0.1 \
#--weight_beta 0.3 \
#--weight_lamda 5 \
#--exp_version "v20-4-2"
#
#
#python main.py  --model_type "swin_transformer" \
#--batch_size 50 \
#--epochs 200 \
#--load_model_path "" \
#--weight_alpha 0.1 \
#--weight_beta 0.3 \
#--weight_lamda 5 \
#--exp_version "v20-4-1"
#
#python main.py  --model_type "swin_transformer" \
#--batch_size 80 \
#--epochs 200 \
#--load_model_path "" \
#--weight_alpha 0.1 \
#--weight_beta 0.3 \
#--weight_lamda 7 \
#--exp_version "v20-3-1"
#
#python main.py  --model_type "swin_transformer" \
#--batch_size 80 \
#--epochs 200 \
#--load_model_path "" \
#--weight_alpha 0.1 \
#--weight_beta 0.3 \
#--weight_lamda 9 \
#--exp_version "v20-4-1"

# python main.py  --model_type "swin_transformer" \
# --batch_size 64 \
# --epochs 30 \
# --load_model_path "/home/gpu/Documents/MachineLearningCode/FaceAttr-Analysis-master/result/v15-2-5-swin_transformer/model_best.pth.tar" \
# --weight_alpha 0.1 \
# --weight_beta 0.5 \
# --exp_version "v18-5-1"

# python main.py --model_type "swin_transformer" \
# --batch_size 64 \
# --epochs 30 \
# --load_model_path "/home/gpu/Documents/MachineLearningCode/FaceAttr-Analysis-master/result/v15-2-5-swin_transformer/model_best.pth.tar"\
# --weight_alpha 0.1 \
# --weight_beta 0.3 \
# --exp_version "v18-6-1"

# python main.py --model_type "swin_transformer" \
# --batch_size 64 \
# --epochs 30 \
# --load_model_path "/home/gpu/Documents/MachineLearningCode/FaceAttr-Analysis-master/result/v15-2-5-swin_transformer/model_best.pth.tar" \
# --weight_alpha 0.1 \
# --weight_beta 0.1 \
# --exp_version "v18-7-1"

# python main.py --model_type "swin_transformer" \
# --batch_size 64 \
# --epochs 30 \
# --load_model_path "/home/gpu/Documents/MachineLearningCode/FaceAttr-Analysis-master/result/v15-2-5-swin_transformer/model_best.pth.tar" \
# --weight_alpha 0.1 \
# --weight_beta 0.05 \
# --exp_version "v18-8-1"

# python test_model.py --model_type "swin_transformer" \
# --batch_size 64 \
# --epochs 10 \
# --load_model_path "/home/gpu/Documents/MachineLearningCode/FaceAttr-Analysis-master/result/v18-4-2-swin_transformer/model_best.pth.tar" \
# --exp_version "v18-4-2"
# --model_type Resnet18 \
# --batch_size 32 \
# --epoches 50 \
# --learning_rate 1e-3 \
# --momentum 0.9 \
# --optim_type SGD \
# --loss_type BCE_loss \
# --exp_version v16 \
# --load_model_path "" & \

# parser.add_argument('--model_type', choices=[
#                     'Resnet101','Resnet152','Resnet50',
#                     'gc_resnet101','gc_resnet50',
#                     'se_resnet101','se_resnet50', 
#                     'sk_resnet101', 'sk_resnet50',
#                     'sge_resnet101','sge_resnet50', 
#                     "shuffle_netv2", 'densenet121',
#                     "cbam_resnet101","cbam_resnet50"], 
#                     default=cfg.model_type)
# parser.add_argument('--batch_size', default=cfg.batch_size, type=int, help='batch_size')
# parser.add_argument('--epochs', default=cfg.epochs, type=int, help='epochs')
# parser.add_argument('--learning_rate', default=[cfg.learning_rate_large, cfg.learning_rate_small], type=float, help='learning_rate')
# parser.add_argument('--momentum', default=cfg.momentum, type=float, help='momentum')
# parser.add_argument('--optim_type', choices=['SGD','Adam'], default=cfg.optim_type)
# parser.add_argument('--pretrained', action='store_true', default=cfg.pretrained)
# parser.add_argument("--loss_type", choices=['BCE_loss', 'focal_loss'], default=cfg.loss_type)
# parser.add_argument("--exp_version",type=str, default=cfg.exp_version)
# parser.add_argument("--load_model_path", default=cfg.load_model_path, type=str)
# parser.add_argument("--weight_alpha", default=cfg.weight_global_identity, type=float)
# parser.add_argument("--weight_beta", default=cfg.weight_local_identity, type=float)



