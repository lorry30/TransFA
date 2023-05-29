CUDA_VISIBLE_DEVICES=1 python test_model.py --model_type "swin_transformer" \
--batch_size 50 \
--epochs 10 \
--load_model_path "/mnt/Harddisk/HeWJ/face_attribute_evaluation_20220529/result/v22-1-2-swin_transformer/model_best.pth.tar" \
--exp_version "v22-1-2"

CUDA_VISIBLE_DEVICES=1 python test_model.py --model_type "swin_transformer" \
--batch_size 50 \
--epochs 10 \
--load_model_path "/mnt/Harddisk/HeWJ/face_attribute_evaluation_20220529/result/v22-2-2-swin_transformer/model_best.pth.tar" \
--exp_version "v22-2-2"

CUDA_VISIBLE_DEVICES=1 python test_model.py --model_type "swin_transformer" \
--batch_size 50 \
--epochs 10 \
--load_model_path "/mnt/Harddisk/HeWJ/face_attribute_evaluation_20220529/result/pretrained_model/model.pth.tar" \
--exp_version "v20-5-1"

# python test_model.py --model_type "swin_transformer" \
# --batch_size 64 \
# --epochs 10 \
# --load_model_path "/home/gpu/Documents/MachineLearningCode/FaceAttr-Analysis-master/result/v18-3-1-swin_transformer/model_best.pth.tar" \
# --exp_version "v18-3-1"

# python test_model.py --model_type "swin_transformer" \
# --batch_size 64 \
# --epochs 10 \
# --load_model_path "/home/gpu/Documents/MachineLearningCode/FaceAttr-Analysis-master/result/v18-2-1-swin_transformer/model_best.pth.tar" \
# --exp_version "v18-2-1"

# python test_model.py --model_type "swin_transformer" \
# --batch_size 64 \
# --epochs 10 \
# --load_model_path "/home/gpu/Documents/MachineLearningCode/FaceAttr-Analysis-master/result/v15-2-5-swin_transformer/model_best.pth.tar" \
# --exp_version "v15-2-5"

python test_model.py --model_type "swin_transformer" \
--batch_size 64 \
--epochs 10 \
--load_model_path "/home/gpu/Documents/MachineLearningCode/FaceAttr-Analysis-master/result/model_from_2080/beta0.5_model_best.pth.tar" \
--exp_version "v20-2-1"

python test_model.py --model_type "swin_transformer" \
--batch_size 64 \
--epochs 10 \
--load_model_path "/home/gpu/Documents/MachineLearningCode/FaceAttr-Analysis-master/result/v20-3-1-swin_transformer/model_best.pth.tar" \
--exp_version "v20-3-1-swin_S"

python test_model.py --model_type "swin_transformer" \
--batch_size 64 \
--epochs 10 \
--load_model_path "/home/gpu/Documents/MachineLearningCode/FaceAttr-Analysis-master/result/model_from_2080/beta0.05_model_best.pth.tar" \
--exp_version "v20-4-1"

python test_model.py --model_type "swin_transformer" \
--batch_size 64 \
--epochs 10 \
--load_model_path "/home/gpu/Documents/MachineLearningCode/FaceAttr-Analysis-master/result/v19-2-2-swin_transformer/model_best.pth.tar" \
--exp_version "v19-2-2"

python test_model.py --model_type "swin_transformer" \
--batch_size 50 \
--epochs 10 \
--load_model_path "/home/gpu/Documents/MachineLearningCode/FaceAttr-Analysis-master/result/model_from_2080/beta0.3_model_best.pth.tar" \
--exp_version "v20-5-1"

# python test_model.py --model_type "swin_transformer" \
# --batch_size 64 \
# --epochs 10 \
# --load_model_path "/home/gpu/Documents/MachineLearningCode/FaceAttr-Analysis-master/result/model_from_2080/beta0.3_model_best.pth.tar" \
# --exp_version "v20-1-1"

# python test_model.py --model_type "swin_transformer" \
# --batch_size 64 \
# --epochs 10 \
# --load_model_path "/home/gpu/Documents/MachineLearningCode/FaceAttr-Analysis-master/result/v18-4-2-swin_transformer/model_best.pth.tar" \
# --exp_version "v18-4-2"

# python test_model.py --model_type "swin_transformer" \
# --batch_size 64 \
# --epochs 10 \
# --load_model_path "/home/gpu/Documents/MachineLearningCode/FaceAttr-Analysis-master/result/v18-4-2-swin_transformer/model_best.pth.tar" \
# --exp_version "v18-4-2"

# python test_model.py --model_type "swin_transformer" \
# --batch_size 64 \
# --epochs 10 \
# --load_model_path "/home/gpu/Documents/MachineLearningCode/FaceAttr-Analysis-master/result/v18-4-2-swin_transformer/model_best.pth.tar" \
# --exp_version "v18-4-2"