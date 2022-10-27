
# ---------------- all data on the same device ----#
DEVICE_ID = 0

# -----------------dataset spilit---------------------#
train_end_index = 162770 + 1
validate_end_index = 182637 + 1
test_end_index = 202599 + 1 

# ------------- Path setting --------------------- #

log_dir = "./log"
# You should download the celeba dataset in the root dir.

# the dataset local path.
# image_dir = "../../CelebA/Img/img_align_celeba/" 
# attr_path = "../../CelebA/Anno/list_attr_celeba.txt"

#the dataset path run on server.

image_dir = "./images"
attr_path = "./label/list_attr_celeba_location01.txt"

# ------------- loss type----------------------------- #
size_average = False
# -------------- Attribute configuration --------- #
all_attrs = ['Attractive', 'Blurry', 'Chubby', 'Heavy_Makeup', 'Male', 
            'Oval_Face', 'Pale_Skin', 'Smiling', 'Young', 'Bald', 
            'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair', 
            'Receding_Hairline', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Hat', 'Arched_Eyebrows', 
            'Bags_Under_Eyes', 'Bushy_Eyebrows', 'Eyeglasses', 'Narrow_Eyes', 'Big_Nose', 
            'Pointy_Nose', '5_o_Clock_Shadow', 'Big_Lips', 'Double_Chin', 'Goatee', 
            'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Sideburns', 'Wearing_Lipstick', 
            'High_Cheekbones', 'Rosy_Cheeks', 'Wearing_Earrings', 'Wearing_Necklace', 'Wearing_Necktie']

# -------------- Identity File Path --------- #
identity_path = "./label/identity_CelebA.txt"

# To be optimized
attr_nums = [i for i in range(len(all_attrs))] 
attr_loss_weight = [1 for i in range(len(all_attrs))]

selected_attrs = [] 
for num in attr_nums:
    selected_attrs.append(all_attrs[num])

# To solve the sample imbalance called rescaling, If the threshold > m+ /(m+ + m-), treat it as a positive sample. 
attr_threshold = [0.5 for i in range(len(all_attrs))]  

# -------------- Tensorboard --------------------- #
use_tensorboard = True

# -------------- Experiment Settings --------------------- #
experiment_description = """"""
model_type = "swin_transformer"
batch_size = 100
pretrained = True
load_model_path = "./pretrained/model.pth.tar"
# -------------- Identity nums --------------------- #
train_identity_num = 8192
validation_identity_num = 985
test_identity_num = 1000
identity_num = 10177