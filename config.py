
# ---------------- all data on the same device ----#
DEVICE_ID = 0

# ---------------- experiment mark -----------------#
exp_version = "v3"

# -----------------choose the dataset you want -----------------#
dataset = "CelebA" # CelebA \ LFWA

# -----------------dataset spilit---------------------#
# the num meas the end index of the dataset [)
if dataset == "CelebA":
    train_end_index = 162770 + 1
    validate_end_index = 182637 + 1
    test_end_index = 202599 + 1 
else: # LFWA dataset only has train and test set
    train_end_index = 6263
    validate_end_index = 13143
    test_end_index = 13143

# ------------- Test or not --------------------- #
test_flag = False #!

# ------------- Path setting --------------------- #

log_dir = "./log"
# You should download the celeba dataset in the root dir.


#the dataset path run on server.
if dataset == "CelebA":
    image_dir = "./data/CelebA/celeba"

else:
    image_dir = "./data/LFWA/lfw" # the dir of the images of LFWA

# test_attr_celeba.txt
# list_attr_celeba_dmtl.txt
# list_attr_celeba.txt，celeba
# list_attr_celeba_location01.txt
if dataset == "CelebA":
    attr_paths = ["./data/CelebA/test_attr_celeba.txt",
                "./data/CelebA/list_attr_celeba_dmtl.txt",
                "./data/CelebA/list_attr_celeba.txt",
                "./data/CelebA/list_attr_celeba_location01.txt"]
    attr_path = attr_paths[3] if test_flag else attr_paths[3]# !
else:
    attr_path = "./data/LFWA/list_attr_lfwa_location01.txt" # the dir of attribute file on LFWA

# ----------- model/train/test configuration ---- #
"""
epoches = 50  # 50

batch_size = 32

learning_rate = 0.001

model_type = "Resnet101"  # 34 50 101 152

optim_type = "SGD"

momentum = 0.9

pretrained = True
"""

# ------------- loss type----------------------------- #
# loss_type = "BCE_loss"  #  focal_loss
# loss_type = "focal_loss"

# Focal loss as described in https://arxiv.org/pdf/1708.02002.pdf
focal_loss_alpha = 0.8
focal_loss_gamma = 2
size_average = False
# -------------- Attribute configuration --------- #

# every row has 5 attributes.
if attr_path == "./data/CelebA/test_attr_celeba.txt" or attr_path == "./data/CelebA/list_attr_celeba.txt":
    # origin data
    all_attrs = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 
                'Bangs', 'Big_Lips', 'Big_Nose','Black_Hair', 'Blond_Hair',
                'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 
                'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 
                'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 
                'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 
                'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 
                'Wearing_Hat','Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young' 
    ]
elif attr_path == "./data/CelebA/list_attr_celeba_dmtl.txt":
    # TPAMI data
    all_attrs = ['Attractive', 'Blurry', 'Chubby', 'Heavy_Makeup', 'Male', 
                'Oval_Face', 'Pale_Skin', 'Smiling', 'Young', 'Bald', 
                'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair', 
                'Receding_Hairline', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Hat', 'Arched_Eyebrows', 
                'Bushy_Eyebrows', 'Bags_Under_Eyes', 'Eyeglasses', 'Narrow_Eyes', 'Big_Nose', 
                'Pointy_Nose', 'High_Cheekbones', 'Rosy_Cheeks', 'Sideburns', 'Wearing_Earrings', 
                '5_o_Clock_Shadow', 'Big_Lips', 'Mouth_Slightly_Open', 'Mustache', 'Wearing_Lipstick', 
                'Double_Chin', 'Goatee', 'No_Beard', 'Wearing_Necklace', 'Wearing_Necktie']
elif attr_path == "./data/CelebA/list_attr_celeba_location01.txt" or attr_path == "./data/LFWA/list_attr_lfwa_location01.txt":
    # our method location01
    all_attrs = ['Attractive', 'Blurry', 'Chubby', 'Heavy_Makeup', 'Male', 
                'Oval_Face', 'Pale_Skin', 'Smiling', 'Young', 'Bald', 
                'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair', 
                'Receding_Hairline', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Hat', 'Arched_Eyebrows', 
                'Bags_Under_Eyes', 'Bushy_Eyebrows', 'Eyeglasses', 'Narrow_Eyes', 'Big_Nose', 
                'Pointy_Nose', '5_o_Clock_Shadow', 'Big_Lips', 'Double_Chin', 'Goatee', 
                'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Sideburns', 'Wearing_Lipstick', 
                'High_Cheekbones', 'Rosy_Cheeks', 'Wearing_Earrings', 'Wearing_Necklace', 'Wearing_Necktie']

# -------------- Identity File Path --------- #
if dataset == "CelebA":
    identity_path = "./data/CelebA/identity_CelebA.txt"
else:
    identity_path = "./data/LFWA/identity_LFWA.txt" # the identity file of LFWA dataset
# To be optimized
attr_nums = [i for i in range(len(all_attrs))] 
attr_loss_weight = [1 for i in range(len(all_attrs))]

selected_attrs = [] 
for num in attr_nums:
    selected_attrs.append(all_attrs[num])

# To solve the sample imbalance called rescaling, If the threshold > m+ /(m+ + m-), treat it as a positive sample. 
attr_threshold = [0.5 for i in range(len(all_attrs))]  

""" Cause worse accuracy result.
sample_csv = pd.read_csv('sample_num.csv')
attr_threshold = (sample_csv['positive sample']/(sample_csv['positive sample'] + sample_csv['negative sample'])).tolist()
"""

# -------------- Tensorboard --------------------- #
use_tensorboard = True

# -------------- Early_stop --------------------- #
early_stop = 500

# -------------- Learning Rate Steps Array --------------------- #
lr_step_array = [200, 300]

# -------------- Experiment Settings --------------------- #
experiment_description = """"""

model_type = "Resnet18" if test_flag else "swin_transformer" #！
batch_size = 100 #！
epochs = 5 if test_flag else 100 #！
learning_rate_large = 1e-2 #！
learning_rate_small = 1e-2 #！
momentum = 0.9
optim_type = 'SGD' # choices=['SGD','Adam']
pretrained = True
loss_type = 'BCE_loss' # choices=['BCE_loss', 'focal_loss']
exp_version = "v1" if test_flag else "v22-2-1" #！
load_model_path = "./model_best.pth.tar"
lamda = 1

face_attribute_model_type = 2 # ！
# -------------- Identity nums --------------------- #
if dataset == "CelebA":
    train_identity_num = 8192
    validation_identity_num = 985
    test_identity_num = 1000
    identity_num = 10177
else:
    # base on dataset to update the num
    train_identity_num = 2732
    validation_identity_num = 2989
    test_identity_num = 2989
    identity_num = 5721

# ------------ weight of branch ------------ #
weight_attribute = 1 # lamda, LossA
weight_global_identity = 0.1 # 0.1 # alpha, LossF
weight_global_contrastive_identity = 0.9 # 1 - weight_global_identity, 1 - alpha LossC
weight_local_identity = 0.3 # beta 1 Lossg
