import datetime
from utils import check_make_dir
import torch
time_now = datetime.datetime.now()
class_all = [0, 1, 2, 3, 4, 5, 6, 7, 8] # 0默认为背景类
class_to_keep = [1, 2, 3, 4, 5, 6, 7, 8] # 0不能在这里出现
class_to_remove = []
# data
# data_address
train_map = '../GuiTorch/segmentation_data/train/img/'
train_label = '../GuiTorch/segmentation_data/train/gth/'
val_map = '../GuiTorch/segmentation_data/val/img/'
val_label = '../GuiTorch/segmentation_data/val/gth/'
test_map = '../GuiTorch/segmentation_data/test/img/'
test_label = '../GuiTorch/segmentation_data/test/gth/'
# data num
train_num = None
val_num = None
test_num = None

# data type
map_seffix = '.tif'
label_seffix = '.tif'
# data size
in_h = 256
in_w = 256
# augumentation
augmentation_methods = None
# model

in_channels = 8
out_channels = 9
muti_class = True
model_name = 'UNET'
LOAD_TRAIN_MODEL = False
LOAD_TRAIN_PATH = '11'

LOAD_TEST_PATH = '../UNET_CrossEntropy_200_9_2023-04-21/model/UNET_epo122_viou0.6607510961649072_tacc0.8077823707368984_vacc0.7945582994067942.pth'

# settlement
num_workers = 0
pin_memory = False
gpu = 0

# train
# lr
lr_start = 0.0001
# scheduler
scheduler_name = 'torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.85, patience=20, verbose=True)'
# loss
loss_name = 'CrossEntropy'
loss_stop_epoch = 200
# optimizer
optimizer_name = 'torch.optim.SGD(model.parameters(), lr=lr_start, momentum=0.95, weight_decay=0.0003)'
# others
repeat_times = 1
epoch_min = 1
epoch_max = 200
batch_size = 16

# save
# dirza

save_root = f'{model_name}_{loss_name}_{epoch_max}_{out_channels}_{time_now.date()}'
result_dir = f'{save_root}/result'
model_save_dir = f'{save_root}/model'
tensorboard_log_dir = f'{save_root}/tensorboard'
mask_val_save_dir = f'{save_root}/mask_val_pred'
mask_test_save_dir = f'{save_root}/mask_test_pred'
