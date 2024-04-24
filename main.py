import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import shutil

import utils
import training
import testing

import matplotlib.pyplot as plt
import pickle
from config import config
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"cuda" if torch.cuda.is_available() else "cpu"


dir = utils.create_rec_dir()
config_path = f"config.ini"
shutil.copy(config_path, dir)
config = utils.get_config(config_path)


for SNR_model in range(2,3):
    print(f"Train CNN Model with SNR = {SNR_model} dB")
    training.train_model(device, config, dir, SNR=SNR_model, model_idx=0)

# for SNR_model in range(2,8):
#     print(f"Train cALCNN Model with SNR = {SNR_model} dB")
#     training.train_received_pattern(device, config, dir, SNR=SNR_model, model_idx=1)

# for SNR_model in range(2,8):
#     print(f"Train cCNN_cat_feature Model with SNR = {SNR_model} dB")
#     training.train_received_pattern(device, config, dir, SNR=SNR_model, model_idx=2)

for SNR_model in range(2, 3):
    BERs = testing.test_model(device, dir, model_idx=0, SNR_model=SNR_model)

# for SNR_model in range(2, 8):
#     BERs = testing.test_model(device, dir, model_idx=1, SNR_model=SNR_model)

# for SNR_model in range(2, 8):
#     BERs = testing.test_model(device, dir, model_idx=2, SNR_model=SNR_model)



draw_range = range(2, 3)
BERss_CNN = []
for SNR_model in draw_range:
    rec_path = f"{dir}/records/test/CNN_{SNR_model}dB.pkl"
    with open(rec_path, 'rb') as f:
        BERs = pickle.load(f)
        print(BERs)
        BERss_CNN.append(BERs)

# BERss_cALCNN = []
# for SNR_model in draw_range:
#     rec_path = f"{dir}/records/test/cCNN/cALCNN_{SNR_model}dB.pkl"
#     with open(rec_path, 'rb') as f:
#         BERs = pickle.load(f)
#         print(BERs)
#         BERss_cALCNN.append(BERs)

# BERss_cCNN_cat = []
# for SNR_model in draw_range:
#     rec_path = f"{dir}/records/test/cCNN/cCNN_cat_ft_{SNR_model}dB.pkl"
#     with open(rec_path, 'rb') as f:
#         BERs = pickle.load(f)
#         print(BERs)
#         BERss_cCNN_cat.append(BERs)
