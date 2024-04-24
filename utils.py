import os
from datetime import datetime
import torch
from torch.utils.data import Dataset, random_split
import configparser
import pickle
import numpy as np
import matplotlib.pyplot as plt

import dataset as ds

# Get Config
def get_config(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    
    return config

# Hamming Distance
def hamming_distance(seq_1, seq_2):   
    return torch.sum(seq_1 != seq_2).item()



# Split Dataset
def split_dataset(path, rate=0.8):
    dataset = ds.BitDataset(path)
    train_size = int(rate * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    return dataset, train_set, test_set


# Save Model
def save_model(model, path):
    torch.save(model.state_dict(), path)


# Load Model
def load_model(model, path):
    test_model = model
    test_model.load_state_dict(torch.load(path))
    test_model.eval()
    # print(test_model.state_dict())

    return test_model

# Create Record Directory
def create_rec_dir():
    results_path = os.path.join(os.getcwd(), "results")
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    dir = os.path.join(results_path, current_time)
    dir_models = os.path.join(dir, "models")
    dir_models_CNN = os.path.join(dir_models, "CNN")
    dir_models_cCNN = os.path.join(dir_models, "cCNN")

    dir_records = os.path.join(dir, "records")

    dir_records_train = os.path.join(dir_records, "train")
    dir_records_train_CNN = os.path.join(dir_records_train, "CNN")
    dir_records_train_cCNN = os.path.join(dir_records_train, "cCNN")

    dir_records_test = os.path.join(dir_records, "test")
    dir_records_test_CNN = os.path.join(dir_records_test, "CNN")
    dir_records_test_cCNN = os.path.join(dir_records_test, "cCNN")

    dir_datasets = os.path.join(dir, "datasets")
    dir_datasets_test = os.path.join(dir_datasets, "test")  # need append seed number

    dir_figures = os.path.join(dir, "figures")
    
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    os.makedirs(dir)
    os.makedirs(dir_models)
    os.makedirs(dir_models_CNN)
    os.makedirs(dir_models_cCNN)

    os.makedirs(dir_records)

    os.makedirs(dir_records_train)
    os.makedirs(dir_records_train_CNN)
    os.makedirs(dir_records_train_cCNN)
    
    os.makedirs(dir_records_test)
    os.makedirs(dir_records_test_CNN)
    os.makedirs(dir_records_test_cCNN)

    os.makedirs(dir_datasets)
    os.makedirs(dir_datasets_test)

    os.makedirs(dir_figures)

    return dir

def draw_loss(dir, model_idx, SNR_range):
    model_names = ['CNN', 'cALCNN', 'cCNN_cat_ft']
    CNN_losses = []

    for SNR in SNR_range:
        path = f'records/train/loss_{model_names[model_idx]}_{SNR}dB.pkl'
        with open(os.path.join(dir, path), 'rb') as f:
            results = pickle.load(f)
            losses = np.array(results)
            CNN_losses.append(losses)

    CNN_losses = np.array(CNN_losses)

    x_values = range(1, CNN_losses.shape[1]+1)

    for i in range(0, CNN_losses.shape[0]):
        plt.plot(x_values, CNN_losses[i], label=f'{i+2} dB')
    plt.xlabel('Batch Index')
    plt.ylabel('Average Loss')
    plt.title(f'Batch Loss ({model_names[model_idx]})')
    plt.legend()
    plt.show()
