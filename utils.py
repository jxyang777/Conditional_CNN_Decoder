import os
from datetime import datetime
import torch
from torch.utils.data import Dataset, random_split
import configparser
import pickle
import numpy as np
import matplotlib.pyplot as plt
from plottable import Table
import pandas as pd
import shutil

import dataset as ds
import model as md

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
def create_rec_dir(dir_name):
    results_path = os.path.join(os.getcwd(), "results")
    # current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # dir = os.path.join(results_path, current_time)
    dir             = os.path.join(results_path, dir_name)
    dir_models      = os.path.join(dir, "models")
    dir_models_CNN  = os.path.join(dir_models, "CNN")
    dir_models_cCNN = os.path.join(dir_models, "cCNN")

    dir_records            = os.path.join(dir, "records")
    dir_records_train      = os.path.join(dir_records, "train")
    dir_records_train_cCNN = os.path.join(dir_records_train, "cCNN")
    dir_records_test       = os.path.join(dir_records, "test")
    dir_records_test_cCNN  = os.path.join(dir_records_test, "cCNN")

    dir_figures = os.path.join(dir, "figures")
    
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    if not os.path.exists(dir):
        os.makedirs(dir)
        os.makedirs(dir_models)
        os.makedirs(dir_models_CNN)
        os.makedirs(dir_models_cCNN)
        os.makedirs(dir_records)
        os.makedirs(dir_records_train)
        os.makedirs(dir_records_train_cCNN)
        os.makedirs(dir_records_test)
        os.makedirs(dir_records_test_cCNN)
        os.makedirs(dir_figures)

        default_config_path = "config.ini"
        shutil.copy(default_config_path, dir)

    config = get_config(f"{dir}/config.ini")

    return dir, config

def draw_loss(result_dir, model_idx, SNR_range):
    model_names = md.res_model_names
    losses = []

    for SNR in SNR_range:
        path_record = f'records/train/loss_{model_names[model_idx]}_{SNR}dB.pkl'
        with open(os.path.join(result_dir, path_record), 'rb') as f:
            results = pickle.load(f)
            results = np.array(results)
            losses.append(results)

    losses = np.array(losses)

    x_values = range(1, losses.shape[1]+1)

    for i in SNR_range:
        idx = SNR_range.index(i)
        plt.plot(x_values, losses[idx], label=f'{i} dB')
    plt.xlabel('Batch Index')
    plt.ylabel('Average Loss')
    plt.title(f'Batch Loss ({model_names[model_idx]})')
    plt.legend()
    plt.savefig(f'{result_dir}/figures/loss_{model_names[model_idx]}.png')
    plt.show()

def load_BER(result_dir, model_name, code_name, train_range):
    BERss = []
    for SNR_model in train_range:
        rec_path = f"{result_dir}/records/test/{model_name}_{code_name}_{SNR_model}dB.pkl"
        with open(rec_path, 'rb') as f:
            BERs = pickle.load(f)
            BERss.append(BERs)
    return BERss

def load_BER_combine(result_dir, model_name, code_name, train_range):
    BERss = []
    for SNR_model in train_range:
        rec_path = records_path_BER = f"{result_dir}/records/test/{model_name}_classifier_BER_{code_name}_{SNR_model}dB.pkl"
        with open(rec_path, 'rb') as f:
            BERs = pickle.load(f)
            BERss.append(BERs)
    return BERss

def load_BER_uncoded(code_name):
    BERs_uncoded = []
    with open(f'Dataset/Test/{code_name}/uncoded_BER.pkl', 'rb') as f:
        BERs_uncoded = pickle.load(f)
    return BERs_uncoded

def load_BER_hard(code_name):
    BERs_hard = []
    with open(f'Dataset/Test/{code_name}/hard_BER.pkl', 'rb') as f:
        BERs_hard = pickle.load(f)
    return BERs_hard


def load_ACC(result_dir, code_name, train_range):
    ACCs = []
    for SNR_model in train_range:
        rec_path = f"{result_dir}/records/test/CNN_Classifier_ACC_{code_name}_{SNR_model}dB.pkl"
        with open(rec_path, 'rb') as f:
            ACC = pickle.load(f)
            ACCs.append(ACC)
    return ACCs

def load_diff(result_dir, code_name, train_range, model_name_1, model_name_2):
    diffs = []
    for SNR_model in train_range:
        rec_path = f"{result_dir}/records/test/Diff_{model_name_1}_vs_{model_name_2}_{code_name}_{SNR_model}dB.pkl"
        with open(rec_path, 'rb') as f:
            diff = pickle.load(f)
            diffs.append(diff)
    return diffs

def set_lower_BER(BERs):
    BERs = BERs.copy()
    for i in range(1, len(BERs)):
        if BERs[i] == 0:
            if BERs[i-1] != 1e-6 and BERs[i-1] != 0:
                BERs[i] = 1e-6
            elif BERs[i-1] == 1e-6:
                BERs[i] = 0
    return BERs

def cal_diff(BERs_1, BERs_2):
    diff = []
    for i in range(len(BERs_1)):
        try:
            diff.append((BERs_1[i] - BERs_2[i]) / BERs_1[i])
        except ZeroDivisionError:
            diff.append(1)

    return [f"{round(d*100, 2)}%" for d in diff]

def draw_BER_single(result_dir, BERs_uncoded, BERs_hard, model_idx, test_code, train_range, test_range):
    model_name = md.res_model_names[model_idx]
    colors = ['red', 'orange', 'cyan', 'green', 'blue', 'purple']
    SNR_x = np.array(list(test_range))

    BERss = load_BER(result_dir, model_name, test_code, train_range)

    plt.figure(figsize=(8, 8))
    BERs_uncoded = set_lower_BER(BERs_uncoded)
    plt.semilogy(SNR_x, BERs_uncoded, 'o-', label='uncoded')
    BERs_hard = set_lower_BER(BERs_hard)
    plt.semilogy(SNR_x, BERs_hard, 'o-', label='Viterbi', color='black')

    for i, BERs in enumerate(BERss, start=0):
        print(f"{model_name} {train_range[i]}dB: {BERs}")
        BERs = set_lower_BER(BERs)
        plt.semilogy(SNR_x, BERs, '-', label=f"{model_name} {train_range[i]}dB", color=colors[i])
    plt.legend()
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.ylim(1e-6, 1)
    plt.title(f'BER of {model_name}_{test_code}')
    plt.grid()
    plt.savefig(f"{result_dir}/figures/BER_{model_name}_{test_code}.jpg")
    plt.show()

def draw_BER_single_combine(result_dir, BERs_uncoded, BERs_hard, model_idx, test_code, train_range, test_range):
    # model_name = md.model_names[model_idx]
    # model_name = md.res_model_names[model_idx]
    model_name = 'CCNN_P'
    colors = ['red', 'orange', 'cyan', 'green', 'blue', 'purple']
    SNR_x = np.array(list(test_range))

    BERss = load_BER_combine(result_dir, model_name, test_code, train_range)

    plt.figure(figsize=(8, 8))
    BERs_uncoded = set_lower_BER(BERs_uncoded)
    plt.semilogy(SNR_x, BERs_uncoded, 'o-', label='uncoded')
    BERs_hard = set_lower_BER(BERs_hard)
    plt.semilogy(SNR_x, BERs_hard, 'o-', label='Viterbi', color='black')

    for i, BERs in enumerate(BERss, start=0):
        print(f"{model_name} {train_range[i]}dB: {BERs}")
        BERs = set_lower_BER(BERs)
        plt.semilogy(SNR_x, BERs, '-', label=f"{model_name}_combine {train_range[i]}dB", color=colors[i])
    plt.legend()
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.ylim(1e-6, 1)
    plt.title(f'BER of {model_name}_{test_code}')
    plt.grid()
    plt.savefig(f"{result_dir}/figures/BER_{model_name}_combine_{test_code}.jpg")
    plt.show()

def draw_BER_compare(result_dir, BERs_uncoded, BERs_hard, model_idx_1, model_idx_2, test_code, train_range, test_range):
    res_name_1 = md.res_model_names[model_idx_1]
    res_name_2 = md.res_model_names[model_idx_2]
    model_name_1 = md.model_names[model_idx_1]
    model_name_2 = md.model_names[model_idx_2]
    colors = ['red', 'orange', 'cyan', 'green', 'blue', 'purple']
    SNR_x = list(test_range)

    BERss_1 = load_BER(result_dir, model_name_1, test_code, train_range)
    BERss_2 = load_BER(result_dir, model_name_2, test_code, train_range)

    plt.figure(figsize=(8, 8))
    
    BERs_uncoded = set_lower_BER(BERs_uncoded)
    plt.semilogy(SNR_x, BERs_uncoded, 'o-', label='uncoded')
    BERs_hard = set_lower_BER(BERs_hard)
    plt.semilogy(SNR_x, BERs_hard, 'o-', label='Viterbi', color='black')

    diffs = []

    for i in range(len(BERss_1)):
        BERs_1 = BERss_1[i]
        BERs_2 = BERss_2[i]
        print(f"{res_name_1} {train_range[i]}dB: {BERs_1}")
        print(f"{res_name_2} {train_range[i]}dB: {BERs_2}")
        BERs_1 = set_lower_BER(BERs_1)
        BERs_2 = set_lower_BER(BERs_2)
        diff = cal_diff(BERs_1, BERs_2)
        print(diff)
        with open(f"{result_dir}/records/test/Diff_{res_name_1}_vs_{res_name_2}_{test_code}_{train_range[i]}dB.pkl", 'wb') as f:
            pickle.dump(diff, f)
        plt.semilogy(SNR_x, BERs_1, '-', label=f"{res_name_1} {train_range[i]}dB", color=colors[i])
        plt.semilogy(SNR_x, BERs_2, '--', label=f"{res_name_2} {train_range[i]}dB", color=colors[i])

    plt.legend()
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.ylim(1e-6, 1)
    plt.title(f'BER of {res_name_1} vs {res_name_2} {test_code}')
    plt.grid()
    plt.savefig(f"{result_dir}/figures/BER_{res_name_1}_vs_{res_name_2}_{test_code}.jpg")
    plt.show()

def draw_BER_compare_multi(result_dir, BERs_uncoded, BERs_hard, model_idx_1, model_idx_t, test_code, train_range, test_range):
    model_name_1 = md.model_names[model_idx_1]  # CNN
    model_name_t = md.model_names[model_idx_t]  # CCNN_FM_T
    model_name_p = md.model_names[model_idx_t]  # CCNN_FM_P
    model_name_j = md.model_names[6]            # CCNN_FM_J
    res_name_1 = md.res_model_names[model_idx_1]
    res_name_t = md.res_model_names[model_idx_t]
    res_name_p = 'CCNN_P'
    res_name_j = 'CCNN_ J'
    colors = ['red', 'orange', 'cyan', 'green', 'blue', 'purple']
    SNR_x = list(test_range)

    BERss_1 = load_BER(result_dir, model_name_1, test_code, train_range)
    BERss_t = load_BER(result_dir, model_name_t, test_code, train_range)
    BERss_p = load_BER_combine(result_dir, model_name_p, test_code, train_range)
    BERss_j = load_BER(result_dir, model_name_j, test_code, train_range)

    plt.figure(figsize=(8, 8))
    
    BERs_uncoded = set_lower_BER(BERs_uncoded)
    plt.semilogy(SNR_x, BERs_uncoded, '-', label='uncoded')
    BERs_hard = set_lower_BER(BERs_hard)
    plt.semilogy(SNR_x, BERs_hard, '-', label='Viterbi', color='black')

    for i in range(len(BERss_1)):
        BERs_1 = BERss_1[i]
        BERs_t = BERss_t[i]
        BERs_p = BERss_p[i]
        BERs_j = BERss_j[i]
        print(f"{res_name_1} {train_range[i]}dB: {BERs_1}")
        print(f"{res_name_t} {train_range[i]}dB: {BERs_t}")
        print(f"{res_name_p} {train_range[i]}dB: {BERs_p}")
        print(f"{res_name_j} {train_range[i]}dB: {BERs_j}")

        BERs_1 = set_lower_BER(BERs_1)
        BERs_t = set_lower_BER(BERs_t)
        BERs_p = set_lower_BER(BERs_p)
        BERs_j = set_lower_BER(BERs_j)
        print(cal_diff(BERs_1, BERs_t))
        print(cal_diff(BERs_1, BERs_p))
        print(cal_diff(BERs_1, BERs_j))
        plt.semilogy(SNR_x, BERs_1, '-', label=f"{res_name_1} {train_range[i]}dB", color=colors[0], marker='x')
        plt.semilogy(SNR_x, BERs_t, '-', label=f"{res_name_t} {train_range[i]}dB", color=colors[1], marker='o')
        plt.semilogy(SNR_x, BERs_p, '-', label=f"{res_name_p} {train_range[i]}dB", color=colors[4], marker='>')
        plt.semilogy(SNR_x, BERs_j, '-', label=f"{res_name_j} {train_range[i]}dB", color=colors[5], marker='*')
    plt.legend()
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.ylim(1e-6, 1)
    plt.title(f'BER with 2dB {test_code}')
    plt.grid()
    plt.savefig(f"{result_dir}/figures/BER_{res_name_1}_vs_{res_name_t}_vs_{res_name_p}_{test_code}.jpg")
    plt.show()

def draw_ACC(result_dir, code_names, train_range, test_range):
    fig, axes = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
    
    for i, ax in enumerate(axes.flat):
        ACCs = load_ACC(result_dir, code_names[i], train_range)
        df = pd.DataFrame(ACCs, index=[f"{SNR} dB" for SNR in train_range], columns=[f"{SNR} dB" for SNR in test_range])
        df = (df*100).round(2).astype(str) + '%'
        
        table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        ax.axis('off')
        ax.set_title(f'{code_names[i]}')
    
    plt.tight_layout()
    plt.savefig(f"{result_dir}/figures/ACC_Classifier.jpg")
    plt.show()

def draw_diff(result_dir, code_names, model_name_1, model_name_2, train_range, test_range):
    fig, axes = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
    
    for i, ax in enumerate(axes.flat):
        ACCs = load_diff(result_dir, code_names[i], model_name_1, model_name_2, train_range)
        df = pd.DataFrame(ACCs, index=[f"{SNR} dB" for SNR in train_range], columns=[f"{SNR} dB" for SNR in test_range])
        df = (df*100).round(2).astype(str) + '%'
        
        table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        ax.axis('off')
        ax.set_title(f'{code_names[i]}')
    
    plt.tight_layout()
    plt.savefig(f"{result_dir}/figures/Diff_code_{model_name_1}_vs_{model_name_2}.jpg")
    plt.show()