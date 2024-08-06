import os
from datetime import datetime
import torch
from torch.utils.data import Dataset, random_split
import configparser
import pickle
import numpy as np
import matplotlib.pyplot as plt
import shutil
import seaborn as sns
import pandas as pd
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

def draw_single_decoder_loss(result_dir, model_idx, SNR_range):
    model_names = md.model_names
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
    plt.ylabel('Loss')
    plt.title(f'Batch Loss ({md.res_model_names[model_idx]})')
    plt.legend()
    plt.savefig(f'{result_dir}/figures/loss_{model_names[model_idx]}.png')
    plt.show()

def draw_multiple_decoder_loss(result_dir, model_idx, SNR_range):
    def get_losses(result_dir, model_idx, SNR_range):
        losses = []
        for SNR in SNR_range:
            path_record = f'records/train/loss_{md.model_names[model_idx]}_{SNR}dB.pkl'
            with open(os.path.join(result_dir, path_record), 'rb') as f:
                results = pickle.load(f)
                results = np.array(results)
                losses.append(results)
        x = range(1, len(losses[0])+1)
        y = np.array(losses)
        return x, y
    
    def create_subplots(ax, x_values, losses, title):
        for i in SNR_range:
            idx = SNR_range.index(i)
            ax.plot(x_values, losses[idx], label=f'{i} dB')
        ax.set_xlabel('Batch Index')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.legend()
        return ax
    
    model_names = md.model_names
    model_idxs = [3,5,6,7,8,9]

    fig, axs = plt.subplots(2, 3, figsize=(10, 8))
    for ax, model_idx in zip(axs.flat, model_idxs):
        x, y = get_losses(result_dir, model_idx, SNR_range)
        title = f'Batch Loss ({md.res_model_names[model_idx]})'
        create_subplots(ax, x, y, title)

    plt.tight_layout()
    plt.savefig(f'{result_dir}/figures/loss_decoder.png')
    plt.show()

def draw_classifier_loss(result_dir, SNR_range):
    losses = []

    for SNR in SNR_range:
        path_record = f'records/train/CNN_Classifier_LOSS_code_mix_{SNR}.pkl'
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
    plt.ylabel('Loss')
    plt.title(f'Batch Loss (CCNN_P classifier)')
    plt.legend()
    plt.savefig(f'{result_dir}/figures/loss_CCNN_P_classifier_{SNR_range[0]}-{SNR_range[-1]}dB.png')
    plt.show()

def draw_gumbel_classifier_loss(result_dir, SNR_range, gumbel_num):
    losses = []

    for SNR in SNR_range:
        if gumbel_num == 1:
            path_record = f'records/train/loss_clas_CCNN_Joint_Gumbel_{SNR}dB.pkl'
            path_img = f'{result_dir}/figures/loss_CCNN_Joint_Gumbel_classifier_{SNR_range[0]}-{SNR_range[-1]}dB.png'
            title = 'Batch Loss (CCNN_G classifier)'
        elif gumbel_num == 2:
            path_record = f'records/train/loss_clas_CCNN_Joint_Gumbel_2_{SNR}dB.pkl'
            path_img = f'{result_dir}/figures/loss_CCNN_Joint_Gumbel_2_classifier_{SNR_range[0]}-{SNR_range[-1]}dB.png'
            title = 'Batch Loss (CCNN_G2 classifier)'
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
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.savefig(path_img)
    plt.show()

def draw_multiple_classifier_loss(result_dir, SNR_range):
    def get_losses(result_dir, SNR_range, gumbel):
        losses = []
        for SNR in SNR_range:
            if gumbel==0:
                path_record = f'records/train/CNN_Classifier_LOSS_code_mix_{SNR}.pkl'
                title = 'Batch Loss (CCNN_P classifier)'
            elif gumbel==1:
                path_record = f'records/train/loss_clas_CCNN_Joint_Gumbel_{SNR}dB.pkl'
                title = 'Batch Loss (CCNN_G1 classifier)'
            elif gumbel==2:
                path_record = f'records/train/loss_clas_CCNN_Joint_Gumbel_2_{SNR}dB.pkl'
                title = 'Batch Loss (CCNN_G2 classifier)'

            with open(os.path.join(result_dir, path_record), 'rb') as f:
                results = pickle.load(f)
                results = np.array(results)
                losses.append(results)
        x = range(1, len(losses[0])+1)
        y = np.array(losses)
        return x, y, title

    def create_subplots(ax, x_values, losses, title):
        for i in SNR_range:
            idx = SNR_range.index(i)
            ax.plot(x_values, losses[idx], label=f'{i} dB')
        ax.set_xlabel('Batch Index')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.legend()
        return ax
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for ax, gumbel in zip(axs.flat, [0,1,2]):
        x, y, title = get_losses(result_dir, SNR_range, gumbel)
        create_subplots(ax, x, y, title)

    plt.tight_layout()
    plt.savefig(f'{result_dir}/figures/loss_classifier.png')
    plt.show()

def draw_confus_matrix(cm, SNR_model, code_name, SNR_data):
    labels = [(2,1,3), (2,1,5), (2,1,7), (2,1,9)]
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt=".4f", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix {SNR_model} dB Model {code_name} {SNR_data} dB")
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
            if BERs[i-1] != 5e-7 and BERs[i-1] != 0:
                BERs[i] = 5e-7
            elif BERs[i-1] == 5e-7:
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

def draw_BER_single(result_dir, model_idx, test_code, train_range, test_range, train_code):
    BERs_uncoded = load_BER_uncoded(train_code)
    BERs_hard = load_BER_hard(train_code)

    model_name = md.model_names[model_idx]
    res_name = md.res_model_names[model_idx]

    colors = ['red', 'orange', 'cyan', 'green', 'blue', 'purple']

    SNR_x = list(test_range)

    BERss = load_BER(result_dir, model_name, test_code, train_range)

    plt.figure(figsize=(8, 8))
    BERs_uncoded = set_lower_BER(BERs_uncoded)
    plt.semilogy(SNR_x, BERs_uncoded, '--', label='uncoded')
    BERs_hard = set_lower_BER(BERs_hard)
    plt.semilogy(SNR_x, BERs_hard, '-.', label='Viterbi', color='black')

    for i in range(len(BERss)):
        BERs = BERss[i]
        print(f"{res_name} {train_range[i]}dB: {BERs}")
        BERs = set_lower_BER(BERs)
        if i==0:
            plt.semilogy(SNR_x, BERs, '-', label=f"{res_name} {train_range[i]}dB", color=colors[i], marker='o')
        else:
            plt.semilogy(SNR_x, BERs, '-', label=f"{res_name} {train_range[i]}dB", color=colors[i])
    
    plt.legend(loc='best', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.ylim(1e-6, 0.1)
    plt.grid()
    plt.title(f'BER of {res_name} model tested on {test_code} dataset')
    if len(train_range) == 1:
        plt.savefig(f"{result_dir}/figures/BER_{res_name}_SNR{train_range[0]}.jpg")
    else:
        plt.savefig(f"{result_dir}/figures/BER_{res_name}_SNR{train_range[0]}~{train_range[-1]}.jpg")
    plt.show()

def draw_BER_single_combine(result_dir, test_code, train_range, test_range, train_code):
    BERs_uncoded = load_BER_uncoded(train_code)
    BERs_hard = load_BER_hard(train_code)

    model_idx = 5
    model_name = md.model_names[model_idx]
    res_name = 'CCNN_P'

    colors = ['red', 'orange', 'cyan', 'green', 'blue', 'purple']

    SNR_x = list(test_range)

    BERss = load_BER_combine(result_dir, model_name, test_code, train_range)

    plt.figure(figsize=(8, 8))
    BERs_uncoded = set_lower_BER(BERs_uncoded)
    plt.semilogy(SNR_x, BERs_uncoded, '--', label='uncoded')
    BERs_hard = set_lower_BER(BERs_hard)
    plt.semilogy(SNR_x, BERs_hard, '-.', label='Viterbi', color='black')

    for i in range(len(BERss)):
        BERs = BERss[i]
        print(f"{res_name} {train_range[i]}dB: {BERs}")
        BERs = set_lower_BER(BERs)
        if i==0:
            plt.semilogy(SNR_x, BERs, '-', label=f"{res_name} {train_range[i]}dB", color=colors[i], marker='o')
        else:
            plt.semilogy(SNR_x, BERs, '-', label=f"{res_name} {train_range[i]}dB", color=colors[i])
    
    plt.legend(loc='best', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.ylim(1e-6, 0.1)
    plt.title(f'BER with 2dB {test_code}')
    plt.grid()
    if len(train_range) == 1:
        plt.savefig(f"{result_dir}/figures/BER_CCNN_P_SNR{train_range[0]}.jpg")
    else:
        plt.savefig(f"{result_dir}/figures/BER_CCNN_P_SNR{train_range[0]}~{train_range[-1]}.jpg")
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

def draw_BER_compare_multi(result_dir, model_idx_base, model_idx_independent, test_code, train_range, test_range, train_code):
    BERs_uncoded = load_BER_uncoded(train_code)
    BERs_hard = load_BER_hard(train_code)
    
    model_name_b = md.model_names[model_idx_base]    # CNN (baseline)
    model_name_t = md.model_names[model_idx_independent]     # CCNN_FM_T (true)
    model_name_p = md.model_names[model_idx_independent]     # CCNN_FM_P (predict)
    model_name_j = md.model_names[6]                 # CCNN_FM_J (joint)
    model_name_g = md.model_names[7]                 # CCNN_FM_G (joint+gumbel)
    model_name_j2 = md.model_names[8]                # CCNN_FM_J2 (joint_2)
    model_name_g2 = md.model_names[9]                # CCNN_FM_G2 (joint+gumbel_2)

    res_name_1 = md.res_model_names[model_idx_base]
    res_name_t = md.res_model_names[model_idx_independent]
    res_name_p = 'CCNN_P'
    res_name_j = md.res_model_names[6]
    res_name_g = md.res_model_names[7]
    res_name_j2 = md.res_model_names[8]
    res_name_g2 = md.res_model_names[9]

    colors = ['red', 'orange', 'cyan', 'green', 'blue', 'purple', 'brown']
    SNR_x = list(test_range)

    BERss_b  = load_BER(result_dir, model_name_b, test_code, train_range)
    BERss_t  = load_BER(result_dir, model_name_t, test_code, train_range)
    BERss_p  = load_BER_combine(result_dir, model_name_p, test_code, train_range)
    BERss_j  = load_BER(result_dir, model_name_j, test_code, train_range)
    BERss_g  = load_BER(result_dir, model_name_g, test_code, train_range)
    BERss_j2 = load_BER(result_dir, model_name_j2, test_code, train_range)
    BERss_g2 = load_BER(result_dir, model_name_g2, test_code, train_range)

    plt.figure(figsize=(8, 8))
    
    BERs_uncoded = set_lower_BER(BERs_uncoded)
    plt.semilogy(SNR_x, BERs_uncoded, '--', label='uncoded')
    BERs_hard = set_lower_BER(BERs_hard)
    plt.semilogy(SNR_x, BERs_hard, '-.', label='Viterbi', color='black')

    for i in range(len(BERss_b)):
        BERs_1  = BERss_b[i]
        BERs_t  = BERss_t[i]
        BERs_p  = BERss_p[i]
        BERs_j  = BERss_j[i]
        BERs_g  = BERss_g[i]
        BERs_j2 = BERss_j2[i]
        BERs_g2 = BERss_g2[i]

        print(f"{res_name_1} {train_range[i]}dB: {BERs_1}")
        print(f"{res_name_t} {train_range[i]}dB: {BERs_t}")
        print(f"{res_name_p} {train_range[i]}dB: {BERs_p}")
        print(f"{res_name_j} {train_range[i]}dB: {BERs_j}")
        print(f"{res_name_g} {train_range[i]}dB: {BERs_g}")
        print(f"{res_name_j2} {train_range[i]}dB: {BERs_j2}")
        print(f"{res_name_g2} {train_range[i]}dB: {BERs_g2}", end='\n\n')

        BERs_1 = set_lower_BER(BERs_1)
        BERs_t = set_lower_BER(BERs_t)
        BERs_p = set_lower_BER(BERs_p)
        BERs_j = set_lower_BER(BERs_j)
        BERs_g = set_lower_BER(BERs_g)
        BERs_j2 = set_lower_BER(BERs_j2)
        BERs_g2 = set_lower_BER(BERs_g2)

        print(f"Compare CNN vs CCNN_T: {cal_diff(BERs_1, BERs_t)}")
        print(f"Compare CNN vs CCNN_P: {cal_diff(BERs_1, BERs_p)}")
        print(f"Compare CNN vs CCNN_J1: {cal_diff(BERs_1, BERs_j)}")
        print(f"Compare CNN vs CCNN_J2: {cal_diff(BERs_1, BERs_j2)}")
        print(f"Compare CNN vs CCNN_G1: {cal_diff(BERs_1, BERs_g)}")
        print(f"Compare CNN vs CCNN_G2: {cal_diff(BERs_1, BERs_g2)}")
        print(f"Compare CNN vs uncoded: {cal_diff(BERs_1, BERs_uncoded)}")
        print(f"Compare CNN vs hard: {cal_diff(BERs_1, BERs_hard)}")

        plt.semilogy(SNR_x, BERs_1, '-', label=f"{res_name_1} {train_range[i]}dB", color=colors[0], marker='x')
        plt.semilogy(SNR_x, BERs_t, '-', label=f"{res_name_t} {train_range[i]}dB", color=colors[1], marker='o')
        plt.semilogy(SNR_x, BERs_p, '-', label=f"{res_name_p} {train_range[i]}dB", color=colors[4], marker='>')
        plt.semilogy(SNR_x, BERs_j, '-', label=f"{res_name_j} {train_range[i]}dB", color=colors[5], marker='*')
        plt.semilogy(SNR_x, BERs_j2, '-', label=f"{res_name_j2} {train_range[i]}dB", color=colors[3], marker='d')
        plt.semilogy(SNR_x, BERs_g, '-', label=f"{res_name_g} {train_range[i]}dB", color=colors[2], marker='s')
        plt.semilogy(SNR_x, BERs_g2, '-', label=f"{res_name_g2} {train_range[i]}dB", color=colors[6], marker='^')
    
    plt.legend(loc='best', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.ylim(1e-6, 0.1)
    plt.title(f'BER of 2dB model test on {test_code} testset')
    plt.grid()
    if len(train_range) == 1:
        plt.savefig(f"{result_dir}/figures/BER_compare_SNR{train_range[0]}.jpg")
    else:
        plt.savefig(f"{result_dir}/figures/BER_compare_SNR{train_range[0]}~{train_range[-1]}.jpg")
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