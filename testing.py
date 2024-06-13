import torch
from torch.utils.data import DataLoader
import torchmetrics
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

import dataset as ds
import utils
import model as md
# from config import config
import config as cf

def test_decoder(device, config, testset, model, cond, rand):
    batch_size = config['decoder'].getint('batch_size')

    codes = torch.tensor(cf.codes).to(device)
    codes = codes.unsqueeze(0).expand(batch_size, -1, -1)  # Repeat codes for each code_info

    model.eval()
    BER = 0      # Bit Error Rate
    
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, drop_last=True)

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            seqs      = data[0].unsqueeze(1).to(device)
            targets    = data[1].to(device)

            if cond: # Conditional training
                code_info = data[2].to(device)
                if rand:  # Random create one-hot code
                    random_indices = torch.randint(0, len(codes), len(code_info,)).to(device)
                    info = torch.zeros(len(code_info), len(codes)).to(device)
                    info.scatter_(1, random_indices.unsqueeze(1), 1)
                    code_info = info.float()
                else:
                    code_info = torch.all(code_info.unsqueeze(1).eq(codes), dim=2).float()

                predictions = model(seqs, code_info).round().to(device)  # Use Cond_CNN_Decoder
            else:  # Unconditional training
                predictions = model(seqs).round().to(device)  # Use CNN_Decoder

            HD = torch.sum(predictions != targets, dim=1)
            BER += HD.sum()

            # if ((batch_idx+1) % 100) == 0:
            #     batch = (batch_idx+1) * len(seqs)
            #     percentage = (100. * (batch_idx+1) / len(test_loader))
            #     print(f'[{batch:5d} / {len(test_loader.dataset)}] ({percentage:.0f} %)')
                
    BER = BER / (len(test_loader) * batch_size * 12)  # 1 frame has 12 bits
    BER = round(float(BER), 10)


    return BER

def test_model(device, config, dir, code_name, model_idx, SNR_model, test_range, rand=False):
    in_ch = config['decoder'].getint('input_channels')
    out_ch = config['decoder'].getint('output_channels')
    k_size = config['decoder'].getint('kernel_size')

    models = [md.CNN_Decoder(in_ch, out_ch, k_size), 
            md.CCNN_AL_Decoder(in_ch, out_ch, k_size), 
            md.CCNN_FM_Decoder(in_ch, out_ch, k_size),
            md.CNN_2L_Decoder(in_ch, out_ch, k_size, config),
            md.CCNN_AL_2L_Decoder(in_ch, out_ch, k_size, config),
            md.CCNN_FM_2L_Decoder(in_ch, out_ch, k_size, config),
            md.CCNN_joint_Decoder(config),]
    
    model_names = md.model_names

    cond = (model_idx != 0) and (model_idx != 3) and (model_idx != 6)

    if rand:
        records_path = f"{dir}/records/test/{model_names[model_idx]}_{code_name}_{SNR_model}dB_random.pkl"
        model_path = f"{dir}/models/{model_names[model_idx]}_{SNR_model}dB_random.pt"
    else:
        records_path = f"{dir}/records/test/{model_names[model_idx]}_{code_name}_{SNR_model}dB.pkl"
        model_path = f"{dir}/models/{model_names[model_idx]}_{SNR_model}dB.pt"

    model = utils.load_model(models[model_idx], model_path).to(device)
    model.eval()

    BERs = []
    for SNR_data in test_range:
        data_path = f"Dataset/Test/{code_name}/receive_{SNR_data}dB.csv"
        testset = ds.LoadDataset(data_path)
        BER = test_decoder(device, config, testset, model, cond=cond, rand=rand)
        BERs.append(BER)
        print(f"SNR: {SNR_data} dB, BER: {BER}")

    with open(records_path, 'wb') as f:
        pickle.dump(BERs, f)

    print(records_path)

    return BERs

def classify(device, config, dataset, model):
    codes = torch.tensor(cf.codes).to(device)
    onehot_codes = torch.eye(len(codes)).to(device)

    batch_size = config['decoder'].getint('batch_size')
    
    model.eval()

    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    accuracices = 0
    total_correct = 0
    total_samples = 0

    num_classes = len(codes)
    confusion_matrix_metric = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=num_classes).to(device)


    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            seqs = data[0].float().unsqueeze(1).to(device)
            code = data[2].to(device)
            # transfer code to onehot code
            targets = torch.all(code.unsqueeze(1).eq(codes), dim=2).float().to(device)

            predictions = model(seqs).to(device)

            _, predicted_labels = torch.max(predictions, 1)
            predictions = torch.index_select(onehot_codes, 0, predicted_labels)
            correct = (predictions == targets).all(dim=1).sum().item()

            total_correct += correct
            total_samples += len(targets)
            accuracices = total_correct / total_samples

            confusion_matrix_metric.update(predicted_labels, targets.argmax(dim=1))


            # show progress and loss
            # if ((batch_idx+1) % 10) == 0:
            #     batch = (batch_idx+1) * len(seqs)
            #     percentage = (100. * (batch_idx+1) / len(test_loader))
            #     print(f'[{batch:5d} / {len(test_loader.dataset)}] ({percentage:.0f} %)' + f' Accuracy: {accuracices:.6f}')

    cm = confusion_matrix_metric.compute().cpu().numpy()
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    return accuracices, cm

def test_classifier(device, config, dir, code_name, SNR_model, test_range):
    input_channels  = config['classifier'].getint('input_channels')
    output_channels = config['classifier'].getint('output_channels')
    kernel_size     = config['classifier'].getint('kernel_size')

    model_path   = f"{dir}/models/CNN_Classifier_{SNR_model}dB.pt"
    records_path = f"{dir}/records/test/CNN_Classifier_ACC_{code_name}_{SNR_model}dB.pkl"

    # Load model
    model = md.CNN_Classifier(input_channels, output_channels, kernel_size, config).to(device)
    model = utils.load_model(model, model_path).to(device)

    # Test model for each SNR_data in test_range
    ACCs = []
    for SNR_data in test_range:
        dataset_path     = f"Dataset/Test/{code_name}/receive_{SNR_data}dB.csv"

        testset = ds.LoadDataset(dataset_path)
        acc, cm = classify(device, config, testset, model)

        labels = [(2,1,3), (2,1,5), (2,1,7), (2,1,9)]

        plt.figure(figsize=(5, 4))
        # sns.heatmap(cm, annot=True, fmt=".4f", cmap="Blues", xticklabels=cf.codes, yticklabels=cf.codes)
        sns.heatmap(cm, annot=True, fmt=".4f", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix {SNR_model} dB Model {code_name} {SNR_data} dB")
        plt.show()

        print(f"SNR: {SNR_data} dB, ACC: {acc}")
        ACCs.append(acc)

    with open(records_path, 'wb') as f:
        pickle.dump(ACCs, f)

    return ACCs

def classify_and_decode(device, config, testset, decoder_model, classifier_model):
    cond = True
    rand = False
    batch_size = config['decoder'].getint('batch_size')

    codes = torch.tensor(cf.codes).to(device)
    onehot_codes = torch.eye(len(codes)).to(device)

    decoder_model.eval()
    classifier_model.eval()
    BER = 0      # Bit Error Rate
    ACC = 0      # Accuracy
    total_correct = 0
    total_samples = 0
    
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, drop_last=True)

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            seqs    = data[0].unsqueeze(1).to(device)
            targets_decoder = data[1].to(device)
            tragets_classifier = data[2].to(device)
            tragets_classifier = torch.all(tragets_classifier.unsqueeze(1).eq(codes), dim=2).float().to(device)

            # code_info = data[2].to(device)
            # code_info = torch.all(code_info.unsqueeze(1).eq(codes), dim=2).float()

            # First: classify
            pred_code = classifier_model(seqs).to(device)
            _, pred_label = torch.max(pred_code, 1)
            pred_code = torch.index_select(onehot_codes, 0, pred_label)
            correct = (pred_code == tragets_classifier).all(dim=1).sum().item()
            total_correct += correct
            total_samples += len(tragets_classifier)
            ACC = total_correct / total_samples

            # Second: decode
            pred_msg = decoder_model(seqs, pred_code).round().to(device)  # Use Cond_CNN_Decoder

            HD = torch.sum(pred_msg != targets_decoder, dim=1)
            BER += HD.sum()

            # if ((batch_idx+1) % 10) == 0:
            #     batch = (batch_idx+1) * len(seqs)
            #     percentage = (100. * (batch_idx+1) / len(test_loader))
            #     print(f'[{batch:5d} / {len(test_loader.dataset)}] ({percentage:.0f} %)')
                
    BER = BER / (len(test_loader) * batch_size * 12)  # 1 block has 12 bits
    BER = round(float(BER), 10)


    return BER, ACC

def test_classifier_decoder(device, config, dir, code_name, dec_model_idx, SNR_model, test_range):
    in_ch_dec  = config['decoder'].getint('input_channels')
    out_ch_dec = config['decoder'].getint('output_channels')
    k_size_dec = config['decoder'].getint('kernel_size')
    in_ch_clf  = config['classifier'].getint('input_channels')
    out_ch_clf = config['classifier'].getint('output_channels')
    k_size_clf = config['classifier'].getint('kernel_size')

    models = [md.CNN_Decoder(in_ch_dec, out_ch_dec, k_size_dec), 
            md.CCNN_AL_Decoder(in_ch_dec, out_ch_dec, k_size_dec), 
            md.CCNN_FM_Decoder(in_ch_dec, out_ch_dec, k_size_dec),
            md.CNN_2L_Decoder(in_ch_dec, out_ch_dec, k_size_dec, config),
            md.CCNN_AL_2L_Decoder(in_ch_dec, out_ch_dec, k_size_dec, config),
            md.CCNN_FM_2L_Decoder(in_ch_dec, out_ch_dec, k_size_dec, config),
            md.CCNN_joint_Decoder(config),]
    
    model_names = md.model_names

    records_path_BER = f"{dir}/records/test/{model_names[dec_model_idx]}_classifier_BER_{code_name}_{SNR_model}dB.pkl"
    records_path_ACC = f"{dir}/records/test/{model_names[dec_model_idx]}_classifier_ACC_{code_name}_{SNR_model}dB.pkl"

    # Load classifer model
    model_path_clf = f"{dir}/models/CNN_Classifier_{SNR_model}dB.pt"
    model_clf      = md.CNN_Classifier(in_ch_clf, out_ch_clf, k_size_clf, config).to(device)
    model_clf      = utils.load_model(model_clf, model_path_clf).to(device)
    model_clf.eval()

    # Load decoder model
    model_path_dec = f"{dir}/models/{model_names[dec_model_idx]}_{SNR_model}dB.pt"
    model_dec      = utils.load_model(models[dec_model_idx], model_path_dec).to(device)
    model_dec.eval()

    BERs = []
    ACCs = []
    for SNR_data in test_range:
        data_path = f"Dataset/Test/{code_name}/receive_{SNR_data}dB.csv"
        testset = ds.LoadDataset(data_path)
        BER, ACC = classify_and_decode(device, config, testset, model_dec, model_clf)
        BERs.append(BER)
        ACCs.append(ACC)
        print(f"SNR: {SNR_data} dB, ACC: {ACC}, BER: {BER}")

    with open(records_path_BER, 'wb') as f:
        pickle.dump(BERs, f)
    with open(records_path_ACC, 'wb') as f:
        pickle.dump(ACCs, f)

    print(records_path_BER)
    print(records_path_ACC)

    return BERs, ACCs