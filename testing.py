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

def test_decoder(device, config, testset, model, cond, rand, gumbel, cal_cm):
    batch_size = config['decoder'].getint('batch_size')

    codes = torch.tensor(cf.codes).to(device)
    codes_batch = codes.unsqueeze(0).expand(batch_size, -1, -1)  # Repeat codes for each code_info
    codes_onehot = torch.eye(len(codes)).to(device)

    model.eval()
    BER = 0      # Bit Error Rate
    ACC = 0      # Accuracy
    total_correct = 0
    total_samples = 0
    
    # Create DataLoader
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, drop_last=True)

    if cal_cm:
        num_classes = len(codes)
        confusion_matrix_metric = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=num_classes).to(device)

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            # Load data
            codeword    = data[0].unsqueeze(1).to(device)
            targets_msg = data[1].to(device)
            code_info = data[2].to(device)

            if cond: 
                # Conditional training
                if rand:  
                    # Random create one-hot code
                    random_indices = torch.randint(0, len(codes_batch), len(code_info,)).to(device)
                    info = torch.zeros(len(code_info), len(codes_batch)).to(device)
                    info.scatter_(1, random_indices.unsqueeze(1), 1)
                    code_info = info.float()
                else:
                    # Transfer code type to 'onehot' code_info
                    code_info = torch.all(code_info.unsqueeze(1).eq(codes_batch), dim=2).float()

                pred_msg = model(codeword, code_info).to(device)  # Use Cond_CNN_Decoder
            else:  
                # Unconditional training
                prediction = model(codeword)
                if not gumbel:
                    pred_msg = prediction.to(device)
                else:
                    pred_msg = prediction[0].to(device)
                    if cal_cm:
                        pred_class_soft = prediction[1].to(device)
                        pred_class_hard = prediction[2].to(device)

                        _, max_idx = torch.max(pred_class_soft, 1)
                        pred_class_hard = torch.index_select(codes_onehot, 0, max_idx)
                        target_class = torch.all(code_info.unsqueeze(1).eq(codes), dim=2).float().to(device)

                        correct_cnt = (pred_class_hard == target_class).all(dim=1).sum().item()

                        total_correct += correct_cnt
                        total_samples += len(target_class)
                        ACC = total_correct / total_samples
                        confusion_matrix_metric.update(max_idx, target_class.argmax(dim=1))

            HD = torch.sum(pred_msg.round() != targets_msg, dim=1)
            BER += HD.sum()

            # if ((batch_idx+1) % 100) == 0:
            #     batch = (batch_idx+1) * len(seqs)
            #     percentage = (100. * (batch_idx+1) / len(test_loader))
            #     print(f'[{batch:5d} / {len(test_loader.dataset)}] ({percentage:.0f} %)')
                
    BER = BER / (len(test_loader) * batch_size * 12)  # 1 frame has 12 bits
    BER = round(float(BER), 10)

    if (gumbel and cal_cm):
        CM = confusion_matrix_metric.compute().cpu().numpy()
        CM = CM.astype('float') / CM.sum(axis=1)[:, np.newaxis]
        return BER, ACC, CM

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
            md.CCNN_Joint_Decoder(config),
            md.CCNN_Joint_Gumbel_Decoder(config),
            md.CCNN_Joint_2_Decoder(config),
            md.CCNN_Joint_Gumbel_2_Decoder(config),]
    
    model_names = md.model_names

    cal_cm = (code_name=="code_mix")

    match model_idx:
        case 0 | 3 | 6 | 7 | 8 | 9:
            cond = False
        case _:
            cond = True
    
    match model_idx:
        case 7 | 9:
            gumbel = True
        case _:
            gumbel = False

    if rand:
        test_BER_path = f"{dir}/records/test/{model_names[model_idx]}_{code_name}_{SNR_model}dB_random.pkl"
        model_path = f"{dir}/models/{model_names[model_idx]}_{SNR_model}dB_random.pt"
    else:
        test_BER_path = f"{dir}/records/test/{model_names[model_idx]}_{code_name}_{SNR_model}dB.pkl"
        model_path = f"{dir}/models/{model_names[model_idx]}_{SNR_model}dB.pt"

    if cal_cm:
        test_ACC_path = f"{dir}/records/test/ACC_{model_names[model_idx]}_{code_name}_{SNR_model}dB.pkl"
        # test_CM_path = f"{dir}/records/test/CM_{model_names[model_idx]}_{code_name}_{SNR_model}dB.pkl"

    model = utils.load_model(models[model_idx], model_path).to(device)
    model.eval()

    BERs = []
    if cal_cm:
        ACCs = []

    for SNR_data in test_range:
        data_path = f"Dataset/Test/{code_name}/receive_{SNR_data}dB.csv"

        testset = ds.LoadDataset(data_path)
        ret = test_decoder(device, config, testset, model, cond=cond, rand=rand, gumbel=gumbel, cal_cm=cal_cm)
        
        if (gumbel and cal_cm):
            BER = ret[0]
            BERs.append(BER)
            print(f"SNR: {SNR_data} dB, BER: {BER}")

            ACC = ret[1]
            ACCs.append(ACC)
            print(f"SNR: {SNR_data} dB, ACC: {ACC}")

            CM = ret[2]
            utils.draw_confus_matrix(CM, SNR_model, code_name, SNR_data)
        else:
            BER = ret
            BERs.append(BER)
            print(f"SNR: {SNR_data} dB, BER: {BER}")

    with open(test_BER_path, 'wb') as f:
        pickle.dump(BERs, f)
    print(test_BER_path)

    if cal_cm:
        with open(test_ACC_path, 'wb') as f:
            pickle.dump(ACCs, f)
        print(test_ACC_path)

        return BERs, ACCs


    return BERs

def classify(device, config, dataset, model, cal_cm):
    codes = torch.tensor(cf.codes).to(device)
    codes_onehot = torch.eye(len(codes)).to(device)

    batch_size = config['decoder'].getint('batch_size')
    
    model.eval()

    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    ACC = 0
    total_correct = 0
    total_samples = 0

    if cal_cm:
        num_classes = len(codes)
        confusion_matrix_metric = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=num_classes).to(device)


    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            codeword = data[0].float().unsqueeze(1).to(device)
            code_info = data[2].to(device)
            # transfer code to onehot code
            target_class = torch.all(code_info.unsqueeze(1).eq(codes), dim=2).float().to(device)

            pred_class_soft = model(codeword).to(device)

            _, max_idx = torch.max(pred_class_soft, 1)
            pred_class_hard = torch.index_select(codes_onehot, 0, max_idx)
            correct_cnt = (pred_class_hard == target_class).all(dim=1).sum().item()

            total_correct += correct_cnt
            total_samples += len(target_class)
            ACC = total_correct / total_samples

            if cal_cm:
                confusion_matrix_metric.update(max_idx, target_class.argmax(dim=1))

    if cal_cm:
        cm = confusion_matrix_metric.compute().cpu().numpy()
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        return ACC, cm

    return ACC

def test_classifier(device, config, dir, code_name, SNR_model, test_range):
    # If code_name is "code_mix", then calculate confusion matrix
    cal_cm = (code_name=="code_mix")

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
        classify_results = classify(device, config, testset, model, cal_cm)
        
        if not cal_cm:
            ACC = classify_results
        if cal_cm:
            ACC = classify_results[0]
            cm = classify_results[1]
            utils.draw_confus_matrix(cm, SNR_model, code_name, SNR_data)

        print(f"SNR: {SNR_data} dB, ACC: {ACC}")
        ACCs.append(ACC)

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
            md.CCNN_Joint_Decoder(config),]
    
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