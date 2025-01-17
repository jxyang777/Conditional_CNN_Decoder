import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle

import utils
import config as cf
import dataset as ds
import model as md
from config import config

def train(device, config, dataset, model, model_path, cond, rand, gumbel):
    torch.autograd.set_detect_anomaly(True)

    codes = torch.tensor(cf.codes).to(device)
    
    epochs         = config['decoder'].getint('epochs')
    batch_size     = config['decoder'].getint('batch_size')
    lr             = config['decoder'].getfloat('learning_rate')
    loss_fn_dec  = config['decoder']['loss_func']
    tau = 1
    tau_min = 0.1
    tau_decay = 0.99

    model.train()
    if loss_fn_dec == 'BCE':
        loss_fn_dec = nn.BCELoss()
    elif loss_fn_dec == 'MSE':
        loss_fn_dec = nn.MSELoss()
    loss_func_clas = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses_dec = []
    losses_clas = []
    BER_best = 1
    epoch = 0

    while True:
        epoch += 1
        BER = 0
        tau = max(tau_min, tau * tau_decay)
        print(f"Epoch {epoch} , tau: {tau}")
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        for batch_idx, data in enumerate(train_loader):

            codeword        = data[0].unsqueeze(1).to(device)
            target_msg      = data[1].to(device)
            target_codetype = data[2].to(device)
            # transfer code type to 'onehot' code class
            target_class = torch.all(target_codetype.unsqueeze(1).eq(codes), dim=2).float().to(device)

            optimizer.zero_grad()

            if cond: # Independent training
                code_info = data[2].to(device)
                if rand:  
                    # random create 'onehot' code_info
                    random_indices = torch.randint(0, len(ds.trellises_info), (len(code_info),)).to(device)
                    info = torch.zeros(len(code_info), len(ds.trellises_info)).to(device)
                    info.scatter_(1, random_indices.unsqueeze(1), 1)
                    code_info = info.float()
                else:
                    # Transfer code type to 'onehot' code_info
                    code_info = torch.all(code_info.unsqueeze(1).eq(codes), dim=2).float()

                pred_msg = model(codeword, code_info).to(device)
            else:    # Joint training
                if not gumbel:
                    prediction = model(codeword)
                    pred_msg = prediction.to(device)
                else:
                    prediction = model(codeword, tau)
                    pred_msg   = prediction[0].to(device)
                    pred_class_soft = prediction[1].to(device)
                
            # Calculate Hamming Distance
            HD = torch.sum(pred_msg.round() != target_msg, dim=1)
            BER += HD.sum()

            if gumbel:
                loss_clas = loss_func_clas(pred_class_soft, target_class)
                losses_clas.append(loss_clas.item())
                loss_clas.backward(retain_graph=True)

            loss_dec = loss_fn_dec(pred_msg, target_msg)
            losses_dec.append(loss_dec.item())
            loss_dec.backward()
            optimizer.step()


            # show progress and loss
            # if ((batch_idx+1) % 20) == 0:
            #     batch = (batch_idx+1) * len(seqs)
            #     percentage = (100. * (batch_idx+1) / len(train_loader))
            #     print(f'Epoch {epoch}: [{batch:5d} / {len(train_loader.dataset)}] ({percentage:.0f} %)' + f'  Loss: {loss.item():.6f}')

        # Calculate BER per epoch
        BER = BER / (len(train_loader.dataset) * 12)
        print(f"Epoch {epoch} , BER: {BER}")

        if BER < BER_best:
            BER_best = BER
            utils.save_model(model, model_path)
        
        if epoch==epochs:
            if not gumbel:
                return losses_dec
            else:
                return losses_dec, losses_clas


def train_model(device, config, dir, code_name, SNR, model_idx, rand=False):
    in_ch = config['decoder'].getint('input_channels')
    out_ch = config['decoder'].getint('output_channels')
    k_size = config['decoder'].getint('kernel_size')

    model_names = md.model_names
    models = [md.CNN_Decoder(in_ch, out_ch, k_size), 
            md.CCNN_AL_Decoder(in_ch, out_ch, k_size), 
            md.CCNN_FM_Decoder(in_ch, out_ch, k_size),
            md.CNN_2L_Decoder(in_ch, out_ch, k_size, config),
            md.CCNN_AL_2L_Decoder(in_ch, out_ch, k_size, config),
            md.CCNN_FM_2L_Decoder(in_ch, out_ch, k_size, config),
            md.CCNN_Joint_Decoder(config),
            md.CCNN_Joint_Gumbel_Decoder(config),
            md.CCNN_Joint_2_Decoder(config),
            md.CCNN_Joint_Gumbel_2_Decoder(config)]
    
    # Select model by model_idx
    model = models[model_idx].to(device)

    # Whether to apply conditional training and gumbel softmax
    # cond = (model_idx != 0) and (model_idx != 3) and (model_idx != 6) and (model_idx != 7)
    # gumbel = (model_idx == 7)
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

    # Set record path
    model_path   = f"{dir}/models/{model_names[model_idx]}_{SNR}dB.pt"
    records_path = f"{dir}/records/train/loss_{model_names[model_idx]}_{SNR}dB.pkl"
    dataset_path = f"Dataset/Train/{code_name}/receive_{SNR}dB.csv"

    # Load dataset
    dataset   = ds.LoadDataset(dataset_path)

    # Train model
    losses = train(device, config, dataset, model, model_path, cond=cond, rand=rand, gumbel=gumbel)
    if not gumbel:
        losses_dec = losses
    else:
        losses_dec, losses_clas = losses

    # Save decoder training loss
    with open(records_path, 'wb') as f:
        print(f"Saving records: {records_path}")
        pickle.dump(losses_dec, f)

    if gumbel:
        # Save classifier training loss
        with open(records_path.replace('loss', 'loss_clas'), 'wb') as f:
            print(f"Saving records: {records_path.replace('loss', 'loss_clas')}")
            pickle.dump(losses_clas, f)

def classify(device, config, dataset, model, model_path):
    codes = torch.tensor(cf.codes).to(device)
    onehot_codes = torch.eye(len(codes)).to(device)
    
    epochs     = config['classifier'].getint('epochs')
    batch_size = config['classifier'].getint('batch_size')
    lr         = config['classifier'].getfloat('learning_rate')

    model.train()

    loss_fn   = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

    losses = []
    accuracices = []
    epoch = 0

    while True:
        epoch += 1

        trainset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        total_correct = 0
        total_samples = 0
        acc = 0
        for batch_idx, data in enumerate(trainset_loader):
            optimizer.zero_grad()

            seqs = data[0].float().unsqueeze(1).to(device)
            code = data[2].to(device)
            # transfer code to onehot code
            targets = torch.all(code.unsqueeze(1).eq(codes), dim=2).float().to(device)

            predictions = model(seqs).to(device)

            loss = loss_fn(predictions, targets)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            
            _, predicted_labels = torch.max(predictions, 1)
            predictions = torch.index_select(onehot_codes, 0, predicted_labels)
            correct = (predictions == targets).all(dim=1).sum().item()

            total_correct += correct
            total_samples += len(targets)
            acc = total_correct/total_samples

            # show progress and loss
            if ((batch_idx+1) % 100) == 0:
                batch = (batch_idx+1) * len(seqs)
                percentage = (100. * (batch_idx+1) / len(trainset_loader))
                print(f'Epoch {epoch}: [{batch:5d} / {len(trainset_loader.dataset)}] ({percentage:.0f} %)' + f'  Loss: {loss.item():.6f}  Accuracy: {acc:.6f}')
        
        accuracices.append(acc)
        if epoch==epochs:
            utils.save_model(model, model_path)
            return losses, accuracices
    
def train_classifier(device, config, dir, code_name, SNR_model):
    in_ch = config['classifier'].getint('input_channels')
    out_ch = config['classifier'].getint('output_channels')
    k_size = config['classifier'].getint('kernel_size')

    model_path = f"{dir}/models/CNN_Classifier_{SNR_model}dB.pt"
    model      = md.CNN_Classifier(in_ch, out_ch, k_size, config).to(device)

    dataset_path = f"Dataset/Test/{code_name}/receive_{SNR_model}dB.csv"
    dataset      = ds.LoadDataset(dataset_path)

    records_loss_path = f"{dir}/records/train/CNN_Classifier_LOSS_{code_name}_{SNR_model}.pkl"
    records_acc_path  = f"{dir}/records/train/CNN_Classifier_ACC_{code_name}_{SNR_model}.pkl"

    losses, accuracies = classify(device, config, dataset, model, model_path)
    with open(records_loss_path, 'wb') as f:
        pickle.dump(losses, f)
    with open(records_acc_path, 'wb') as f:
        pickle.dump(accuracies, f)

    