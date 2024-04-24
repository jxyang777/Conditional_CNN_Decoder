import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle

import utils
import config as cf
import dataset as ds
import model as md

def train(device, config, dataset, model, model_path, cond, rand):
    codes = torch.tensor(cf.codes).to(device)
    
    epochs     = config['train'].getint('epochs')
    batch_size = config['train'].getint('batch_size')
    lr         = config['train'].getfloat('learning_rate')

    model.train()
    loss_fn   = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    BER_best = 1
    epoch = 0

    while True:
        epoch += 1
        ACC, BER = 0, 0
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        for batch_idx, data in enumerate(train_loader):

            seqs      = data[0].float().unsqueeze(1).to(device)
            targets    = data[1].float().to(device)

            optimizer.zero_grad()

            if cond: # Conditional training
                code_info = data[2].to(device)
                if rand:  # random create code_info
                    random_indices = torch.randint(0, len(ds.trellises_info), (len(code_info),)).to(device)
                    info = torch.zeros(len(code_info), len(ds.trellises_info)).to(device)
                    info.scatter_(1, random_indices.unsqueeze(1), 1)
                    code_info = info.float()
                else:
                    code_info = torch.all(code_info.unsqueeze(1).eq(codes), dim=2).float()
                predictions = model(seqs, code_info).to(device)
            else:    # Unconditional training
                predictions = model(seqs).to(device)
                

            # Calculate Hamming Distance
            HD = torch.sum(predictions.round() != targets, dim=1)
            BER += HD.sum()

            loss = loss_fn(predictions, targets)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            # show progress and loss
            if ((batch_idx+1) % 10) == 0:
                batch = (batch_idx+1) * len(seqs)
                percentage = (100. * (batch_idx+1) / len(train_loader))
                print(f'Epoch {epoch}: [{batch:5d} / {len(train_loader.dataset)}] ({percentage:.0f} %)' + f'  Loss: {loss.item():.6f}')

        # Calculate BER per epoch
        BER = BER / (len(train_loader.dataset) * 12)
        print(f"Epoch {epoch} Result: ACC: {ACC}, BER: {BER}")

        if BER < BER_best:
            BER_best = BER
            utils.save_model(model, model_path)
        
        if epoch==epochs:
            return losses


def train_model(device, config, dir, SNR, model_idx, rand=False):
    in_ch = config['CNN'].getint('input_channels')
    out_ch = config['CNN'].getint('output_channels')
    k_size = config['CNN'].getint('kernel_size')

    models = [md.CNN_Decoder(in_ch, out_ch, k_size), 
            md.cALCNN_Decoder(in_ch, out_ch, k_size), 
            md.cCNN_cat_ft_Decoder(in_ch, out_ch, k_size)]
    
    model_names = ['CNN', 
                  'cALCNN', 
                  'cCNN_cat_ft']
    
    model = models[model_idx].to(device)

    cond = (model_idx != 0)

    model_path   = f"{dir}/models/{model_names[model_idx]}_{SNR}dB.pt"
    records_path = f"{dir}/records/train/loss_{model_names[model_idx]}_{SNR}dB.pkl"


    dataset_path = f"Dataset/Train/receive_{SNR}dB.csv"
    dataset   = ds.LoadDataset(dataset_path)

    losses = train(device, config, dataset, model, model_path, cond=cond, rand=rand)

    with open(records_path, 'wb') as f:
        print(f"Saving records: {records_path}")
        pickle.dump(losses, f)

    