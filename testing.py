import torch
from torch.utils.data import DataLoader
import pickle
import os

import dataset as ds
import utils
import model as md
from config import config
import config as cf

def test(device, model, data_path, cond, rand):
    batch_size = config['test'].getint('batch_size')

    codes = torch.tensor(cf.codes).to(device)
    codes = codes.unsqueeze(0).expand(batch_size, -1, -1)  # Repeat codes for each code_info

    model.eval()
    BER = 0      # Bit Error Rate


    testset = ds.LoadDataset(data_path)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, drop_last=True)

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            seqs      = data[0].float().unsqueeze(1).to(device)
            targets    = data[1].float().to(device)

            if cond: # Conditional training
                code_info = data[2].to(device)
                if rand:
                    random_indices = torch.randint(0, len(ds.trellises_info), (len(code_info),)).to(device)
                    info = torch.zeros(len(code_info), len(ds.trellises_info)).to(device)
                    info.scatter_(1, random_indices.unsqueeze(1), 1)
                    code_info = info.float()
                else:
                    code_info = torch.all(code_info.unsqueeze(1).eq(codes), dim=2).float()

                predictions = model(seqs, code_info).round().to(device)  # Use Cond_CNN_Decoder
            else:  # Unconditional training
                predictions = model(seqs).round().to(device)  # Use CNN_Decoder

            HD = torch.sum(predictions != targets, dim=1)
            BER += HD.sum()

            if ((batch_idx+1) % 10) == 0:
                batch = (batch_idx+1) * len(seqs)
                percentage = (100. * (batch_idx+1) / len(test_loader))
                print(f'[{batch:5d} / {len(test_loader.dataset)}] ({percentage:.0f} %)')
                
    BER = BER / (len(test_loader) * batch_size * 12)  # 1 frame has 12 bits
    BER = round(float(BER), 10)


    return BER

def test_model(device, dir, model_idx, SNR_model, rand=False):
    in_ch = config['CNN'].getint('input_channels')
    out_ch = config['CNN'].getint('output_channels')
    k_size = config['CNN'].getint('kernel_size')

    models = [md.CNN_Decoder(in_ch, out_ch, k_size), 
            md.cALCNN_Decoder(in_ch, out_ch, k_size), 
            md.cCNN_cat_ft_Decoder(in_ch, out_ch, k_size)]
    
    model_names = ['CNN', 
                  'cALCNN', 
                  'cCNN_cat_ft']

    cond = (model_idx != 0)

    if rand:
        records_path = f"{dir}/records/test/{model_names[model_idx]}_{SNR_model}dB_random.pkl"
        model_path = f"{dir}/models/{model_names[model_idx]}_{SNR_model}dB_random.pt"
    else:
        records_path = f"{dir}/records/test/{model_names[model_idx]}_{SNR_model}dB.pkl"
        model_path = f"{dir}/models/{model_names[model_idx]}_{SNR_model}dB.pt"

    model = utils.load_model(models[model_idx], model_path).to(device)
    model.eval()

    BERs = []
    for SNR_data in range(1, 11):
        print(f"SNR: {SNR_data} dB")
        data_path = f"Dataset/Test/receive_{SNR_data}dB.csv"
        BER = test(device, model, data_path, cond=cond, rand=rand)
        BERs.append(BER)
        print(f"BER: {BER}")

    with open(records_path, 'wb') as f:
        pickle.dump(BERs, f)

    print(records_path)

    return BERs