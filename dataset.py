# Import

import numpy as np
import commpy.channelcoding.convcode as cc
import commpy.modulation as modulation
import commpy.channels as channels
import torch
from torch.utils.data import Dataset
import csv
import os
from pathlib import Path

# Trellis
trellises = [cc.Trellis(memory=np.array(2, ndmin=1), g_matrix=np.array((0o5, 0o7), ndmin=2)),
             cc.Trellis(memory=np.array(4, ndmin=1), g_matrix=np.array((0o23, 0o33), ndmin=2)),
             cc.Trellis(memory=np.array(6, ndmin=1), g_matrix=np.array((0o133, 0o65), ndmin=2)),
             cc.Trellis(memory=np.array(8, ndmin=1), g_matrix=np.array((0o753, 0o561), ndmin=2))]
trellises_info = [[2, 0o5, 0o7], [4, 0o23, 0o33], [6, 0o133, 0o65], [8, 0o753, 0o561]]

class LoadDataset(Dataset):
    def __init__(self, csv_file):
        self.data = []
        self.labels = []
        self.mems = []
        self.gen_1s = []
        self.gen_2s = []
        self.uncoded = []

        with open(csv_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.data.append([int(bit) for bit in row['data']])
                self.labels.append([int(bit) for bit in row['label']])
                self.mems.append([int(''.join(map(str, row['mem'])))])
                self.gen_1s.append([int(''.join(map(str, row['gen_1'])))])
                self.gen_2s.append([int(''.join(map(str, row['gen_2'])))])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx], dtype=torch.float)
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        mem = torch.tensor(self.mems[idx])
        gen_1 = torch.tensor(self.gen_1s[idx])
        gen_2 = torch.tensor(self.gen_2s[idx])

        code_info = torch.cat((mem, gen_1, gen_2), 0)
        return data, label, code_info
    

# Add 1 to sequence
def add(seq):
    done = False
    seq = seq.copy()
    pos = len(seq) - 1

    if (seq==np.ones(12, np.int8)).all():
        done = True

    while(done==False):
        if seq[pos]==0:
            seq[pos] = 1
            break
        else:
            seq[pos] = 0
            pos -= 1

    if (seq==np.ones(12, np.int8)).all():
        done = True

    return seq, done

def hamming_distance(arr1, arr2):
    return np.count_nonzero(arr1 != arr2)

def check_directory():
    if not os.path.exists('Dataset'):              # create dataset directory
        os.makedirs('Dataset')
    if not os.path.exists('Dataset/Train'):
        os.makedirs('Dataset/Train')
        if not os.path.exists('Dataset/Train/code_1'): os.makedirs('Dataset/Train/code_1')
        if not os.path.exists('Dataset/Train/code_2'): os.makedirs('Dataset/Train/code_2')
        if not os.path.exists('Dataset/Train/code_3'): os.makedirs('Dataset/Train/code_3')
        if not os.path.exists('Dataset/Train/code_4'): os.makedirs('Dataset/Train/code_4')
        if not os.path.exists('Dataset/Train/code_mix'): os.makedirs('Dataset/Train/code_mix')
    if not os.path.exists('Dataset/Test'):
        os.makedirs('Dataset/Test')
        if not os.path.exists('Dataset/Test/code_1'): os.makedirs('Dataset/Test/code_1')
        if not os.path.exists('Dataset/Test/code_2'): os.makedirs('Dataset/Test/code_2')
        if not os.path.exists('Dataset/Test/code_3'): os.makedirs('Dataset/Test/code_3')
        if not os.path.exists('Dataset/Test/code_4'): os.makedirs('Dataset/Test/code_4')
        if not os.path.exists('Dataset/Test/code_mix'): os.makedirs('Dataset/Test/code_mix')

def awgn(signal, snr_dB, rate):
    snr_linear = 10**(snr_dB / 10.0)
    signal_power = np.mean(np.abs(signal)**2)
    noise_power = signal_power * rate / snr_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    
    return signal + noise

# Error Pattern
def create_error_pattern(num_error = 1, dataset_path=None):
    info = trellises_info
    
    with open(dataset_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['data', 'label', 'mem', 'gen_1', 'gen_2'])

        for trellis in trellises:
            idx = trellises.index(trellis)
            message = np.zeros(12, np.int8)
            
            for i in range(np.power(2, 12)):                # 12 bits message
                if i!=0:
                    message, done = add(message)
                encoded = cc.conv_encode(message, trellis, 'cont')
                
                # print(message, encoded, f"Message {i+1}, using trellis_{trellises.index(trellis)}")
                writer.writerow([''.join(map(str, encoded)), ''.join(map(str, message)), str(info[idx][0]), str(info[idx][1]), str(info[idx][2])])              # correct encoded

                for j in range(len(encoded)):
                    err_1_encoded = np.copy(encoded)
                    if err_1_encoded[j]==0:
                        err_1_encoded[j] = 1
                    else:
                        err_1_encoded[j] = 0
                    writer.writerow([''.join(map(str, err_1_encoded)), ''.join(map(str, message)), str(info[idx][0]), str(info[idx][1]), str(info[idx][2])])                    # 1-error

                    if num_error>=2:
                        for k in range(j+1, len(encoded)):
                            err_2_encoded = np.copy(err_1_encoded)
                            if err_2_encoded[k]==0:
                                err_2_encoded[k] = 1
                            else:
                                err_2_encoded[k] = 0
                            writer.writerow([''.join(map(str, err_2_encoded)), ''.join(map(str, message)), str(info[idx][0]), str(info[idx][1]), str(info[idx][2])])            # 2-error

                            if num_error>=3:
                                for l in range(k+1, len(encoded)):
                                    err_3_encoded = np.copy(err_2_encoded)
                                    if err_3_encoded[l]==0:
                                        err_3_encoded[l] = 1
                                    else:
                                        err_3_encoded[l] = 0

                                    # print(message, err_3_encoded, f"Error at pos {j}, {k}, {l}")
                                    writer.writerow([''.join(map(str, err_3_encoded)), ''.join(map(str, message)), str(info[idx][0]), str(info[idx][1]), str(info[idx][2])])    # 3-error

# Received Pattern
def create_received_pattern(message_cnt, snrdB, test, code_type):
    check_directory()
    message_bits = 12
    M = 2                            # M-PSK modulation
    rate = 1/2                       # code rate
    k = np.log2(M)                   # number of bit per modulation symbol
    modem = modulation.PSKModem(M)   # M-PSK modem initialization   
    tb_depth = 5

    BER_uncoded = 0
    BER_hard    = 0    # decoded BER by vitrebi decoder

    trells = trellises
    info = trellises_info
    type = code_type - 1

    # set dataset path for mixed code type
    path_dataset = f'Dataset/Train/code_mix/receive_{snrdB}dB.csv'
    if test:
        path_dataset = f'Dataset/Test/code_mix/receive_{snrdB}dB.csv'

    # set trellises and info for single code type
    if type != 4:   # code type equal to 1,2,3,4
        trells = trells[type:type+1]
        info = info[type:type+1]      

        path_dataset = f'Dataset/Train/code_{type+1}/receive_{snrdB}dB.csv'
        if test:
            path_dataset = f'Dataset/Test/code_{type+1}/receive_{snrdB}dB.csv'

    trellis_message_cnt = int(message_cnt / len(trells))

    with open(path_dataset, 'w', newline='') as csv_hard:
        writer = csv.writer(csv_hard)
        writer.writerow(['data', 'label', 'mem', 'gen_1', 'gen_2', 'uncoded', 'bit_error'])

        for trell in trells:
            idx = trells.index(trell)

            total_errors = 0
            total_bits = 0

            for cnt in range(trellis_message_cnt):
                message = np.random.randint(0, 2, message_bits) # message , label
                coded = cc.conv_encode(message, trell, 'cont')  # encode

                modulated = modem.modulate(coded)               # modulation
                modulated_uncoded = modem.modulate(message)   

                noised_coded = awgn(modulated, snrdB, 1/2)
                noised_uncoded = awgn(modulated_uncoded, snrdB, 1)

                # input of the model is the output of the channel, no demodulate.
                demodulated_hard    = modem.demodulate(noised_coded, demod_type='hard')      # demodulation (hard output)
                demodulated_uncoded = modem.demodulate(noised_uncoded, demod_type='hard')    # demodulation (uncoded case)
                demodulated_hard_nonoise = modem.demodulate(modulated, demod_type='hard')   # demodulation (hard output, no noise)

                BER_uncoded += hamming_distance(message, demodulated_uncoded)
                HD_noise = hamming_distance(demodulated_hard, demodulated_hard_nonoise)

                # print(f'Test {cnt+1}: Number of errors = {HD_noise}')
                num_errors = HD_noise
                total_errors += num_errors
                total_bits += len(coded)
                
                writer.writerow([''.join(map(str, demodulated_hard)), 
                                 ''.join(map(str, message)), 
                                 str(info[idx][0]), 
                                 str(info[idx][1]), 
                                 str(info[idx][2]), 
                                 ''.join(map(str, demodulated_uncoded)), 
                                 str(HD_noise)])
            
            # print(f"Avg BER: {total_errors}/{total_bits}")
    BER_uncoded = BER_uncoded / (message_cnt * 12)

    return BER_uncoded, BER_hard