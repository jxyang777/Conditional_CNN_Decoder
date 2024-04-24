# Import

import numpy as np
import commpy.channelcoding.convcode as cc
import commpy.modulation as modulation
import torch
from torch.utils.data import Dataset
import csv

import viterbi

# Trellis
trellis_213 = cc.Trellis(memory=np.array(2, ndmin=1), g_matrix=np.array((0o5, 0o7), ndmin=2))
trellis_215 = cc.Trellis(memory=np.array(4, ndmin=1), g_matrix=np.array((0o23, 0o33), ndmin=2))
trellis_217 = cc.Trellis(memory=np.array(6, ndmin=1), g_matrix=np.array((0o133, 0o65), ndmin=2))
trellis_219 = cc.Trellis(memory=np.array(8, ndmin=1), g_matrix=np.array((0o753, 0o561), ndmin=2))

# trellis_214 = cc.Trellis(memory=np.array(3, ndmin=1), g_matrix=np.array((0o15, 0o17), ndmin=2))
# trellis_216 = cc.Trellis(memory=np.array(5, ndmin=1), g_matrix=np.array((0o53, 0o75), ndmin=2))
# trellis_218 = cc.Trellis(memory=np.array(7, ndmin=1), g_matrix=np.array((0o247, 0o371), ndmin=2))
# trellis_2110 = cc.Trellis(memory=np.array(9, ndmin=1), g_matrix=np.array((0o777, 0o1341), ndmin=2))

# trellis__213 = cc.Trellis(memory=np.array(2, ndmin=1), g_matrix=np.array((0o3, 0o5), ndmin=2))
# trellis__215 = cc.Trellis(memory=np.array(4, ndmin=1), g_matrix=np.array((0o7, 0o15), ndmin=2))
# trellis__217 = cc.Trellis(memory=np.array(6, ndmin=1), g_matrix=np.array((0o21, 0o43), ndmin=2))
# trellis__219 = cc.Trellis(memory=np.array(8, ndmin=1), g_matrix=np.array((0o101, 0o211), ndmin=2))

# trellis__214 = cc.Trellis(memory=np.array(3, ndmin=1), g_matrix=np.array((0o7, 0o15), ndmin=2))
# trellis__216 = cc.Trellis(memory=np.array(5, ndmin=1), g_matrix=np.array((0o21, 0o43), ndmin=2))
# trellis__218 = cc.Trellis(memory=np.array(7, ndmin=1), g_matrix=np.array((0o41, 0o101), ndmin=2))
# trellis__2110 = cc.Trellis(memory=np.array(9, ndmin=1), g_matrix=np.array((0o205, 0o421), ndmin=2))

trellises = [trellis_213, trellis_215, trellis_217, trellis_219]
trellises_info = [[2, 0o5, 0o7], [4, 0o23, 0o33], [6, 0o133, 0o65], [8, 0o753, 0o561]]

# trellises = [trellis_213, trellis_215, trellis_217, trellis_219, trellis_214, trellis_216, trellis_218, trellis_2110]
# trellises_info = [[2, 0o5, 0o7], [4, 0o23, 0o33], [6, 0o133, 0o65], [8, 0o753, 0o561], [3, 0o15, 0o17], [5, 0o53, 0o75], [7, 0o247, 0o371], [9, 0o777, 0o1341]]

# trellises = [trellis_213, trellis_215, trellis_217, trellis_219, trellis_214, trellis_216, trellis_218, trellis_2110, trellis__213, trellis__215, trellis__217, trellis__219, trellis__214, trellis__216, trellis__218, trellis__2110]
# trellises_info = [[2, 0o5, 0o7], [4, 0o23, 0o33], [6, 0o133, 0o65], [8, 0o753, 0o561], [3, 0o15, 0o17], [5, 0o53, 0o75], [7, 0o247, 0o371], [9, 0o777, 0o1341], [2, 0o3, 0o5], [4, 0o7, 0o15], [6, 0o21, 0o43], [8, 0o101, 0o211], [3, 0o7, 0o15], [5, 0o21, 0o43], [7, 0o41, 0o101], [9, 0o205, 0o421]]


class LoadDataset(Dataset):
    def __init__(self, csv_file):
        self.data = []
        self.labels = []
        self.mems = []
        self.gen_1s = []
        self.gen_2s = []

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
        data = torch.tensor(self.data[idx])
        label = torch.tensor(self.labels[idx])
        mem = torch.tensor(self.mems[idx])
        gen_1 = torch.tensor(self.gen_1s[idx])
        gen_2 = torch.tensor(self.gen_2s[idx])

        code_info = torch.cat((mem, gen_1, gen_2), 0)
        return data, label, code_info

    

# Error-Pattern Dataset
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

def create_received_pattern(message_cnt, snrdB, test):
    M = 2                            # M-PSK modulation
    rate = 1/2                       # code rate
    k = np.log2(M)                  # number of bit per modulation symbol
    modem = modulation.PSKModem(M)  # M-PSK modem initialization   
    info = trellises_info
    trellis_message_cnt = int(message_cnt / len(trellises))

    path_dataset = f'Dataset/Train/receive_{snrdB}dB.csv'

    if test:
        path_dataset = f'Dataset/Test/receive_{snrdB}dB.csv'

    
    with open(path_dataset, 'w', newline='') as csv_hard:
        writer = csv.writer(csv_hard)
        writer.writerow(['data', 'label', 'mem', 'gen_1', 'gen_2'])

        for trellis in trellises:
            idx = trellises.index(trellis)
            EbNo = snrdB - 10*np.log10(k*rate)
            noiseVar = 10**(-snrdB/10)  # noise variance (power)


            for cnt in range(trellis_message_cnt):
                if (cnt+1+trellis_message_cnt*idx) % 10000==0:
                    print(f"[{cnt+1+trellis_message_cnt*idx}/{message_cnt}]")
                message = np.random.randint(0, 2, 12)                 # message , label
                coded = cc.conv_encode(message, trellises[0], 'cont') # encode

                modulated = modem.modulate(coded)               # modulation
                modulated_uncoded = modem.modulate(message)     # modulation (uncoded)

                Es = np.mean(np.abs(modulated)**2)              # symbol energy
                No = Es/((10**(EbNo/10))*np.log2(M))            # noise spectrum density

                noisy = modulated + np.sqrt(No/2) *\
                    (np.random.randn(modulated.shape[0]) + 1j*np.random.randn(modulated.shape[0]))  # AWGN

                noisy_uncoded = modulated_uncoded + np.sqrt(No/2) *\
                    (np.random.randn(modulated_uncoded.shape[0]) + 1j*np.random.randn(modulated_uncoded.shape[0]))  # AWGN (uncoded case)
                
                demodulated_soft    = modem.demodulate(noisy, demod_type='soft', noise_var=noiseVar)  # demodulation (soft output)
                demodulated_hard    = modem.demodulate(noisy, demod_type='hard')                      # demodulation (hard output)
                demodulated_uncoded = modem.demodulate(noisy_uncoded, demod_type='hard')              # demodulation (uncoded case)

                writer.writerow([''.join(map(str, demodulated_hard)), ''.join(map(str, message)), str(info[idx][0]), str(info[idx][1]), str(info[idx][2])])
                # writer_soft.writerow([''.join(map(str, demodulated_hard)), ''.join(map(str, demodulated_uncoded)), ','.join(map(str, demodulated_soft)), ''.join(map(str, message)), str(info[idx][0]) + ',' + str(int(info[idx][1], 8)) + ',' + str(int(info[idx][2], 8))])

    return path_dataset


