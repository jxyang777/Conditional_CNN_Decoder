import numpy as np
import commpy.channels as channels
import commpy.modulation as modulation
import commpy.channelcoding.convcode as cc
from multiprocessing import Pool
import time
import pickle

trellis_213 = cc.Trellis(memory=np.array(2, ndmin=1), g_matrix=np.array((0o5, 0o7), ndmin=2))
trellis_215 = cc.Trellis(memory=np.array(4, ndmin=1), g_matrix=np.array((0o23, 0o33), ndmin=2))
trellis_217 = cc.Trellis(memory=np.array(6, ndmin=1), g_matrix=np.array((0o133, 0o65), ndmin=2))
trellis_219 = cc.Trellis(memory=np.array(8, ndmin=1), g_matrix=np.array((0o753, 0o561), ndmin=2))

trellises = [trellis_213, trellis_215, trellis_217, trellis_219]

def awgn(signal, snr_dB, rate=1):
    snr_linear = 10**(snr_dB / 10.0)
    signal_power = np.mean(np.abs(signal)**2)
    noise_power = signal_power * rate /snr_linear
    noise = np.sqrt(noise_power) * np.random.randn(len(signal))
    return signal + noise

def BER_calc(a, b):
    num_ber = np.sum(a != b)
    ber = num_ber / len(a)
    return int(num_ber), ber

def simulate(args):
    snrdB, N, N_c, trellis, modem, rate = args
    BER_hard = np.zeros(N_c)

    for cntr in range(N_c):
        message_bits = np.random.randint(0, 2, N)

        # Convolutional Code Encoding
        coded_bits = cc.conv_encode(message_bits, trellis, 'cond')
        
        # Modulation
        modulated = modem.modulate(coded_bits)

        # Add channel noise
        noised_coded   = awgn(modulated, snrdB)

        # Demodulation
        demodulated_hard = modem.demodulate(noised_coded, demod_type='hard')

        # Viterbi decoding
        decoded_hard = cc.viterbi_decode(demodulated_hard, trellis, decoding_type='hard')

        # Calculate bit-error ratio
        NumErr, BER_hard[cntr] = BER_calc(message_bits, decoded_hard[:message_bits.size])

    # averaged bit-error ratio
    mean_BER_hard = BER_hard.mean()
    return mean_BER_hard


N = 12                         # number of bits per block
M = 2                          # modulation order (BPSK)
k = np.log2(M)                 # number of bit per modulation symbol
rate = 1/2                     # code rate
modem = modulation.PSKModem(M) # M-PSK modem initialization 

test_range = range(1, 11)
N_c = 1000 # number of trials

# 使用 multiprocessing.Pool 並行處理
pool = Pool()
results = []
start_time = time.time()

for trellis in trellises:
    for snrdB in test_range:
        args = (snrdB, N, N_c, trellis, modem, rate)
        results.append(pool.apply_async(simulate, args=(args,)))

pool.close()
pool.join()

BERs_hard = {i: [] for i in range(4)}
BERs_uncoded = {i: [] for i in range(4)}

for i, result in enumerate(results):
    mean_BER_hard = result.get()
    trellis_index = i // len(test_range)
    BERs_hard[trellis_index].append(mean_BER_hard)

end_time = time.time()

for i in range(4):
    print(f"Trellis {i+1} Hard decision:\n{BERs_hard[i]}\n")
    rec_path = f"Dataset/Test/code_{i+1}/hard_BER.pkl"
    with open(rec_path, 'wb') as f:
        pickle.dump(BERs_hard[i], f)

BERs_hard_list = np.array(list(BERs_hard.values()))
BER_hard_mix = np.mean(BERs_hard_list, axis=0)
print(f"Hard Mix BER:\n{BER_hard_mix}\n")

# Save BER
rec_path = f"Dataset/Test/code_mix/hard_BER.pkl"
with open(rec_path, 'wb') as f:
    pickle.dump(BER_hard_mix, f)

print(f"Simulation completed in {end_time - start_time:.2f} seconds.")
