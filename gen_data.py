import dataset as ds
import pickle
import time
import dataset as ds
import multiprocessing as mp
import argparse



def generate_data(test, message_cnt, code_type): 
    if test: rg = range(1, 11, 1)
    else:    rg = range(2, 8)


    BERs_uncoded = []
    BERs_hard = []
    for SNR_data in rg:
        print("SNR: ", SNR_data)
        BER_uncoded, BER_hard = ds.create_received_pattern(message_cnt, SNR_data, test=test, code_type=code_type)
        BERs_uncoded.append(BER_uncoded)
        BERs_hard.append(BER_hard)
    # print(BERs_uncoded)


    if code_type == 5: code_type = "mix"

    if test: 
        uncoded_BER_path = f"Dataset/Test/code_{code_type}/uncoded_BER.pkl"   
        hard_BER_path = f"Dataset/Test/code_{code_type}/hard_BER.pkl"
    else:    
        uncoded_BER_path = f"Dataset/Train/code_{code_type}/uncoded_BER.pkl"
        hard_BER_path = f"Dataset/Train/code_{code_type}/hard_BER.pkl"


    with open(uncoded_BER_path, 'wb') as f:
        # print(f"Saving Uncoded BER: {uncoded_BER_path}")
        pickle.dump(BERs_uncoded, f)
    

def gen_data(test, message_cnt):
    if test: code_types = [1, 2, 3, 4, 5]
    else: code_types = [5]

    with mp.Pool() as pool:
        pool.starmap(generate_data, [(test, message_cnt, code_type) for code_type in code_types])

# code_type: 1~4 correspond to trellis 1, 2, 3, 4
# code_type: 5   corresponds to all trellises mixed

parser = argparse.ArgumentParser(description='Generate dataset')
parser.add_argument("-t", "--test", action="store_true", help='Generate test dataset')
parser.add_argument("-m", "--message_cnt", type=int, default=100000, help='Number of messages to generate')
args = parser.parse_args()

test = args.test
message_cnt = args.message_cnt

print(test, message_cnt)
print(f"Generating {'test' if test else 'train'} dataset with {message_cnt} messages for each SNR value")

start_time = time.time()
# Generate training data
# generate_data(test=False, message_cnt=100000, code_type=5)
# generate_data(test=False, message_cnt=100000, code_type=1)
# generate_data(test=False, message_cnt=100000, code_type=2)
# generate_data(test=False, message_cnt=100000, code_type=3)
# generate_data(test=False, message_cnt=100000, code_type=4)

# # Generate testing data
# generate_data(test=True, message_cnt=100000, code_type=1)
# generate_data(test=True, message_cnt=100000, code_type=2)
# generate_data(test=True, message_cnt=100000, code_type=3)
# generate_data(test=True, message_cnt=100000, code_type=4)
# generate_data(test=True, message_cnt=100000, code_type=5)

# generate data using multiprocessing
gen_data(test, message_cnt)
end_time = time.time()
print(f"Time: {round((end_time - start_time)/60)} min {round((end_time - start_time)%60, 2)} sec")
