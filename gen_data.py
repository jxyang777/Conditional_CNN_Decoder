import dataset as ds

test = True
message_cnt = 100000

if test:
    rg = range(1, 11)
else:
    rg = range(2, 8)

for SNR_data in rg:
    print("SNR: ", SNR_data)
    ds.create_received_pattern(message_cnt, SNR_data, test=test)
