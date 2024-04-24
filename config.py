import configparser

config = configparser.ConfigParser()
config.read('config.ini')

# # Read code informations
codes = []
for info in config['codes']:
    info = config['codes'][info].split(',')
    for i in range(len(info)):
        base = 10 if i==0 else 8
        info[i] = int(info[i], base)
    codes.append(info)
