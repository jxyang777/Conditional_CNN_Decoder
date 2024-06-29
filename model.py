import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config
                            # Decoder model index 
model_names = ['CNN',       # 0
              'CCNN_AL',    # 1
              'CCNN_FM',    # 2
              'CNN_2L',     # 3
              'CCNN_AL_2L', # 4
              'CCNN_FM_2L', # 5
              'CCNN_Joint'  # 6
              ]

res_model_names = ['CNN',
                    'CCNN',
                    'CCNN',
                    'CNN',
                    'CCNN',
                    'CCNN_T',
                    'CCNN_ J'
                    ]

    
class CNN_Decoder(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        super().__init__()

        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel_size)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(15*output_channels, 1000)
        self.fc2 = nn.Linear(1000, 12)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
class CNN_2L_Decoder(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, config):
        super().__init__()
        layer1 = config['decoder'].getint('layer_1')
        layer2 = config['decoder'].getint('layer_2')
        dropout = config['decoder'].getfloat('dropout')
        input_size = config['decoder'].getint('input_size')
        feature_size = input_size - kernel_size + 1

        self.conv1   = nn.Conv1d(input_channels, output_channels, kernel_size)
        self.flatten = nn.Flatten()
        self.fc1     = nn.Linear(feature_size*output_channels, layer1)
        self.fc2     = nn.Linear(layer1, layer2)
        self.fc3     = nn.Linear(layer2, 12)
        self.sigmoid = nn.Sigmoid()
        self.dp      = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dp(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.dp(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
    
class CCNN_AL_Decoder(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        super().__init__()

        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(15*output_channels + len(config['codes']), 1000)
        self.fc2 = nn.Linear(1000 + len(config['codes']), 12)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, conditional_input):
        x = F.relu(self.conv(x))
        x = self.flatten(x)
        x = torch.cat((x, conditional_input), dim=1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = torch.cat((x, conditional_input), dim=1)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    

class CCNN_AL_2L_Decoder(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, config):
        super().__init__()
        layer1 = config['decoder'].getint('layer_1')
        layer2 = config['decoder'].getint('layer_2')
        dropout = config['decoder'].getfloat('dropout')
        input_size = config['decoder'].getint('input_size')
        feature_size = input_size - kernel_size + 1

        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(feature_size*output_channels + len(config['codes']), layer1)
        self.fc2 = nn.Linear(layer1 + len(config['codes']), layer2)
        self.fc3 = nn.Linear(layer2 + len(config['codes']), 12)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, conditional_input):
        x = F.relu(self.conv(x))
        x = self.flatten(x)
        x = torch.cat((x, conditional_input), dim=1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = torch.cat((x, conditional_input), dim=1)
        x = self.fc2(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = torch.cat((x, conditional_input), dim=1)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
    
class CCNN_FM_Decoder(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        super().__init__()
        self.output_channels = output_channels

        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel_size)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(output_channels*(15 + len(config['codes'])), 1000)
        self.fc2 = nn.Linear(1000, 12)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, onehot_code):
        onehot_code = onehot_code.unsqueeze(1)
        onehot_code = onehot_code.expand(-1, self.output_channels, -1)

        x = F.relu(self.conv1(x))
        x = torch.cat((x, onehot_code), dim=2)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
class CCNN_FM_2L_Decoder(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, config):
        super().__init__()
        layer1 = config['decoder'].getint('layer_1')
        layer2 = config['decoder'].getint('layer_2')
        dropout = config['decoder'].getfloat('dropout')
        input_size = config['decoder'].getint('input_size')
        feature_size = input_size - kernel_size + 1

        self.output_channels = output_channels
        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel_size)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(output_channels*(feature_size + len(config['codes'])), layer1)
        self.fc2 = nn.Linear(layer1, layer2)
        self.fc3 = nn.Linear(layer2, 12)
        self.bn = nn.BatchNorm1d(output_channels)
        self.sigmoid = nn.Sigmoid()
        self.dp1 = nn.Dropout(dropout)
        self.dp2 = nn.Dropout(dropout)

    def forward(self, x, onehot_code):
        onehot_code = onehot_code.unsqueeze(1)
        onehot_code = onehot_code.expand(-1, self.output_channels, -1)

        x = F.relu(self.conv1(x))
        x = torch.cat((x, onehot_code), dim=2)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dp1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.dp2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
    
class CNN_Classifier(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, config):
        super().__init__()
        layer1 = config['classifier'].getint('layer_1')
        layer2 = config['classifier'].getint('layer_2')
        dropout = config['decoder'].getfloat('dropout')
        input_size = config['classifier'].getint('input_size')
        feature_size = input_size - kernel_size + 1

        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel_size)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(feature_size*output_channels, layer1)
        self.fc2 = nn.Linear(layer1, layer2)
        self.fc3 = nn.Linear(layer2, 4)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        
        return x
    
class CCNN_joint_Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Decoder Part
        self.dec_in_ch        = config['decoder'].getint('input_channels')
        self.dec_out_ch       = config['decoder'].getint('output_channels')
        self.dec_kernel_size  = config['decoder'].getint('kernel_size')
        self.dec_layer_1      = config['decoder'].getint('layer_1')
        self.dec_layer_2      = config['decoder'].getint('layer_2')
        self.dec_dropout      = config['decoder'].getfloat('dropout')
        self.dec_in_size      = config['decoder'].getint('input_size')
        self.dec_out_size     = config['decoder'].getint('output_size')
        self.dec_feature_size = self.dec_in_size - self.dec_kernel_size + 1

        self.dec_conv    = nn.Conv1d(self.dec_in_ch, self.dec_out_ch, self.dec_kernel_size)
        self.dec_flatten = nn.Flatten()
        self.dec_fc1     = nn.Linear(self.dec_out_ch*(self.dec_feature_size + len(config['codes'])), self.dec_layer_1)
        self.dec_fc2     = nn.Linear(self.dec_layer_1, self.dec_layer_2)
        self.dec_fc3     = nn.Linear(self.dec_layer_2, self.dec_out_size)
        self.dec_sigmoid = nn.Sigmoid()
        self.dec_dropout = nn.Dropout(self.dec_dropout)

        # Condition Part
        self.cond_in_ch        = config['classifier'].getint('input_channels')
        self.cond_out_ch       = config['classifier'].getint('output_channels')
        self.cond_kernel_size  = config['classifier'].getint('kernel_size')
        self.cond_layer_1      = config['classifier'].getint('layer_1')
        self.cond_layer_2      = config['classifier'].getint('layer_2')
        self.cond_dropout      = config['classifier'].getfloat('dropout')
        self.cond_in_size      = config['classifier'].getint('input_size')
        self.cond_out_size     = config['classifier'].getint('output_size')
        self.cond_feature_size = self.cond_in_size - self.cond_kernel_size + 1


        self.cond_conv = nn.Conv1d(self.cond_in_ch, self.cond_out_ch, self.cond_kernel_size)
        self.cond_flatten = nn.Flatten()
        self.cond_fc1 = nn.Linear(self.cond_out_ch*self.cond_feature_size, self.cond_layer_1)
        self.cond_fc2 = nn.Linear(self.cond_layer_1, self.cond_layer_2)
        self.cond_fc3 = nn.Linear(self.cond_layer_2, self.cond_out_size)
        self.cond_sigmoid = nn.Sigmoid()
        self.cond_dropout = nn.Dropout(self.cond_dropout)

    def forward(self, x):
        # Condition Part
        x_1 = self.cond_conv(x)
        x_1 = F.relu(x_1)
        x_1 = self.cond_flatten(x_1)
        x_1 = self.cond_fc1(x_1)
        x_1 = self.cond_dropout(x_1)
        x_1 = F.relu(x_1)
        x_1 = self.cond_fc2(x_1)
        x_1 = self.cond_dropout(x_1)
        x_1 = F.relu(x_1)
        conditional_feature = self.cond_fc3(x_1)
        conditional_feature = conditional_feature.unsqueeze(1)
        conditional_feature = conditional_feature.expand(-1, self.dec_out_ch, -1)

        # Decoder Part
        x_2 = self.dec_conv(x)
        x_2= F.relu(x_2)
        x_2 = torch.cat((x_2, conditional_feature), dim=2)
        x_2 = self.dec_flatten(x_2)
        x_2 = self.dec_fc1(x_2)
        x_2 = self.dec_dropout(x_2)
        x_2 = F.relu(x_2)
        x_2 = self.dec_fc2(x_2)
        x_2 = self.dec_dropout(x_2)
        x_2 = F.relu(x_2)
        x_2 = self.dec_fc3(x_2)
        x_2 = self.dec_sigmoid(x_2)
        return x_2

        