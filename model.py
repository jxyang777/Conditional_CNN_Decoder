import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config
    
class CNN_Decoder(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        super().__init__()

        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel_size)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(15*output_channels, 1000)
        self.fc2 = nn.Linear(1000, 12)
        self.bn = nn.BatchNorm1d(output_channels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class cALCNN_Decoder(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        super().__init__()

        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(15*output_channels + len(config['codes']), 1000)
        self.fc2 = nn.Linear(1000 + len(config['codes']), 12)
        self.bn = nn.BatchNorm1d(output_channels)

    def forward(self, x, conditional_input):
        x = F.relu(self.conv(x))
        x = self.bn(x)
        x = self.flatten(x)
        x = torch.cat((x, conditional_input), dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = torch.cat((x, conditional_input), dim=1)
        x = self.fc2(x)

        return x
    
class cCNN_cat_ft_Decoder(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        super().__init__()
        self.output_channels = output_channels

        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel_size)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(output_channels*(15 + len(config['codes'])), 1000)
        self.fc2 = nn.Linear(1000, 12)
        self.bn = nn.BatchNorm1d(output_channels)

    def forward(self, x, onehot_code):
        onehot_code = onehot_code.unsqueeze(1)
        onehot_code = onehot_code.expand(-1, self.output_channels, -1)

        x = F.relu(self.conv1(x))
        x = self.bn(x)
        x = torch.cat((x, onehot_code), dim=2)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x