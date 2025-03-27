import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class PressNet_Block(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.5):
        super(PressNet_Block, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout2d(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.leaky_relu(out1)
        out1 = self.dropout(out1)

        out2 = self.conv2(x)
        out2 = self.bn2(out2)
        out2 = self.leaky_relu(out2)
        out2 = self.dropout(out2)

        out3 = self.conv3(x)
        out3 = self.bn3(out3)
        out3 = self.leaky_relu(out3)
        out3 = self.dropout(out3)

        out = out1 + out2 + out3
        return out

class PressNet(nn.Module):
    def __init__(self, input_size, output_size, foot_mask, dropout=0.0, mult_by_mask=False):
        super(PressNet, self).__init__()

        self.fc = nn.Linear(input_size, 6144)
        self.blocks = nn.ModuleList([
            PressNet_Block(512, 256, dropout),
            PressNet_Block(256, 128, dropout),
            PressNet_Block(128, 64, dropout),
        ])

        self.conv_branch = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.fc_branch = nn.Linear(64 * 4 * 3, 1)
        self.foot_mask = foot_mask
        self.output_size = output_size
        self.mult_by_mask = mult_by_mask

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 512, 4, 3)
        
        for block in self.blocks:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = block(x)

        x1 = self.conv_branch(x)
        x2 = self.fc_branch(x.view(x.size(0), -1))

        out = x1 + x2
        out = F.relu(out)
        out = out.view(-1, self.output_size)
        out = F.softmax(out, dim=1)
        
        if self.mult_by_mask:
            out = out * self.foot_mask
        return out

class PNS(nn.Module):
    def __init__(self, input_size, hidden_count, FC_size, output_size, foot_mask, dropout=0.0, mult_by_mask=False):
        super(PNS, self).__init__()
        self.initial = nn.Sequential(
            nn.Linear(input_size, FC_size),
        )
        self.layers = nn.ModuleList()
        for _ in range(hidden_count):
            self.layers.append(nn.Sequential(
                nn.Linear(FC_size, FC_size),
                nn.BatchNorm1d(FC_size),
                nn.ReLU(),
                nn.Dropout(dropout),

                nn.Linear(FC_size, FC_size),
                nn.BatchNorm1d(FC_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
        self.out = nn.Sequential(
            nn.Linear(FC_size,output_size),  
        )
        self.mult_by_mask = mult_by_mask
        self.foot_mask = foot_mask.flatten()
        self.output_size = output_size
        
    def __normal_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
                
    def init_weights(self):
        self.apply(self.__normal_init)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        residual = self.initial(x)
        for layer in self.layers:
            x = layer(residual)
            residual = x + residual  
        x = self.out(x)
        
        if self.mult_by_mask:
            x = x * self.foot_mask
       
        # Only returning pressure estimation from PNS 
        ret = {}
        ret['pressure'] = x    
        return ret