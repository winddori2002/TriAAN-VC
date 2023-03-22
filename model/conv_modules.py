import torch
import torch.nn as nn
import torch.nn.functional as F
import math
   
class TimeInstanceNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def cal_stats(self, x):
        
        # B,C,T
        b, c, t = x.shape
        mean = x.mean(1)
        std  = (x.var(1) + self.eps).sqrt()
        mean = mean.view(b, 1, t) # B, 1, T
        std  = std.view(b, 1, t)
        
        return mean, std

    def forward(self, x, return_stats=False):
        
        mean, std = self.cal_stats(x)
        x         = (x - mean) / std
        
        if return_stats:
            return x, mean, std
        else:
            return x

class InstanceNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def cal_stats(self, x):
        
        # input: B, C, T
        mean = x.mean(-1).unsqueeze(-1)                    # B, C, 1
        std  = (x.var(-1) + self.eps).sqrt().unsqueeze(-1) # B, C, 1
        
        return mean, std

    def forward(self, x, return_stats=False):
        
        mean, std = self.cal_stats(x)
        x         = (x - mean) / std

        if return_stats:
            return x, mean, std
        else:
            return x

class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super().__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn   = nn.BatchNorm1d(out_channels, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
            
        return x
    
class EncoderBlock(nn.Module):
    def __init__(self, c_in, c_h):
        super().__init__()
        
        self.conv_block = nn.Sequential(BasicConv(c_in, c_h, kernel_size=3, stride=1, padding=1, relu=True, bn=True),
                                        BasicConv(c_h, c_in, kernel_size=3, stride=1, padding=1, relu=False, bn=False))
        
    def forward(self, x):

        x = x + self.conv_block(x)

        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, c_in, c_h, c_out):
        super().__init__()
        
        self.conv_block1 = nn.Sequential(BasicConv(c_in, c_h, kernel_size=3, stride=1, padding=1, relu=True, bn=True),
                                         BasicConv(c_h, c_in, kernel_size=3, stride=1, padding=1, relu=False, bn=False))
        self.conv_block2 = nn.Sequential(BasicConv(c_in, c_h, kernel_size=3, stride=1, padding=1, relu=True, bn=True),
                                         BasicConv(c_h, c_in, kernel_size=3, stride=1, padding=1, relu=False, bn=False))
        
    def forward(self, x):
        
        x = x + self.conv_block1(x)
        x = x + self.conv_block2(x)
        
        return x