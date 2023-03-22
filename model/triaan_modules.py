import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.conv_modules import InstanceNorm, TimeInstanceNorm, BasicConv

class SelfAttentionPooling(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.W = nn.Linear(input_dim, 1)
        
    def forward(self, x):

        attn = self.W(x).squeeze(-1)
        attn = F.softmax(attn, dim=-1).unsqueeze(-1)
        x    = torch.sum(x * attn, dim=1)

        return x
    
class ChannelAdaptiveNormalization(nn.Module):
    """
    Channel wise Adaptive Normalization (CAN)
    """
    def __init__(self, channels):
        super().__init__()
        
        self.temp    = channels**0.5
        self.inorm   = InstanceNorm()
        self.w_q     = nn.Linear(channels, channels, bias=False)
        self.w_k     = nn.Linear(channels, channels, bias=False)
        self.w_v     = nn.Linear(channels, channels, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src, trg):

        src_q = self.w_q(self.inorm(src).transpose(1,2)) # B, T1, C
        trg_k = self.w_k(self.inorm(trg).transpose(1,2)) # B, T2, C
        trg_v = self.w_v(trg.transpose(1,2))             # B, T2, C
        
        attn = torch.matmul(src_q / self.temp, trg_k.transpose(1,2))  # B, T1, T2
        attn = self.softmax(attn)
        
        mean = torch.matmul(attn, trg_v)                      # B, T1, C
        var  = F.relu(torch.matmul(attn, trg_v**2) - mean**2) # B, T1, C

        mean = mean.transpose(1,2) # B, C, T1
        var  = var.transpose(1,2)  # B, C, T1

        mean = mean.mean(-1).unsqueeze(-1)            # B, C, 1
        std  = torch.sqrt(var.mean(-1)).unsqueeze(-1) # B, C, 1

        return std * self.inorm(src) + mean


class TimeAdaptiveNormalization(nn.Module):
    """
    Time-wise Adaptive Normalization (TAN)
    """
    def __init__(self, channels):
        super().__init__()
        
        self.temp    = channels**0.5
        self.tinorm  = TimeInstanceNorm()
        self.inorm   = InstanceNorm()
        self.w_q     = nn.Linear(channels, channels, bias=False)
        self.w_k     = nn.Linear(channels, channels, bias=False)
        self.w_v     = nn.Linear(channels, channels, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src, trg):

        src_q = self.w_q(self.tinorm(src).transpose(1,2))
        trg_k = self.w_k(self.tinorm(trg).transpose(1,2))
        trg_v = self.w_v(trg.transpose(1,2))
        
        attn = torch.matmul(src_q / self.temp, trg_k.transpose(1,2)) # B, T1, T2
        attn = self.softmax(attn)
        
        mean = torch.matmul(attn, trg_v)                      # B, T1, c
        var  = F.relu(torch.matmul(attn, trg_v**2) - mean**2) # B, T1, C

        mean = mean.transpose(1,2) # B, C, T1
        var  = var.transpose(1,2)  # B, C, T1

        mean = mean.mean(-1).unsqueeze(-1)            # B, C, 1
        std  = torch.sqrt(var.mean(-1)).unsqueeze(-1) # B, C, 1

        return std * self.inorm(src) + mean

class DuAN(nn.Module):
    """
    Dual Adaptive Normalization block (DuAN)
    """
    def __init__(self, channels):
        super().__init__()
        
        self.tan   = TimeAdaptiveNormalization(channels)
        self.can   = ChannelAdaptiveNormalization(channels)
        self.conv1 = nn.Conv1d(2*channels, channels, kernel_size=1, stride=1)

    def forward(self, src, trg):

        src1 = self.tan(src, trg)
        src2 = self.can(src, trg)

        output = torch.cat([src1,src2], dim=1)
        output = self.conv1(output)

        return output

class GLobalAdaptiveNormalization(nn.Module):
    """
    GLoabl Adaptive Normalization (GLAN)
    """
    def __init__(self, channels):
        super().__init__()
        
        self.inorm  = InstanceNorm()
        self.msap   = SelfAttentionPooling(channels)
        self.ssap   = SelfAttentionPooling(channels)

    def forward(self, x, means, stds):

        mean   = self.msap(means).unsqueeze(-1) # B, L, C -> B, C, 1
        std    = self.ssap(stds).unsqueeze(-1)  # B, L, C -> B, C, 1 
        output = self.inorm(x) * std + mean

        return output
    
class TriAANBlock(nn.Module):
    """
    Triple Adaptive Attention Normalization (TriAAN) block 
    """
    def __init__(self, channels):
        super().__init__()
        
        self.tan   = TimeAdaptiveNormalization(channels)
        self.can   = ChannelAdaptiveNormalization(channels)
        self.conv1 = nn.Conv1d(2*channels, channels, kernel_size=1, stride=1)
        self.conv2 = BasicConv(channels, channels, kernel_size=3, stride=1, padding=1, relu=True, bn=True)
        self.glan  = GLobalAdaptiveNormalization(channels)

    def forward(self, src, trg, means, stds):

        src1 = self.tan(src, trg)
        src2 = self.can(src, trg)

        output = torch.cat([src1,src2], dim=1)
        output = self.conv1(output)
        output = self.conv2(output)
        output = self.glan(output, means, stds)

        return output