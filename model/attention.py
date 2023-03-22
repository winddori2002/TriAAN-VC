import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.conv_modules import InstanceNorm, TimeInstanceNorm

class ContentAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        self.d_k     = channels ** 0.5
        self.inorm   = InstanceNorm()
        self.w_q     = nn.Linear(channels, channels, bias=False)
        self.w_k     = nn.Linear(channels, channels, bias=False)
        self.w_v     = nn.Linear(channels, channels, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.linear  = nn.Linear(channels, channels, bias=False)
    
    def forward(self, x):
        
        # inputs: B, C, T
        q = self.w_q(self.inorm(x).transpose(1,2))
        k = self.w_k(x.transpose(1,2))
        v = self.w_v(x.transpose(1,2))

        attn = torch.matmul(q / self.d_k, k.transpose(1,2))  # B, T, T
        attn = self.softmax(attn)  
        
        output = torch.matmul(attn, v)
        output = self.linear(output).transpose(1,2)
        output = x + output
        
        return output

class SpeakerAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        self.d_k     = channels ** 0.5
        self.tinorm  = TimeInstanceNorm()
        self.w_q     = nn.Linear(channels, channels, bias=False)
        self.w_k     = nn.Linear(channels, channels, bias=False)
        self.w_v     = nn.Linear(channels, channels, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.linear  = nn.Linear(channels, channels, bias=False)
    
    def forward(self, x):
        
        # inputs: B, C, T
        q = self.w_q(self.tinorm(x).transpose(1,2))  # TIN(x)
        k = self.w_k(x.transpose(1,2)) 
        v = self.w_v(x.transpose(1,2))

        attn = torch.matmul(q / self.d_k, k.transpose(1,2))  # B, T, T
        attn = self.softmax(attn)              
        
        output = torch.matmul(attn, v)
        output = self.linear(output).transpose(1,2)
        output = x + output
        
        return output