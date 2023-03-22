import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.conv_modules import BasicConv, TimeInstanceNorm, InstanceNorm, EncoderBlock, DecoderBlock
from model.attention import SpeakerAttention
from model.triaan_modules import DuAN, TriAANBlock

class LF0Encoder(nn.Module):
    def __init__(self, c_h, c_out, c_in=1):
        super().__init__()

        self.in_conv   = BasicConv(c_in, c_h, kernel_size=3, stride=1, padding=1, relu=True, bn=True)
        self.rnn_layer = nn.GRU(c_h, c_h//2, 2, batch_first=True, bidirectional=True)
        self.out_conv  = BasicConv(c_h, c_out, kernel_size=3, stride=1, padding=1, relu=True, bn=True)

    def forward(self, lf0):

        lf0    = self.in_conv(lf0)                  # B, C, T
        lf0, _ = self.rnn_layer(lf0.transpose(1,2)) # B, T, C
        lf0    = self.out_conv(lf0.transpose(1,2))  # B, C, T

        return lf0

class SpeakerEncoder(nn.Module):
    def __init__(self, c_in, c_out, num_layer, c_h):
        super().__init__()

        self.inorm       = InstanceNorm()
        self.in_conv     = BasicConv(c_in, c_h, kernel_size=3, stride=1, padding=1, relu=True, bn=True)
        self.conv_blocks = nn.ModuleList([
                                          nn.Sequential(EncoderBlock(c_h, c_h), SpeakerAttention(c_h))
                                          for _ in range(num_layer)
                                          ])
        self.out_conv    = BasicConv(c_h, c_out, kernel_size=3, stride=1, padding=1, relu=True, bn=True)

    def forward(self, x):

        x = self.in_conv(x.squeeze(1))

        skips = []
        for block in self.conv_blocks:
            x = block(x)
            skips.append(x)
            x = self.inorm(x)

        x = self.out_conv(x)

        return x, skips

class ContentEncoder(nn.Module):
    def __init__(self, c_in, c_out, num_layer, c_h):
        super().__init__()

        self.inorm       = InstanceNorm()
        self.in_conv     = BasicConv(c_in, c_h, kernel_size=3, stride=1, padding=1, relu=True, bn=True)
        self.conv_blocks = nn.ModuleList([
                                          EncoderBlock(c_h, c_h)
                                          for _ in range(num_layer)
                                          ])
        self.out_conv    = BasicConv(c_h, c_out, kernel_size=3, stride=1, padding=1, relu=True, bn=True)

    def forward(self, x):

        x = self.in_conv(x.squeeze(1))
        
        for block in self.conv_blocks:
            x = block(x)
            x = self.inorm(x)

        x = self.out_conv(x)

        return x
    
class Decoder(nn.Module):
    def __init__(self, c_in, c_h, c_out, num_layer):
        super().__init__()
        
        self.inorm   = InstanceNorm()
        self.duan    = DuAN(c_in)
        self.in_conv = BasicConv(c_in, c_h, kernel_size=3, stride=1, padding=1, relu=True, bn=True)
        
        self.triaan_blocks = nn.ModuleList([
                                          TriAANBlock(c_h) 
                                          for _ in range(num_layer)
                                          ])
        self.conv_blocks   = nn.ModuleList([
                                          DecoderBlock(c_h, c_h, c_h) 
                                          for _ in range(num_layer)
                                          ])

        self.rnn_layer   = nn.GRU(c_h, c_h, 2)
        self.out_linear  = nn.Linear(c_h, c_out)

    def _stack_stats(self, trg_skips):
        
        trg_means = []
        trg_stds  = []
        for trg in trg_skips:
            m, s = self.inorm.cal_stats(trg)
            trg_means.append(m) # [B, C, 1]
            trg_stds.append(s)
        trg_means = torch.cat(trg_means, dim=-1).transpose(1,2) # [(B,C,1);(B,C,1)] -> (B,L,C)
        trg_stds  = torch.cat(trg_stds, dim=-1).transpose(1,2)

        return trg_means, trg_stds

    def forward(self, src, trg, trg_skips):

        output = self.duan(src, trg)                       # initial conversion with Dual Adaptive Normalization
        output = self.in_conv(output)
        trg_means, trg_stds = self._stack_stats(trg_skips) # stack stats for GLobal Adaptive Normalization

        trg_skips.reverse()
        for i, (block, triaan, trg) in enumerate(zip(self.conv_blocks, self.triaan_blocks, trg_skips)):
            output = block(output)
            output = triaan(output, trg, trg_means, trg_stds)

        output    = output.transpose(1,2)
        output, _ = self.rnn_layer(output)
        output    = self.out_linear(output)
        output    = output.transpose(1,2)
        
        return output

class TriAANVC(nn.Module):
    """
    TriAAN-VC for any-to-any voice conversion.

    encoder_params:
        c_in:    : input size of encoder  (Mel: 80, CPC: 256)
        c_h:     : hidden size of encoder
        c_out:   : output size of encoder
        num_layer: number of layer
        
    decoder_params:
        c_in:    : input size of decoder  (same as c_out of encoder)
        c_h:     : hidden size of decoder
        c_out:   : output size of decoder (mel size: 80)
        num_layer: number of layer
    """
    def __init__(self, encoder_params, decoder_params):
        super().__init__()

        self.cnt_encoder = ContentEncoder(**encoder_params)
        self.spk_encoder = SpeakerEncoder(**encoder_params)

        self.rnn_layer   = nn.GRU(encoder_params.c_out+1, encoder_params.c_out, 2, batch_first=True, bidirectional=True)
        self.linear      = nn.Linear(encoder_params.c_out*2, encoder_params.c_out)

        self.decoder     = Decoder(**decoder_params)
        self.post_net    = nn.Sequential(nn.Conv1d(80, 512,  kernel_size=5, padding=2), nn.BatchNorm1d(512), nn.Tanh(), nn.Dropout(0),
                                         nn.Conv1d(512, 512, kernel_size=5, padding=2), nn.BatchNorm1d(512), nn.Tanh(), nn.Dropout(0),
                                         nn.Conv1d(512, 512, kernel_size=5, padding=2), nn.BatchNorm1d(512), nn.Tanh(), nn.Dropout(0),
                                         nn.Conv1d(512, 512, kernel_size=5, padding=2), nn.BatchNorm1d(512), nn.Tanh(), nn.Dropout(0),
                                         nn.Conv1d(512, 80,  kernel_size=5, padding=2), nn.BatchNorm1d(80), nn.Dropout(0))

    def forward(self, src, src_lf0, trg):
        """
        B: batch size   
        C: channel size
        T: frame length
        M: number of mel bins to predict
        Inputs:
            src:    [B, C, T] - source feature
            src_lf0 [B, T]    - source pitch
            trg:    [B, C, T] - target pitch
        Outputs:
            cnv:    [B, M, T] - conversion output
        """

        # length matching
        src_len = src.shape[-1]
        if src_len != src_lf0.shape[-1]:
            src = F.pad(src, (0, src_lf0.shape[-1] - src_len))
        
        # target shuffle for training strategy
        if self.training:
            trg = torch.cat((trg[:, :, src_len//2:], trg[:, :, :src_len//2]), axis=2)

        
        src_emds       = src_lf0.unsqueeze(-1)  # source pitch
        src            = self.cnt_encoder(src)  # source: cnt
        trg, trg_skips = self.spk_encoder(trg)  # target: spk

        src    = torch.cat([src.transpose(1,2), src_emds], dim=-1)
        src, _ = self.rnn_layer(src)
        src    = self.linear(src).transpose(1,2)

        output = self.decoder(src, trg, trg_skips)
        output = output + self.post_net(output)
        output = output[...,:src_len]
        
        return output