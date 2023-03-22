import numpy as np
import random
import torch

class Augment:
    def __init__(self, frame_len=128, freq_mask_num=1, freq_mask_para=50, time_mask_num=1, time_mask_para=27, freq_mask=False, time_mask=True):

        self.frame_len      = frame_len
        self.freq_mask_num  = freq_mask_num
        self.freq_mask_para = freq_mask_para
        self.time_mask_num  = time_mask_num
        self.time_mask_para = time_mask_para
        self.freq_mask      = freq_mask
        self.time_mask      = time_mask 
        # self.frame_shift    = frame_shift

    def spec_augment(self, feat, lf0):
        v   = feat.shape[1]  # freq
        tau = feat.shape[2]  # time

        if self.freq_mask:
            for i in range(self.freq_mask_num):
                f  = np.random.uniform(low=0.0, high=self.freq_mask_para)
                f  = int(f)
                f0 = random.randint(0, v-f)
                feat[:, f0:f0+f, :] = 0

        if self.time_mask:
            for i in range(self.time_mask_num):
                t  = np.random.uniform(low=0.0, high=self.time_mask_para)
                t  = int(t)
                t0 = random.randint(0, tau - t)
                feat[:, :, t0:t0 + t]  = 0
                lf0[:, t0:t0 + t]      = 0

        return feat, lf0

    def shift_augment(self, feat):
        
        if self.frame_shift:
            idx  = int(np.random.uniform(self.frame_len)) 
            feat = torch.cat([feat[...,idx:], feat[...,:idx]], dim=-1)
        else:
            feat = torch.cat([feat[:,:,self.frame_len//2:], feat[:,:,:self.frame_len//2]], dim=-1)
        
        return feat
