import os
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
from src.utils import *
from preprocess.audio import *


class TrainDataset(Dataset):
    def __init__(self, cfg, mode):
        
        self.eps       = 1e-8
        self.cfg       = cfg
        self.data_path = Path(cfg.data_path)
        self.n_frames  = cfg.setting.n_frames
        self.speakers  = sorted(os.listdir(self.data_path/f'{mode}/mels'))
        self.metadata  = Read_json(self.data_path/f'{mode}.json') # [info (mel_len, speaker, wav_path, mel_path, lf0_path, cpc_path, txt, txt_path, test_type)]
        mel_stats = np.load(cfg.data_path + '/mel_stats.npy')
        self.mean = np.expand_dims(mel_stats[0], -1)
        self.std = np.expand_dims(mel_stats[1], -1)
            
        print(f'Number of Sample Frames: {self.n_frames}, {mode} Data Size:', len(self.metadata))

    def normalize_lf0(self, lf0):      
        zero_idxs    = np.where(lf0 == 0)[0]
        nonzero_idxs = np.where(lf0 != 0)[0]
        if len(nonzero_idxs) > 0 :
            mean = np.mean(lf0[nonzero_idxs])
            std  = np.std(lf0[nonzero_idxs])
            if std == 0:
                lf0 -= mean
                lf0[zero_idxs] = 0.0
            else:
                lf0 = (lf0 - mean) / (std + 1e-8)
                lf0[zero_idxs] = 0.0
        return lf0

    def sample_frame(self, feat, mel, lf0):
        pos  = random.randint(0, feat.shape[-1] - self.n_frames)
        feat = feat[:, pos:pos + self.n_frames]
        mel  = mel[:, pos:pos + self.n_frames]
        lf0  = lf0[pos:pos + self.n_frames]  
        return feat, mel, lf0

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
              
        mel_path = self.metadata[index]['mel_path']
        lf0_path = self.metadata[index]['lf0_path']
        if self.cfg.train.cpc:
            feat_path = self.metadata[index]['cpc_path']
            feat      = np.load(feat_path).T
        else:
            feat_path = self.metadata[index]['mel_path']
            feat      = np.load(feat_path).T
            feat      = (feat - self.mean) / (self.std + self.eps)
            
        mel  = np.load(mel_path).T
        mel  = (mel - self.mean) / (self.std + self.eps)
        lf0  = np.load(lf0_path)

        if self.n_frames > feat.shape[-1]:
            feat_len = feat.shape[-1]
            feat     = np.tile(feat, int(np.ceil(self.n_frames / feat_len)))
            mel      = np.tile(mel, int(np.ceil(self.n_frames / feat_len)))
            lf0      = np.tile(lf0, int(np.ceil(self.n_frames / feat_len)))

        lf0            = self.normalize_lf0(lf0)
        feat, mel, lf0 = self.sample_frame(feat, mel, lf0)
        
        return {'feat': torch.from_numpy(feat),
                'mel': torch.from_numpy(mel),
                'lf0': torch.from_numpy(lf0)
                }
        
class ConversionDataset(Dataset):
    def __init__(self, cfg, mode):
        
        self.eps       = 1e-8
        self.cfg       = cfg
        self.data_path = Path(cfg.data_path)
        self.speakers  = sorted(os.listdir(self.data_path/f'{mode}/mels'))
        
        metadata       = Read_json(self.data_path/f'{mode}_pair.json')
        self.metadata  = metadata['s2s_st'] + metadata['s2s_ut'] + metadata['u2u_st'] + metadata['u2u_ut']
        mel_stats = np.load(cfg.data_path + '/mel_stats.npy')
        self.mean = np.expand_dims(mel_stats[0], -1)
        self.std  = np.expand_dims(mel_stats[1], -1)

        print(f'{mode} Data Size:', len(self.metadata))
            
    def normalize_lf0(self, lf0):      
        zero_idxs    = np.where(lf0 == 0)[0]
        nonzero_idxs = np.where(lf0 != 0)[0]
        if len(nonzero_idxs) > 0 :
            mean = np.mean(lf0[nonzero_idxs])
            std  = np.std(lf0[nonzero_idxs])
            if std == 0:
                lf0 -= mean
                lf0[zero_idxs] = 0.0
            else:
                lf0 = (lf0 - mean) / (std + 1e-8)
                lf0[zero_idxs] = 0.0
        return lf0
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index):
        
        src_info, trg_info, _ = self.metadata[index]  # source, target, oracle info
        
        assert src_info['test_type']==trg_info['test_type'], 'Test type shoud be the same in the pairs'
        test_type = src_info['test_type']
        src_spk, src_wav_name, src_mel_path, src_lf0_path = src_info['speaker'], src_info['wav_name'], src_info['mel_path'], src_info['lf0_path']
        trg_spk, trg_wav_name, trg_mel_path, trg_lf0_path = trg_info['speaker'], trg_info['wav_name'], trg_info['mel_path'], trg_info['lf0_path']
        
        if self.cfg.train.cpc:
            src_feat_path = src_info['cpc_path']
            trg_feat_path = trg_info['cpc_path']
            src_feat      = np.load(src_feat_path).T
            trg_feat      = np.load(trg_feat_path).T
        else:
            src_feat_path = src_info['mel_path']
            trg_feat_path = trg_info['mel_path']
            src_feat      = np.load(src_feat_path).T
            src_feat      = (src_feat - self.mean) / (self.std + self.eps)
            trg_feat      = np.load(trg_feat_path).T
            trg_feat      = (trg_feat - self.mean) / (self.std + self.eps)

        src_mel  = np.load(src_mel_path).T
        src_mel  = (src_mel - self.mean) / (self.std + self.eps)
        src_lf0  = self.normalize_lf0(np.load(src_lf0_path))

        trg_mel  = np.load(trg_mel_path).T
        trg_mel  = (trg_mel - self.mean) / (self.std + self.eps)
        
        return {'src_feat': torch.from_numpy(src_feat), 
                'src_lf0': torch.from_numpy(src_lf0), 
                'src_mel': torch.from_numpy(src_mel), 
                'src_wav_name': src_wav_name,
                'trg_feat': torch.from_numpy(trg_feat), 
                'trg_mel': torch.from_numpy(trg_mel), 
                'trg_wav_name': trg_wav_name,
                'test_type': test_type
                }

class MultiConversionDataset(Dataset):
    def __init__(self, cfg, mode):
        
        self.eps       = 1e-8
        self.cfg       = cfg
        self.data_path = Path(cfg.data_path)
        self.speakers  = sorted(os.listdir(self.data_path/f'{mode}/mels'))
        
        metadata       = Read_json(self.data_path/f'{mode}_{cfg.n_uttr}_pair.json')
        self.metadata  = metadata['s2s_st'] + metadata['s2s_ut'] + metadata['u2u_st'] + metadata['u2u_ut']
        mel_stats = np.load(cfg.data_path + '/mel_stats.npy')
        self.mean = np.expand_dims(mel_stats[0], -1)
        self.std  = np.expand_dims(mel_stats[1], -1)
        print(f'{mode} Data Size:', len(self.metadata))

    def get_multi_target(self, trg_info):
        
        trg_name = []
        trg_mel  = []
        trg_feat = []        
        for i in range(len(trg_info)):
            trg_name.append(trg_info[i]['wav_name'])
            mel = np.load(trg_info[i]['mel_path']).T
            mel = (mel - self.mean) / (self.std + self.eps)
            trg_mel.append(mel)
            if self.cfg.train.cpc:
                feat = np.load(trg_info[i]['cpc_path']).T
                trg_feat.append(feat)
            else:
                feat = np.load(trg_info[i]['mel_path']).T
                feat = (feat - self.mean) / (self.std + self.eps)
                trg_feat.append(feat)

        trg_feat = np.concatenate([i for i in trg_feat], axis=-1)

        return trg_feat, trg_mel, trg_name
        
    def normalize_lf0(self, lf0):      
        zero_idxs    = np.where(lf0 == 0)[0]
        nonzero_idxs = np.where(lf0 != 0)[0]
        if len(nonzero_idxs) > 0 :
            mean = np.mean(lf0[nonzero_idxs])
            std  = np.std(lf0[nonzero_idxs])
            if std == 0:
                lf0 -= mean
                lf0[zero_idxs] = 0.0
            else:
                lf0 = (lf0 - mean) / (std + 1e-8)
                lf0[zero_idxs] = 0.0
        return lf0
    
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        
        src_info, trg_info, _ = self.metadata[index]  # source, target, oracle info
        test_type = src_info['test_type']
        src_spk, src_wav_name, src_mel_path, src_lf0_path = src_info['speaker'], src_info['wav_name'], src_info['mel_path'], src_info['lf0_path']
        if self.cfg.train.cpc:
            src_feat_path = src_info['cpc_path']
            src_feat      = np.load(src_feat_path).T
        else:
            src_feat_path = src_info['mel_path']
            src_feat      = np.load(src_feat_path).T
            src_feat      = (src_feat - self.mean) / (self.std + self.eps)

        src_mel  = np.load(src_mel_path).T
        src_mel  = (src_mel - self.mean) / (self.std + self.eps)
        src_lf0  = self.normalize_lf0(np.load(src_lf0_path))

        trg_feat, trg_mel, trg_wav_name = self.get_multi_target(trg_info)
        
        return {'src_feat': torch.from_numpy(src_feat), 
                'src_lf0': torch.from_numpy(src_lf0), 
                'src_mel': torch.from_numpy(src_mel), 
                'src_wav_name': src_wav_name,
                'trg_feat': torch.from_numpy(trg_feat), 
                'trg_mel': trg_mel, 
                'trg_wav_name': trg_wav_name,
                'test_type': test_type
                }
        
def get_multi_target_meta(cfg, mode='test'):

    pair        = Read_json(f'{cfg.data_path}/{mode}_pair.json')
    output_path = f'{cfg.data_path}/{mode}_{cfg.n_uttr}_pair.json'
    if not os.path.isfile(output_path):

        metadata = pair['s2s_st'] + pair['s2s_ut'] + pair['u2u_st'] + pair['u2u_ut']
        keys     = ['s2s_st', 's2s_ut', 'u2u_st', 'u2u_ut']
        for k in keys:
            meta = []
            for i in pair[k]:
                meta.append(i[0])
                meta.append(i[1])    
                
            uttr_list = list(set([i['wav_name'] for i in meta]))
            uttr_dict = {}
            for i in meta:
                uttr = i['wav_name']
                uttr_dict[uttr] = i

            spk_uttr_dict = defaultdict(list)
            for uttr_key in uttr_dict:
                spk = uttr_key.split('_')[0]
                spk_uttr_dict[spk].append(uttr_dict[uttr_key])
            
            for i in pair[k]:
                trg_spk   = i[1]['speaker']
                uttrs = spk_uttr_dict[trg_spk]

                if (len(uttrs) > 0) and (len(uttrs) < cfg.n_uttr):
                    uttrs = random.sample(uttrs, k=len(uttrs))
                elif len(uttrs) >= cfg.n_uttr:
                    uttrs = random.sample(uttrs, k=cfg.n_uttr)

                i[1] = uttrs
                i[2] = None

        Write_json(pair, output_path)