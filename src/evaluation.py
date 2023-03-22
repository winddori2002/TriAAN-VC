import numpy as np
import os
import torch
import torchaudio
import torch.nn as nn

import subprocess
import kaldiio
import neptune

from src.dataset import *
from src.utils import *
from src.metric import *
from src.vocoder import decode
from model import TriAANVC

class Tester:
    def __init__(self, cfg):
        
        self.cfg          = cfg
        self.criterion    = self._select_loss().to(cfg.device)

        self.mel_stats    = np.load(f'{cfg.data_path}/mel_stats.npy')
        self.val_dataset  = ConversionDataset(cfg, 'valid')
        self.val_loader   = DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=4)
        if cfg.n_uttr > 1:
            self.test_dataset = MultiConversionDataset(cfg, 'test')
            self.test_loader  = DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=4)
        else:
            self.test_dataset = ConversionDataset(cfg, 'test')
            self.test_loader  = DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=4)
        self.model = TriAANVC(cfg.model.encoder, cfg.model.decoder).to(cfg.device)
        
    def _select_loss(self):
        
        if self.cfg.train.loss == 'l1':
            criterion = nn.L1Loss()
                           
        return criterion
        
    def test(self, set_type='test', evaluation=True):
        
        checkpoint = torch.load(f'{self.cfg.checkpoint}/{self.cfg.model_name}', map_location=self.cfg.device)
        self.model.load_state_dict(checkpoint['state_dict'])
                
        if set_type == 'test':
            data_loader = self.test_loader
        else:
            data_loader = self.val_loader
            
        mean        = self.mel_stats[0]
        std         = self.mel_stats[1]
        output_dict = {'s2s_st':[], 's2s_ut':[], 'u2u_st':[], 'u2u_ut':[]}
        
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader)):
                
                src_feat = batch['src_feat'].to(self.cfg.device)
                src_lf0  = batch['src_lf0'].to(self.cfg.device)
                trg_feat = batch['trg_feat'].to(self.cfg.device)
                output   = self.model(src_feat, src_lf0, trg_feat)
                
                test_type    = batch['test_type'][0]
                src_mel      = batch['src_mel']
                trg_mel      = batch['trg_mel']
                src_name     = batch['src_wav_name'][0]
                trg_name     = batch['trg_wav_name'][0]
                out_filename = f'{src_name}_{trg_name}'
    
                output  = output.squeeze(0).cpu().numpy().T * (std + 1e-8) + mean  # (T, mels)
                src_mel = src_mel.squeeze(0).cpu().numpy().T * (std + 1e-8) + mean
                trg_mel = trg_mel.squeeze(0).cpu().numpy().T * (std + 1e-8) + mean

                output_dict[test_type].append([out_filename, output, src_mel, trg_mel])
        
        self.convert(output_dict, set_type=set_type)
        if evaluation:
            self.evaluate(set_type=set_type)
        
    def test_multi_target(self, set_type='test', evaluation=True):
        
        checkpoint = torch.load(f'{self.cfg.checkpoint}/{self.cfg.model_name}', map_location=self.cfg.device)
        self.model.load_state_dict(checkpoint['state_dict'])
                
        if set_type == 'test':
            data_loader = self.test_loader
        else:
            data_loader = self.val_loader
            
        mean        = self.mel_stats[0]
        std         = self.mel_stats[1]
        output_dict = {'s2s_st':[], 's2s_ut':[], 'u2u_st':[], 'u2u_ut':[]}
        
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader)):
                
                src_feat = batch['src_feat'].to(self.cfg.device)
                src_lf0  = batch['src_lf0'].to(self.cfg.device)
                trg_feat = batch['trg_feat'].to(self.cfg.device)
                output   = self.model(src_feat, src_lf0, trg_feat)

                test_type    = batch['test_type'][0]
                src_mel      = batch['src_mel']
                trg_mels     = batch['trg_mel']
                src_name     = batch['src_wav_name'][0]
                trg_name     = [i[0] for i in batch['trg_wav_name']]
                trg_name     = '_'.join(trg_name)
                out_filename = f'from_{src_name}_to_{trg_name}'
                    
                output   = output.squeeze(0).cpu().numpy().T * (std + 1e-8) + mean  # (T, mels)
                src_mel  = src_mel.squeeze(0).cpu().numpy().T * (std + 1e-8) + mean
                trg_mels = [trg_mel.squeeze(0).cpu().numpy().T * (std + 1e-8) + mean for trg_mel in trg_mels]
                
                output_dict[test_type].append([out_filename, output, src_mel] + trg_mels)
        
        self.convert(output_dict, set_type=set_type)
        if evaluation:
            self.evaluate(set_type=set_type)

    def convert(self, output_dict, set_type):
        
        for key in output_dict:
            output_list = output_dict[key]
            convert_dir = opj(self.cfg.converted_path, set_type, key)
            MakeDir(convert_dir)
            feat_writer = kaldiio.WriteHelper("ark,scp:{o}.ark,{o}.scp".format(o=str(convert_dir)+'/feats.1'))

            if self.cfg.n_uttr > 1:
                for (out_filename, output, src_mel, *trg_mels) in output_list:
                    feat_writer[out_filename+'_cnv'] = output
                    feat_writer[out_filename+'_src'] = src_mel
                    for num in range(len(trg_mels)):
                        feat_writer[out_filename+'_trg'+str(num)] = trg_mels[num]
            else:
                for (out_filename, output, src_mel, trg_mel) in output_list:
                    feat_writer[out_filename+'_cnv'] = output
                    feat_writer[out_filename+'_src'] = src_mel
                    feat_writer[out_filename+'_trg'] = trg_mel       
            feat_writer.close()
            print('synthesize waveform...')
            decode(f'{str(convert_dir)}/feats.1.scp', convert_dir, self.cfg.device)

    def evaluate(self, set_type):
            
        meta_dict = GetTestMetaInfo(self.cfg, mode=set_type)
        evaluater = VC_Evaluate(meta_dict, self.cfg)

        asr_score, asv_score = evaluater.forward()

        print(f'--- Set: {set_type} ---')
        print("CER: | s2s_st: {:.4f} | s2s_ut: {:.4f} | u2u_st: {:.4f} | u2u_ut: {:.4f}".format(asr_score['s2s_st'][0], asr_score['s2s_ut'][0], asr_score['u2u_st'][0], asr_score['u2u_ut'][0]))
        print("WER: | s2s_st: {:.4f} | s2s_ut: {:.4f} | u2u_st: {:.4f} | u2u_ut: {:.4f}".format(asr_score['s2s_st'][1], asr_score['s2s_ut'][1], asr_score['u2u_st'][1], asr_score['u2u_ut'][1]))
        print("ASV ACC: | s2s_st: {:.4f} | s2s_ut: {:.4f} | u2u_st: {:.4f} | u2u_ut: {:.4f}".format(asv_score['s2s_st'][0], asv_score['s2s_ut'][0], asv_score['u2u_st'][0], asv_score['u2u_ut'][0]))
        print("ASV COS: | s2s_st: {:.4f} | s2s_ut: {:.4f} | u2u_st: {:.4f} | u2u_ut: {:.4f}".format(asv_score['s2s_st'][1], asv_score['s2s_ut'][1], asv_score['u2u_st'][1], asv_score['u2u_ut'][1]))    
        
        s2s_cer = (asr_score['s2s_st'][0] + asr_score['s2s_ut'][0]) / 2
        u2u_cer = (asr_score['u2u_st'][0] + asr_score['u2u_ut'][0]) / 2
        
        s2s_wer = (asr_score['s2s_st'][1] + asr_score['s2s_ut'][1]) / 2
        u2u_wer = (asr_score['u2u_st'][1] + asr_score['u2u_ut'][1]) / 2  
        
        s2s_acc = (asv_score['s2s_st'][0] + asv_score['s2s_ut'][0]) / 2
        u2u_acc = (asv_score['u2u_st'][0] + asv_score['u2u_ut'][0]) / 2 
        
        s2s_cos = (asv_score['s2s_st'][1] + asv_score['s2s_ut'][1]) / 2
        u2u_cos = (asv_score['u2u_st'][1] + asv_score['u2u_ut'][1]) / 2 
            
        if self.cfg.logging == True:
            neptune.log_metric(f'{set_type} s2s_cer', s2s_cer)
            neptune.log_metric(f'{set_type} u2u_cer', u2u_cer)
            neptune.log_metric(f'{set_type} s2s_wer', s2s_wer)
            neptune.log_metric(f'{set_type} u2u_wer', u2u_wer)
            
            neptune.log_metric(f'{set_type} s2s_acc', s2s_acc)
            neptune.log_metric(f'{set_type} u2u_acc', u2u_acc)
            neptune.log_metric(f'{set_type} s2s_cos', s2s_cos)
            neptune.log_metric(f'{set_type} u2u_cos', u2u_cos)