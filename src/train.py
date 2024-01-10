import numpy as np
import os
import json
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader 
import torchaudio

from src.dataset import *
from src.utils import *

from model import TriAANVC
from src.evaluation import *
from src.augment import *
from tqdm import tqdm

class Trainer:

    def __init__(self, data, cfg):

        self.cfg       = cfg
        self.model     = TriAANVC(cfg.model.encoder, cfg.model.decoder).to(cfg.device)
        self.criterion = self._select_loss().to(cfg.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(cfg.train.lr))
        self.augment   = Augment()
        
        self.train_loader = data['train']
        self.val_loader   = data['valid']

        self.tester = Tester(cfg)

        # Write param size & [model, conv_module, config files]
        param_size          = count_parameters(self.model)
        self.cfg.param_size = np.round(param_size/1000000, 2)
        print(f'Param size: {cfg.param_size}M')
        Write_yaml(cfg.todict(), f'{cfg.checkpoint}/cfg.yaml')

        # logging
        if cfg.logging:
            print('---logging start---')
            neptune_load(get_cfg_params(cfg))
            
        # checkpoint
        if cfg.resume:
            self._resume_checkpoint()
            
    def _resume_checkpoint(self):
        checkpoint = torch.load(f'{self.cfg.checkpoint}/model-last.pth', map_location=self.cfg.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer']) 
        print('---load previous weigths and optimizer for resume training---')
        
    def _save_checkpoint(self, scores, epoch, opt='best'):
        checkpoint = {'scores':     scores,
                      'state_dict': self.model.state_dict(),
                      'optimizer':  self.optimizer.state_dict()}        
        if opt=='best':
            torch.save(checkpoint, f'{self.cfg.checkpoint}/{self.cfg.model_name}')
        elif opt=='last':
            torch.save(checkpoint, f'{self.cfg.checkpoint}/model-last.pth')        
        else:
            torch.save(checkpoint, f'{self.cfg.checkpoint}/model-{epoch}.pth')

    def _select_loss(self):
        if self.cfg.train.loss == 'l1':
            criterion =nn.L1Loss()
        elif self.cfg.train.loss == 'l2':
            criterion =nn.MSELoss()
        return criterion

    def train(self):
        
        best_loss = 1000000
        for epoch in range(1, self.cfg.train.epoch+1):
            
            self.model.train()
            train_loss = self._run_epoch(self.train_loader)               

            self.model.eval()
            with torch.no_grad():
                val_loss = self._run_epoch(self.val_loader, valid=True)

            if val_loss < best_loss:
                best_loss = val_loss
                self._save_checkpoint([best_loss], epoch, opt='best')      

            if epoch % self.cfg.train.save_epoch == 0:
                self._save_checkpoint([val_loss], epoch, opt='epoch')               
            
            self._save_checkpoint([val_loss], epoch, opt='last')
            print("epoch: {:03d} | trn loss: {:.4f} | valid loss: {:.4f}".format(epoch, train_loss, val_loss))

            if self.cfg.logging == True:
                neptune.log_metric('cur epoch', epoch)
                neptune.log_metric('train loss', train_loss)
                neptune.log_metric('valid loss', val_loss)
                
            if epoch % self.cfg.train.eval_every == 0:
                self.tester.test(set_type='valid')
                
    def _run_epoch(self, data_loader, valid=False):
        
        total_loss = 0
        num_batches = len(data_loader)
        for i, batch in enumerate(tqdm(data_loader)):
            
            src_feat = batch['feat'].to(self.cfg.device)
            src_lf0  = batch['lf0'].to(self.cfg.device)
            src_mel  = batch['mel'].to(self.cfg.device)

            output = self.model(src_feat, src_lf0, src_feat)
            loss   = self.criterion(output, src_mel)         # reconstruction loss

            if self.cfg.train.siam:
                siam_x, siam_lf0 = self.augment.spec_augment(src_feat.clone(), src_lf0.clone())
                output_siam = self.model(siam_x, siam_lf0, siam_x)

                loss1 = loss.clone()                         # reconstruction loss
                loss2 = self.criterion(output_siam, src_mel) # siam reconstruction loss
                loss3 = self.criterion(output, output_siam)  # consistency loss between prediction and siam's prediction
                loss  = (loss1 + loss2) * 0.5 + loss3


            loss.backward()

            if not valid:
                loss.backward()
                if (i+1) % self.cfg.train.accum_step == 0 or (i+1) == num_batches:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
          
            total_loss += loss.item()

        return total_loss / (i+1)



