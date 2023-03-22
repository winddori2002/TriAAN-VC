import os
import sys

# os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
# os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
# os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6

import warnings
warnings.filterwarnings('ignore')

import json
import yaml
import argparse

from src.train import *
from src.dataset import *
from src.utils import *
from config import *

def main(cfg):
    
    seed_init(seed=cfg.seed)
    if args.action == 'train':
        
        print('--- Train Phase ---')
        
        train_dataset = TrainDataset(cfg, 'train')
        train_loader  = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.num_worker) 
        val_dataset   = TrainDataset(cfg, 'valid')
        val_loader    = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.num_worker)
        
        data_loader   = {'train':train_loader, 'valid':val_loader}
        
        trainer = Trainer(data_loader, cfg)
        trainer.train()
        
        print('--- Test Phase ---')
        seed_init(seed=cfg.seed)
        tester = Tester(cfg)
        tester.test(set_type='test')

        if cfg.logging:
            neptune.stop()

    else:
        print('--- Test Phase ---')
        tester = Tester(cfg)
        tester.test(set_type='test')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, default='train', help='Action') # train / test
    parser.add_argument('--config', default='./config/base.yaml', help='config yaml file')
    parser.add_argument('--num_worker', type=int, default=0, help='Num workers')
    parser.add_argument('--seed', type=int, default=1234, help='seed number')
    parser.add_argument('--device', type=str, default='cuda:0', help='Cuda device')
    parser.add_argument('--logging', type=bool, default=False, help='Logging option')
    parser.add_argument('--resume', type=bool, default=False, help='Resume option')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints', help='Results save path')
    parser.add_argument('--model_name', type=str, default='model-best.pth', help='Best model name')
    parser.add_argument('--n_uttr', type=int, default=1, help='Number of target utterances') # default:1 for a fair comparison
    
    args = parser.parse_args()
    cfg  = Config(args.config)
    cfg  = set_experiment(args, cfg) # merge arg and cfg, make directories
    print(cfg)
   
    main(cfg)
