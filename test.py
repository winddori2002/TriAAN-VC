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
from collections import defaultdict

from src.train import *
from src.dataset import *
from src.utils import *
from config import *


def main(cfg):
    
    seed_init(seed=cfg.seed)        

    if cfg.n_uttr > 1:
        print('--- Multi-target utterance conversion ---')
        get_multi_target_meta(cfg, mode='test')

    tester = Tester(cfg)
    print(f'--- Conversion True and Evaluation {cfg.eval} ---')
    if cfg.n_uttr > 1:
        tester.test_multi_target(set_type='test', evaluation=cfg.eval)
    else:
        tester.test(set_type='test', evaluation=cfg.eval)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/base.yaml', help='config yaml file')
    parser.add_argument('--device', type=str, default='cuda:0', help='Cuda device')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints', help='Results load path')
    parser.add_argument('--model_name', type=str, default='model-cpc-split.pth', help='Best model name')
    parser.add_argument('--n_uttr', type=int, default=1, help='Number of target utterances') # can be changed for real application
    parser.add_argument('--seed', type=int, default=1234, help='Seed')
    parser.add_argument('--eval', type=bool, default=True, help='Option for evaluation')  # need to be set as False for evaluation in test.py
    parser.add_argument('--logging', type=bool, default=False, help='Option for logging') # need to be set as False for evaluation in test.py
    
    args = parser.parse_args()
    cfg  = Config(args.config)
    cfg  = set_experiment(args, cfg)
    print(cfg)
   
    main(cfg)
