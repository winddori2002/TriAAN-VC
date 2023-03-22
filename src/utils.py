import os
from os.path import join as opj
import json
import numpy as np
import random
import torch
import neptune
import pickle
import yaml
import fnmatch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def MakeDir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def neptune_load(PARAMS):
    """
    logging: write your neptune account/project, api topken
    """
    neptune.init('ID/ProjectName', api_token='')
    neptune.create_experiment(name=PARAMS['ex_name'], params=PARAMS)
            
def set_experiment(args, cfg):
    
    args = get_params(args)
    for key in args:
        cfg[key] = args[key]
    cfg.ex_name        = os.path.basename(os.getcwd())
    cfg.converted_path = opj(cfg.checkpoint, f'converted_{cfg.n_uttr}_uttr')
    MakeDir(cfg.checkpoint)
    MakeDir(cfg.converted_path)
        
    return cfg
           
def get_params(args):
    
    params    = {}
    args_ref  = vars(args)
    args_keys = vars(args).keys()

    for key in args_keys:
        if '__' in key:
            continue
        else:
            temp_params = args_ref[key]
            if type(temp_params) == dict:
                params.update(temp_params)
            else:
                params[key] = temp_params            
    return params

def get_cfg_params(cfg): 
    
    params   = {}
    cfg_ref  = cfg
    cfg_keys = cfg.keys()
    for key in cfg_keys:

        temp_params = cfg_ref[key]
        if type(temp_params) == DotDict:
            params.update(temp_params)
        else:
            params[key] = temp_params            
            
    return params

def Write_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def Read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data
        
def Write_pkl(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def Read_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def Write_yaml(data, path):
    with open(path, 'w') as f:
        yaml.dump(data, f)

def seed_init(seed=100):
    
    random.seed(seed)  
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)  
    
    
class DotDict(dict):
    """
    a dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct={}):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value
            
    def __getstate__(self):
        return self.__dict__
    
    def __setstate__(self, dct):
        self.__dict__ = dct
        
    def todict(self):
        dct = {}
        for k, v in self.items():
            if issubclass(type(v), DotDict):
                v = v.todict()
            dct[k] = v
        return dct

class Config(DotDict):
    
    @staticmethod
    def yaml_load(path):
        ret = yaml.safe_load(open(path, 'r', encoding='utf8'))
        assert ret is not None, f'Config file {path} is empty.'
        return Config(ret)
    
    @staticmethod
    def trans(inp, dep=0):
        ret = ''
        if issubclass(type(inp), dict):
            for k, v in inp.items():
                ret += f'\n{"    "*dep}{k}: {Config.trans(v, dep+1)}'
        elif issubclass(type(inp), list):
            for v in inp:
                ret += f'\n{"    "*dep}- {v}'
        else:
            ret += f'{inp}'
        return ret
    
    def __init__(self, dct):
        if type(dct) is str:
            dct = Config.yaml_load(dct)
        super().__init__(dct)
        try:
            self._name = dct['_name']
        except:
            self._name = 'Config'
            
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        ret = f'[{self._name}]'
        ret += Config.trans(self)
        return ret


    def _apply_config(self, config, replace=False):
        for k, v in config.items():
            self[k] = v

    def __getattr__(self, name):
        try:
            return self[name]
        except:
            raise AttributeError(name)
        
def find_files(root_dir, query="*.wav", include_root_dir=True):
    files = []
    for root, dirnames, filenames in os.walk(root_dir, followlinks=True):
        for filename in fnmatch.filter(filenames, query):
            files.append(os.path.join(root, filename))
    if not include_root_dir:
        files = [file_.replace(root_dir + "/", "") for file_ in files]

    return files