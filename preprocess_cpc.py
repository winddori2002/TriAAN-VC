import warnings
warnings.filterwarnings(action='ignore')
import os
from os.path import join as opj
import json
from pathlib import Path
import numpy as np
import random
from joblib import Parallel, delayed
from tqdm import tqdm

from src.utils import *
from src.cpc import *
from preprocess.audio import *

def _load_wav(path):
    
    wav, fs = sf.read(path)
    wav, _   = librosa.effects.trim(y=wav, top_db=60)
    if fs != 16000:
        wav = resampy.resample(x=wav, sr_orig=fs, sr_new=16000, axis=0)

    peak = np.abs(wav).max()
    if peak > 1.0:
        wav /= peak

    wav  = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0).float()
    
    return wav

def main(cfg):
    
    data_path = Path(cfg.data_path)
    cpc       = load_cpc(f'{cfg.cpc_path}/cpc.pt').cuda()
    cpc.eval()
    with torch.no_grad():
        modes = ['train', 'valid', 'test']
        for mode in modes:
            metadata = Read_json(data_path/f'{mode}.json')

            for i in tqdm(range(len(metadata))):
                wav       = _load_wav(metadata[i]['wav_path']).cuda()
                feat      =  cpc(wav, None)[0].squeeze().detach().cpu().numpy()
                save_path = metadata[i]['mel_path'].replace('mels', 'cpc')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.save(save_path, feat)
                metadata[i]['cpc_path'] = save_path
                
            Write_json(metadata, data_path/f'{mode}.json')
                                        

if __name__ == '__main__':
    
    cfg = Config('./config/base.yaml')
    main(cfg)