# -*- coding: utf-8 -*-
# Modifications
# 
# Original copyright:
# The copyright is under MIT license from VQMIVC.
# VQMIVC (https://github.com/Wendison/VQMIVC) / author: Wendison


import warnings
warnings.filterwarnings(action='ignore')

import os
from os.path import join as opj
import json
import random
import numpy as np
import librosa
import soundfile as sf
from joblib import Parallel, delayed
from glob import glob
from tqdm import tqdm
import resampy
import pyworld as pw
from datasets import load_dataset

from preprocess.spectrogram import logmelspectrogram

def ProcessingTrainData(path, cfg):
    
    """
        For multiprocess function binding load wav and log-mel 
    """
    wav_name = os.path.basename(path).split('.')[0]
    speaker  = wav_name.split('-')[0]
    sr       = cfg.sampling_rate
    wav, fs  = sf.read(path)
    wav, _   = librosa.effects.trim(y=wav, top_db=cfg.top_db) # trim slience

    if fs != sr:
        wav = resampy.resample(x=wav, sr_orig=fs, sr_new=sr, axis=0)
        fs  = sr
        
    assert fs == 16000, 'Downsampling needs to be done.'
    
    peak = np.abs(wav).max()
    if peak > 1.0:
        wav /= peak
        
    mel = logmelspectrogram(
                            x=wav,
                            fs=cfg.sampling_rate,
                            n_mels=cfg.n_mels,
                            n_fft=cfg.n_fft,
                            n_shift=cfg.n_shift,
                            win_length=cfg.win_length,
                            window=cfg.window,
                            fmin=cfg.fmin,
                            fmax=cfg.fmax
                            )
    tlen         = mel.shape[0]
    frame_period = cfg.n_shift/cfg.sampling_rate*1000
    
    f0, timeaxis = pw.dio(wav.astype('float64'), cfg.sampling_rate, frame_period=frame_period)
    f0           = pw.stonemask(wav.astype('float64'), f0, timeaxis, cfg.sampling_rate)
    f0           = f0[:tlen].reshape(-1).astype('float32')
    
    nonzeros_indices      = np.nonzero(f0)
    lf0                   = f0.copy()
    lf0[nonzeros_indices] = np.log(f0[nonzeros_indices]) # for f0(Hz), lf0 > 0 when f0 != 0
    
    return wav_name, mel, lf0, mel.shape[0], speaker

def LoadWav(path, cfg):
    
    """
        load raw wav from the path -> processed wav
    """
    # skip pre-emphasis
    wav_name = os.path.basename(path).split('.')[0]
    sr       = cfg.sampling_rate
    wav, fs  = sf.read(path)
    wav, _   = librosa.effects.trim(y=wav, top_db=cfg.top_db) # trim slience

    if fs != sr:
        wav = resampy.resample(x=wav, sr_orig=fs, sr_new=sr, axis=0)
        fs  = sr
        
    assert fs == 16000, 'Downsampling needs to be done.'
    
    peak = np.abs(wav).max()
    if peak > 1.0:
        wav /= peak

    return wav, wav_name

def GetLogMel(wav, cfg):

    """
        load log mel from the wav -> mel, f0, mel length
    """
    mel = logmelspectrogram(
                            x=wav,
                            fs=cfg.sampling_rate,
                            n_mels=cfg.n_mels,
                            n_fft=cfg.n_fft,
                            n_shift=cfg.n_shift,
                            win_length=cfg.win_length,
                            window=cfg.window,
                            fmin=cfg.fmin,
                            fmax=cfg.fmax
                            )
    
    tlen         = mel.shape[0]
    frame_period = cfg.n_shift/cfg.sampling_rate*1000
    
    f0, timeaxis = pw.dio(wav.astype('float64'), cfg.sampling_rate, frame_period=frame_period)
    f0           = pw.stonemask(wav.astype('float64'), f0, timeaxis, cfg.sampling_rate)
    f0           = f0[:tlen].reshape(-1).astype('float32')
    
    nonzeros_indices      = np.nonzero(f0)
    lf0                   = f0.copy()
    lf0[nonzeros_indices] = np.log(f0[nonzeros_indices]) # for f0(Hz), lf0 > 0 when f0 != 0
    
    return mel, lf0, mel.shape[0]


def NormalizeLogMel(wav_name, mel, mean, std):
    mel = (mel - mean) / (std + 1e-8)
    return wav_name, mel

def TextCheck(wavs, cfg):
    
    wav_files = [i.split('_mic1')[0] for i in wavs]
    
    txt_path  = glob(f'{cfg.txt_path}/*/*')
    txt_files = [os.path.basename(i).split('.txt')[0] for i in txt_path]
    
    revised_wavs = []
    for i in range(len(wavs)):
        if wav_files[i] in txt_files:
            revised_wavs.append(wavs[i])
            
    return revised_wavs

def GetSpeakerInfo(cfg):
    
    spk_info = open(cfg.spk_info_path, 'r')
    gen2spk  = {}
    all_spks = []
    for i, line in enumerate(spk_info):
        if i == 0:
            continue
        else:
            tmp = line.split()
            spk = tmp[0]
            all_spks.append(spk)
            gen = tmp[2]
            if gen not in gen2spk:
                gen2spk[gen] = [spk]
            else:
                gen2spk[gen].append(spk)    
    
    print(f'Total speaker: {len(all_spks)} with Female: {len(gen2spk["F"])} and Male: {len(gen2spk["M"])}')
    
    return all_spks, gen2spk

def GetSpeakerInfoHF(cfg):
    
    dataset = load_dataset(cfg.hf_dataset_path, split=cfg.hf_dataset_split)

    all_spks = dataset.unique('client_id')
    ds_m = dataset.filter(lambda x: x['gender'] == 'male')
    ds_f = dataset.filter(lambda x: x['gender'] == 'female')
    gen2spk = {}
    gen2spk['M'] = ds_m.unique('client_id')
    gen2spk['F'] = ds_f.unique('client_id')

    print(f'Total speaker: {len(all_spks)} with Female: {len(gen2spk["F"])} and Male: {len(gen2spk["M"])}')

    return all_spks, gen2spk

def SplitDataset(all_spks, cfg):
    
    all_spks = sorted(all_spks)
    random.shuffle(all_spks)
    train_spks = all_spks[:-cfg.eval_spks * 2] # except valid and test unseen speakers
    valid_spks = all_spks[-cfg.eval_spks * 2:-cfg.eval_spks]
    test_spks  = all_spks[-cfg.eval_spks:]

    train_wavs_names = []
    valid_wavs_names = []
    test_wavs_names  = []
    
    for spk in train_spks:
        spk_wavs       = glob(f'{cfg.data_path}/{spk}/*mic1*')
        spk_wavs_names = [os.path.basename(p).split('.')[0] for p in spk_wavs]
        valid_names    = random.sample(spk_wavs_names, int(len(spk_wavs_names) * cfg.s2s_portion))
        train_names    = [n for n in spk_wavs_names if n not in valid_names]
        test_names     = random.sample(train_names, int(len(spk_wavs_names) * cfg.s2s_portion))
        train_names    = [n for n in train_names if n not in test_names]

        train_wavs_names += train_names
        valid_wavs_names += valid_names
        test_wavs_names  += test_names

    for spk in valid_spks:
        spk_wavs         = glob(f'{cfg.data_path}/{spk}/*mic1*')
        spk_wavs_names   = [os.path.basename(p).split('.')[0] for p in spk_wavs]
        valid_wavs_names += spk_wavs_names

    for spk in test_spks:
        spk_wavs        = glob(f'{cfg.data_path}/{spk}/*mic1*')
        spk_wavs_names  = [os.path.basename(p).split('.')[0] for p in spk_wavs]
        test_wavs_names += spk_wavs_names
    
    all_wavs         = glob(f'{cfg.data_path}/*/*mic1.flac')
    train_wavs_names = TextCheck(train_wavs_names, cfg) # delete the wavs which don't have text files
    valid_wavs_names = TextCheck(valid_wavs_names, cfg)
    test_wavs_names  = TextCheck(test_wavs_names, cfg)
    
    print(f'Total files: {len(all_wavs)}, Train: {len(train_wavs_names)}, Valid: {len(valid_wavs_names)}, Test: {len(test_wavs_names)}, Del Files: {len(all_wavs)-len(train_wavs_names)-len(valid_wavs_names)-len(test_wavs_names)}')
    
    return all_wavs, train_wavs_names, valid_wavs_names, test_wavs_names

def SplitDatasetHF(all_spks, cfg):

    dataset = load_dataset(cfg.hf_dataset_path, split=cfg.hf_dataset_split)

    all_spks = sorted(all_spks)
    random.shuffle(all_spks)
    train_spks = all_spks[:-cfg.eval_spks * 2] # except valid and test unseen speakers
    valid_spks = all_spks[-cfg.eval_spks * 2:-cfg.eval_spks]
    test_spks  = all_spks[-cfg.eval_spks:]

    train_wavs_names = []
    valid_wavs_names = []
    test_wavs_names  = []
    
    for spk in train_spks:
        spk_wavs       = dataset.filter(lambda x: x['client_id'] == spk)['path']
        spk_wavs_names = [os.path.basename(p).split('.')[0] for p in spk_wavs]
        valid_names    = random.sample(spk_wavs_names, int(len(spk_wavs_names) * cfg.s2s_portion))
        train_names    = [n for n in spk_wavs_names if n not in valid_names]
        test_names     = random.sample(train_names, int(len(spk_wavs_names) * cfg.s2s_portion))
        train_names    = [n for n in train_names if n not in test_names]

        train_wavs_names += train_names
        valid_wavs_names += valid_names
        test_wavs_names  += test_names

    for spk in valid_spks:
        spk_wavs         = dataset.filter(lambda x: x['client_id'] == spk)['path']
        spk_wavs_names   = [os.path.basename(p).split('.')[0] for p in spk_wavs]
        valid_wavs_names += spk_wavs_names

    for spk in test_spks:
        spk_wavs        = dataset.filter(lambda x: x['client_id'] == spk)['path']
        spk_wavs_names  = [os.path.basename(p).split('.')[0] for p in spk_wavs]
        test_wavs_names += spk_wavs_names
    
    all_wavs = dataset['path']
    
    print(f'Total files: {len(all_wavs)}, Train: {len(train_wavs_names)}, Valid: {len(valid_wavs_names)}, Test: {len(test_wavs_names)}, Del Files: {len(all_wavs)-len(train_wavs_names)-len(valid_wavs_names)-len(test_wavs_names)}')
    
    return all_wavs, train_wavs_names, valid_wavs_names, test_wavs_names


def GetMetaResults(train_results, valid_results, test_results, cfg):
    """
    This is for making additional metadata [txt, text_path, test_type] -1:train, 0:s2s_st, 1:s2s_ut, 2:u2u_st, 3:u2u_ut
    """

    for i in range(len(train_results)):
        
        spk      = train_results[i]['speaker']  # p225
        wav_name = train_results[i]['wav_name'] # p225_001
        txt_path = f'{cfg.txt_path}/{spk}/{wav_name}.txt' 
        
        file    = open(txt_path)
        scripts = file.readline()
        file.close()
       
        train_results[i]['text']      = scripts
        train_results[i]['text_path'] = txt_path
        train_results[i]['test_type'] = 'train'
        
    for i in range(len(valid_results)):
        
        spk      = valid_results[i]['speaker']  # p225
        wav_name = valid_results[i]['wav_name'] # p225_001
        txt_path = f'{cfg.txt_path}/{spk}/{wav_name}.txt' 
        
        file    = open(txt_path)
        scripts = file.readline()
        file.close()
       
        valid_results[i]['text']      = scripts
        valid_results[i]['text_path'] = txt_path
        
    for i in range(len(test_results)):
        
        spk      = test_results[i]['speaker']  # p225
        wav_name = test_results[i]['wav_name'] # p225_001
        txt_path = f'{cfg.txt_path}/{spk}/{wav_name}.txt' 
        
        file    = open(txt_path)
        scripts = file.readline()
        file.close()
       
        test_results[i]['text']      = scripts
        test_results[i]['text_path'] = txt_path

        
    train_spk = set([i['speaker'] for i in train_results])
    valid_spk = set([i['speaker'] for i in valid_results])
    test_spk  = set([i['speaker'] for i in test_results])

    train_txt = set([i['text'] for i in train_results])
    valid_txt = set([i['text'] for i in valid_results])
    test_txt  = set([i['text'] for i in test_results])

    valid_s2s_spk = train_spk.intersection(valid_spk) 
    valid_u2u_spk = valid_spk.difference(train_spk).difference(test_spk)

    test_s2s_spk  = train_spk.intersection(test_spk)
    test_u2u_spk  = test_spk.difference(train_spk).difference(valid_spk)

    valid_s2s_txt = train_txt.intersection(valid_txt) 
    valid_u2u_txt = valid_txt.difference(train_txt).difference(test_txt)

    test_s2s_txt  = train_txt.intersection(test_txt)
    test_u2u_txt  = test_txt.difference(train_txt).difference(valid_txt)
    
    for i in range(len(valid_results)):
        
        spk, txt = valid_results[i]['speaker'], valid_results[i]['text']
        if spk in valid_s2s_spk:
            if txt in valid_s2s_txt:
                valid_results[i]['test_type'] = 's2s_st'
            else:
                valid_results[i]['test_type'] = 's2s_ut'
        else:
            if txt in valid_s2s_txt:
                valid_results[i]['test_type'] = 'u2u_st'
            else:
                valid_results[i]['test_type'] = 'u2u_ut'

    for i in range(len(test_results)):
        
        spk, txt = test_results[i]['speaker'], test_results[i]['text']
        if spk in test_s2s_spk:
            if txt in test_s2s_txt:
                test_results[i]['test_type'] = 's2s_st'
            else:
                test_results[i]['test_type'] = 's2s_ut'
        else:
            if txt in test_s2s_txt:
                test_results[i]['test_type'] = 'u2u_st'
            else:
                test_results[i]['test_type'] = 'u2u_ut'
                
    return train_results, valid_results, test_results

def ExtractMelstats(wn2info, train_wavs_names, cfg):
    
    mels = []
    for wav_name in train_wavs_names:
        mel, *_ = wn2info[wav_name]
        mels.append(mel)   
        
    mels      = np.concatenate(mels, 0)
    mean      = np.mean(mels, 0)
    std       = np.std(mels, 0)
    mel_stats = np.concatenate([mean.reshape(1,-1), std.reshape(1,-1)], 0)    
    print('---Extract Mel statistics and save---')
    np.save(opj(cfg.output_path, 'mel_stats.npy'), mel_stats)
    
    return mean, std

def SaveFeatures(wav_name, info, mode, cfg):
    
    mel, lf0, mel_len, speaker = info
    wav_path      = f'{cfg.data_path}/{speaker}/{wav_name}.flac' # can change to special char *
    mel_save_path = f'{cfg.output_path}/{mode}/mels/{speaker}/{wav_name}.npy'
    lf0_save_path = f'{cfg.output_path}/{mode}/lf0/{speaker}/{wav_name}.npy'
    
    os.makedirs(os.path.dirname(mel_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(lf0_save_path), exist_ok=True)
    np.save(mel_save_path, mel)
    np.save(lf0_save_path, lf0)
    
    wav_name = wav_name.split('_mic')[0] # p231_001

    return {'mel_len':mel_len, 'speaker':speaker, 'wav_name':wav_name, 'wav_path':wav_path, 'mel_path':mel_save_path, 'lf0_path':lf0_save_path}
    