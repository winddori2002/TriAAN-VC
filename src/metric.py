import logging
from fastdtw import fastdtw
import librosa
import numpy as np
import pysptk
import pyworld as pw
import scipy
from scipy.io import wavfile
from scipy.signal import firwin
from scipy.signal import lfilter
from torch._C import ErrorReport
from tqdm import tqdm
from src.utils import *

import jiwer
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, Wav2Vec2Processor
from resemblyzer import preprocess_wav, VoiceEncoder

def GetTestMetaInfo(cfg, mode):
    
    if cfg.n_uttr > 1:
        test_pair = Read_json(f'{cfg.data_path}/{mode}_{cfg.n_uttr}_pair.json')
    else:
        test_pair = Read_json(f'{cfg.data_path}/{mode}_pair.json')

    meta_dict = {'s2s_st':[], 's2s_ut':[],'u2u_st':[],'u2u_ut':[]}
    for key in test_pair:

        cnv_files = find_files(f'{cfg.converted_path}/{mode}/{key}', query='*cnv*.wav')
        for (src_info, trg_info, orc_path) in test_pair[key]:
            
            src_path = src_info['wav_path']
            src_name = src_info['wav_name']
            src_txt  = src_info['text']

            if cfg.n_uttr > 1:
                trg_path = [trg['wav_path'] for trg in trg_info]
                trg_name = [trg['wav_name'] for trg in trg_info]
                trg_name = '_'.join(trg_name)
                cnv_name = f'from_{src_name}_to_{trg_name}'
            else:
                trg_path = trg_info['wav_path']
                trg_name = trg_info['wav_name']
                cnv_name = f'{src_name}_{trg_name}'
            
            for cnv_path in cnv_files:
                if cnv_name in cnv_path:
                    break
            meta_dict[key].append([cnv_path, src_path, trg_path, orc_path, src_txt]) # [cnv_path, src_path, trg_path, orc_path, src_txt]
    return meta_dict   

class VC_Evaluate:
    
    def __init__(self, meta_dict, cfg):
        self.cfg       = cfg
        self.meta_dict = meta_dict
        
        self.asv_model = VoiceEncoder(device=cfg.device)
        self.asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self").to(cfg.device)
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
        
    def transcribe(self, wav):
        
        wav, _         = librosa.load(wav, sr=16000)
        inputs         = self.tokenizer(wav, sampling_rate=16000, return_tensors="pt", padding="longest")
        input_values   = inputs.input_values.to(self.cfg.device)
        attention_mask = inputs.attention_mask.to(self.cfg.device)

        logits        = self.asr_model(input_values, attention_mask=attention_mask).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.tokenizer.batch_decode(predicted_ids)[0]

        return transcription
        
    def calculate_wer_cer(self, gt, pred):
        
        gt   = normalize_sentence(gt)
        pred = normalize_sentence(pred)
        cer  = jiwer.cer(gt, pred)
        wer  = jiwer.wer(gt, pred)     
        
        return cer, wer
        
    def calculate_asr_score(self):
    
        asr_score = {}
        for i, key in enumerate(tqdm(self.meta_dict)):
            
            total_cer = 0
            total_wer = 0
            for (cnv, src, trg, orc, txt) in self.meta_dict[key]:
                
                pred      = self.transcribe(cnv)
                cer, wer  = self.calculate_wer_cer(txt, pred)
                total_cer += cer
                total_wer += wer

            asr_score[key] = [total_cer/len(self.meta_dict[key]), total_wer/len(self.meta_dict[key])]
            
        return asr_score
    
    def calculate_accept(self, cnv, trg):
        
        cnv     = preprocess_wav(cnv)
        cnv_emb = self.asv_model.embed_utterance(cnv)

        if self.cfg.n_uttr > 1:
            trg     = [preprocess_wav(i) for i in trg] 
            trg_emb = self.asv_model.embed_speaker(trg)
        else:
            trg     = preprocess_wav(trg)
            trg_emb = self.asv_model.embed_utterance(trg)

        cos = np.inner(cnv_emb, trg_emb) / (np.linalg.norm(cnv_emb) * np.linalg.norm(trg_emb))
        acp = cos > self.cfg.test.threshold
        
        return acp, cos
    
    def calculate_asv_score(self):
        
        asv_score = {}
        for i, key in enumerate(tqdm(self.meta_dict)):
            
            n_accept = 0
            cos_sum  = 0
            for (cnv, src, trg, orc, txt) in self.meta_dict[key]:

                acp, cos = self.calculate_accept(cnv, trg)
                n_accept += acp
                cos_sum  += cos
            asv_score[key] = [n_accept/len(self.meta_dict[key]), cos_sum/len(self.meta_dict[key])]
            
        return asv_score
    
    def forward(self):

        asr_score = self.calculate_asr_score()
        asv_score = self.calculate_asv_score()
        
        return asr_score, asv_score

def normalize_sentence(sentence):
    """Normalize sentence"""
    # Convert all characters to upper.
    sentence = sentence.upper()
    # Delete punctuations.
    sentence = jiwer.RemovePunctuation()(sentence)
    # Remove \n, \t, \r, \x0c.
    sentence = jiwer.RemoveWhiteSpace(replace_by_space=True)(sentence)
    # Remove multiple spaces.
    sentence = jiwer.RemoveMultipleSpaces()(sentence)
    # Remove white space in two end of string.
    sentence = jiwer.Strip()(sentence)

    # Convert all characters to upper.
    sentence = sentence.upper()

    return sentence