import warnings
warnings.filterwarnings(action='ignore')

import random
from collections import defaultdict
from itertools import permutations
from src.utils import *

def GetSpeakerDict(info):

    spk_dict = {'s2s_st':defaultdict(list), 's2s_ut':defaultdict(list), 'u2u_st':defaultdict(list), 'u2u_ut':defaultdict(list)}
    for i in info:
        spk_dict[i['test_type']][i['speaker']].append(i)

    return spk_dict

def GeneratePairSample(spk_dic, num_samples):
    
    s2s_st_pairs = list(permutations(list(spk_dic['s2s_st'].keys()), 2)) # s2s_st pair
    s2s_st_pairs = random.choices(s2s_st_pairs, k=num_samples)

    s2s_ut_pairs = list(permutations(list(spk_dic['s2s_ut'].keys()), 2)) # s2s_ut pair
    s2s_ut_pairs = random.choices(s2s_ut_pairs, k=num_samples)

    u2u_st_pairs = list(permutations(list(spk_dic['u2u_st'].keys()), 2)) # u2u_st pair
    u2u_st_pairs = random.choices(u2u_st_pairs, k=num_samples)

    u2u_ut_pairs = list(permutations(list(spk_dic['u2u_ut'].keys()), 2)) # u2u_ut _pair
    u2u_ut_pairs = random.choices(u2u_ut_pairs, k=num_samples)
    
    info_dict = {'s2s_st':[], 's2s_ut':[], 'u2u_st':[], 'u2u_ut':[]}
    s2s_st_x  = 0
    s2s_ut_x  = 0
    u2u_st_x  = 0
    u2u_ut_x  = 0

    for (s2s_st_s, s2s_st_t), (s2s_ut_s, s2s_ut_t), (u2u_st_s, u2u_st_t), (u2u_ut_s, u2u_ut_t) in zip(s2s_st_pairs, s2s_ut_pairs, u2u_st_pairs, u2u_ut_pairs):

        info_dict['s2s_st'].append([random.choice(spk_dic['s2s_st'][s2s_st_s]), random.choice(spk_dic['s2s_st'][s2s_st_t])]) # sampling info of the seen source and target speaker
        info_dict['s2s_ut'].append([random.choice(spk_dic['s2s_ut'][s2s_ut_s]), random.choice(spk_dic['s2s_ut'][s2s_ut_t])])
        info_dict['u2u_st'].append([random.choice(spk_dic['u2u_st'][u2u_st_s]), random.choice(spk_dic['u2u_st'][u2u_st_t])])
        info_dict['u2u_ut'].append([random.choice(spk_dic['u2u_ut'][u2u_ut_s]), random.choice(spk_dic['u2u_ut'][u2u_ut_t])])   

        s2s_st_unq = [(i[0]['wav_name'], i[1]['wav_name']) for i in info_dict['s2s_st']] # (src wav name, trg wav name)
        s2s_ut_unq = [(i[0]['wav_name'], i[1]['wav_name']) for i in info_dict['s2s_ut']]
        u2u_st_unq = [(i[0]['wav_name'], i[1]['wav_name']) for i in info_dict['u2u_st']]
        u2u_ut_unq = [(i[0]['wav_name'], i[1]['wav_name']) for i in info_dict['u2u_ut']]

        s2s_st_x += 1; s2s_ut_x += 1; u2u_st_x += 1; u2u_ut_x += 1

        while s2s_st_x != len(set(s2s_st_unq)):
            info_dict['s2s_st'][-1] = [random.choice(spk_dic['s2s_st'][s2s_st_s]), random.choice(spk_dic['s2s_st'][s2s_st_t])]
            s2s_st_unq = [(i[0]['wav_name'], i[1]['wav_name']) for i in info_dict['s2s_st']]
        while s2s_ut_x != len(set(s2s_ut_unq)):
            info_dict['s2s_ut'][-1] = [random.choice(spk_dic['s2s_ut'][s2s_ut_s]), random.choice(spk_dic['s2s_ut'][s2s_ut_t])]
            s2s_ut_unq = [(i[0]['wav_name'], i[1]['wav_name']) for i in info_dict['s2s_ut']]
        while u2u_st_x != len(set(u2u_st_unq)):
            info_dict['u2u_st'][-1] = [random.choice(spk_dic['u2u_st'][u2u_st_s]), random.choice(spk_dic['u2u_st'][u2u_st_t])]
            u2u_st_unq = [(i[0]['wav_name'], i[1]['wav_name']) for i in info_dict['u2u_st']]
        while u2u_ut_x != len(set(u2u_ut_unq)):
            info_dict['u2u_ut'][-1] = [random.choice(spk_dic['u2u_ut'][u2u_ut_s]), random.choice(spk_dic['u2u_ut'][u2u_ut_t])]
            u2u_ut_unq = [(i[0]['wav_name'], i[1]['wav_name']) for i in info_dict['u2u_ut']]
        
    return info_dict

def AddOraclePath(train_info, valid_info, test_info, info_dict):
    
    orc_meta_dict = defaultdict(list)
    for i in train_info:
        spk, wav_path, txt = i['speaker'], i['wav_path'], i['text']
        orc_meta_dict[spk].append([txt, wav_path])
    for i in valid_info:
        spk, wav_path, txt = i['speaker'], i['wav_path'], i['text']
        orc_meta_dict[spk].append([txt, wav_path])
    for i in test_info:
        spk, wav_path, txt = i['speaker'], i['wav_path'], i['text']
        orc_meta_dict[spk].append([txt, wav_path])
        
    for k in info_dict:
        tmp_dict = info_dict[k]
        for idx in range(len(tmp_dict)):

            s_txt, t_spk = tmp_dict[idx][0]['text'], tmp_dict[idx][1]['speaker']
            ref = orc_meta_dict[t_spk]
            info_dict[k][idx].append(None)
            for (ref_txt, ref_path) in ref:
                if s_txt == ref_txt:
                    info_dict[k][idx][-1] = ref_path        
                    
    return info_dict

def main(cfg):
    
    seed_init()
    print('---Read Infos---')
    train_info = Read_json(f'{cfg.output_path}/train.json') # info: (mel_len, speaker, wav_path, mel_path, lf0_path, txt, txt_path, test_type) test_type: train, s2s_st, s2s_ut, u2u_st, u2u_ut
    valid_info = Read_json(f'{cfg.output_path}/valid.json')
    test_info  = Read_json(f'{cfg.output_path}/test.json')
    
    valid_spk_dict = GetSpeakerDict(valid_info) # {test_type:spk:[info (mel_len, speaker, wav_path, mel_path, lf0_path, txt, txt_path, test_type), ...], ...}
    test_spk_dict  = GetSpeakerDict(test_info)
        
    valid_info_dict = GeneratePairSample(valid_spk_dict, cfg.num_samples) # Valid Pair info dict {'s2s_st':[[[source speaker info], [target speaker info]], ..., ], 's2s_ut':...}
    test_info_dict  = GeneratePairSample(test_spk_dict, cfg.num_samples)
    
    valid_info_dict = AddOraclePath(train_info, valid_info, test_info, valid_info_dict) # Add Oracle path if exists
    test_info_dict  = AddOraclePath(train_info, valid_info, test_info, test_info_dict)
    
    print('---Write Infos---')
    Write_json(valid_info_dict, f'{cfg.output_path}/valid_pair.json')
    Write_json(test_info_dict, f'{cfg.output_path}/test_pair.json')
    print('---Done---')
    
if __name__ == '__main__':
    
    cfg = Config('./config/preprocess.yaml')
    main(cfg)