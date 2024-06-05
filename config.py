#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 23:52:13 2023

@author: bfzystudent
"""
# ----------------------------------------- #
SED_label = {
         'TUT16_17': {
             '(object) banging': 0, '(object) impact': 1, '(object) rustling': 2,
             '(object) snapping': 3, '(object) squeaking': 4, 'bird singing': 5,
             'brakes squeaking': 6, 'breathing': 7, 'car': 8, 'children': 9,
             'cupboard': 10, 'cutlery': 11, 'dishes': 12, 'drawer': 13,
             'fan': 14, 'glass jingling': 15, 'keyboard typing': 16,
             'large vehicle': 17, 'mouse clicking': 18, 'mouse wheeling': 19,
             'people talking': 20, 'people walking': 21, 'washing dishes': 22, 
             'water tap running': 23, 'wind blowing': 24, 
         },
         'JSSED': {
             'bagrustle': 0, 'birdsong': 1, 'buspassby': 2, 'chairsmoving': 3,
             'checkoutbeeps': 4, 'clearthroat': 5, 'cooking': 6, 'cough': 7,
             'doorclose': 8, 'doorslam': 9, 'drawer': 10, 'footsteps': 11,
             'footstepsongrass': 12, 'gate': 13, 'keylock': 14, 'keys': 15,
             'knock': 16, 'lake': 17, 'laughter': 18, 'lightrain': 19,
             'mindthegap': 20, 'money': 21, 'motorbike': 22, 'phone': 23,
             'pushbike': 24, 'running': 25, 'slidingDoorclose': 26,
             'speech': 27, 'switch': 28, 'train': 29, 'trolley': 30, 'wind': 31
         }
    }

SED_class_num = {
    'TUT16_17': 25, 'JSSED': 32
    }
SED_labels_name = {
    'TUT16_17': ['(object) banging', '(object) impact', '(object) rustling',
             '(object) snapping', '(object) squeaking', 'bird singing',
             'brakes squeaking', 'breathing', 'car', 'children',
             'cupboard', 'cutlery', 'dishes', 'drawer',
             'fan', 'glass jingling', 'keyboard typing',
             'large vehicle', 'mouse clicking', 'mouse wheeling',
             'people talking', 'people walking', 'washing dishes', 
             'water tap running', 'wind blowing'],
    'JSSED': [
            'bagrustle', 'birdsong', 'buspassby', 'chairsmoving',
            'checkoutbeeps', 'clearthroat', 'cooking', 'cough',
            'doorclose', 'doorslam', 'drawer', 'footsteps',
            'footstepsongrass', 'gate', 'keylock', 'keys',
            'knock', 'lake', 'laughter', 'lightrain',
            'mindthegap', 'money', 'motorbike', 'phone',
            'pushbike', 'running', 'slidingDoorclose',
            'speech', 'switch', 'train', 'trolley', 'wind'
        ]
    }

ASC_label = {
    'TUT16_17': {
        'city_center': 0, 'home': 1, 'office': 2, 'residential_area': 3,
        },
    'JSSED': {
        'bus': 0, 'busystreet': 1, 'office': 2, 'openairmarket': 3,
        'park': 4, 'quietstreet': 5, 'restaurant': 6, 'supermarket': 7,
        'tube': 8, 'tubestation': 9
        }
    }

# ----------------------------------------- #
exp_id = 32
framework = 'MSHE'   # MSHE, cSEM, SEDM
lambdas = [1,1,0.01]
sed_module = 'Conformer'
dataset_source = 'TUT16_17'
out_folder = f'/home/bfzystudent/Personal/YH/SEDM_codes/experiments/{dataset_source}'
loss_type = 'FocalLoss'
val_loss_type = 'BCE'
load_model = True
use_cuda = True
finetune = False
mixup = False
seq_len = 200
seq_hop = 20
lr = 8e-5
lr_decay = 0.9
lr_patience = 10
gpu_id = 0
train_patience = 10
train_batch = 38
val_batch = 38
accumulation = 1
n_mel = 64
n_sed = SED_class_num[dataset_source]

train_data_file = f'/home/bfzystudent/Personal/YH/SEDM_codes/dataset_utils/{dataset_source}/datasets/mel{n_mel}_len{seq_len}_hop{seq_hop}/train'
val_data_file = f'/home/bfzystudent/Personal/YH/SEDM_codes/dataset_utils/{dataset_source}/datasets/mel{n_mel}_len{seq_len}_hop{seq_hop}/eval'

    
# ------------------- model ------------------ #
asc_class_num = 4
model_param = {
    'MSHE_A':{
        'channel': 1, 'encoder_filter': 64, 'sed_filter': 256, 'sed_rnn_dim': 256, 
        'sed_class': SED_class_num[dataset_source], 'asc_filter': 256, 'asc_fc': 512, 
        'asc_class': asc_class_num, 'dropout': 0.1       
        },
    'MSHE_B':{
        'channel': 1, 'encoder_filter': 64, 'sed_filter': 256, 'sed_rnn_dim': 256, 
        'sed_class': SED_class_num[dataset_source], 'asc_filter': 256, 'asc_fc': 512, 
        'asc_class': asc_class_num, 'dropout': 0.1       
        },
    'cSEM_A':{
        'channel': 1, 'encoder_filter': 64, 'sed_filters': 256, 
        'sed_rnn_dim': 256, 'sed_class': SED_class_num[dataset_source], 
        'asc_filters': 256, 'asc_fc_dim': 512, 'asc_class': asc_class_num, 'dropout': 0.1
        },
    'cSEM_B':{
        'channel': 1, 'encoder_filter': 64, 'sed_filters': 256, 
        'sed_rnn_dim': 256, 'sed_class': SED_class_num[dataset_source], 
        'asc_filters': 256, 'asc_fc_dim': 512, 'asc_class': asc_class_num, 'dropout': 0.1
        },
    'SEDM_A':{
        'channel': 1, 'encoder_filter': 64, 'sed_filter': 256, 'sed_rnn_dim': 256, 
        'sed_class': SED_class_num[dataset_source], 'asc_filter': 256, 'asc_fc': 512, 
        'asc_class': asc_class_num, 'dropout': 0.1, 'seq_len': 200
        },
    'SEDM_B':{
        'channel': 1, 'encoder_filter': 64, 'sed_filter': 256, 'sed_rnn_dim': 256, 
        'sed_class': SED_class_num[dataset_source], 'asc_filter': 256, 'asc_fc': 512, 
        'asc_class': asc_class_num, 'dropout': 0.1, 'seq_len': 200
        },
    'SEDM_C':{
        'channel': 1, 'encoder_filter': 64, 'sed_filter': 256, 'sed_rnn_dim': 256, 
        'sed_class': SED_class_num[dataset_source], 'asc_filter': 256, 'asc_fc': 512, 
        'asc_class': asc_class_num, 'dropout': 0.1, 'seq_len': 200, 'head': 8, 
        'num_layers': 6
        }
    
    }