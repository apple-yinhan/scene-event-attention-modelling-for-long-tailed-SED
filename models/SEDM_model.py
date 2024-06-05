#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 15:22:03 2023

@author: bfzystudent
"""
from models.shared_modules import Encoder_Layers, ASC_CNN, SED_CRNN, SED_Conformer,\
ConvBlock, Multiscale_ConvBlock, SED_Transformer

import torch.nn as nn
import torch

class SEDM_A(nn.Module):
    # SEDM Model
    # sed module: CRNN
    def __init__(self, args):
        super(SEDM_A, self).__init__()
        self.encoder_layers = Encoder_Layers(in_channel=args['channel'], 
                                             cnn_filters=args['encoder_filter'], 
                                             dropout_rate=args['dropout'])
        self.sed_module = SED_CRNN(in_filters=args['encoder_filter'], 
                                   cnn_filters=args['sed_filter'], 
                                   rnn_hid=args['sed_rnn_dim'], 
                                   classes_num=args['sed_class'], 
                                   dropout_rate=args['dropout'])
        self.asc_module = ASC_CNN(in_filters=args['encoder_filter'], 
                                  cnn_filters=args['asc_filter'], 
                                  in_fc=args['asc_fc'], class_num=args['asc_class'], 
                                  dropout_rate=args['dropout'])
        # scene-event dictionary
        self.D_se = nn.Parameter(torch.ones(args['asc_class'], args['sed_class']))
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        #self.relu = nn.ReLU()
        
    def forward(self, x):
        R_se = self.encoder_layers(x)
        # print(R_se.shape) [16, 64, 200, 128]
        E_e = self.sed_module(R_se)      # [16, 200, 25]
        y_s_hat = self.asc_module(R_se)  # [16, 200, 4]
        y_s_hat = self.softmax(y_s_hat)
        M_se = torch.matmul(y_s_hat, self.D_se)
        M_se = self.sigmoid(M_se)
        y_e_hat = (E_e * M_se)
        return y_e_hat, y_s_hat, E_e

class SEDM_B(nn.Module):
    # SEDM Model
    # sed module: Conformer
    def __init__(self, args):
        super(SEDM_B, self).__init__()
        self.encoder_layers = Encoder_Layers(in_channel=args['channel'], 
                                             cnn_filters=args['encoder_filter'], 
                                            dropout_rate=args['dropout'])
        self.sed_module = SED_Conformer(in_filters=args['encoder_filter'], 
                                   cnn_filters=args['sed_filter'], 
                                   rnn_hid=args['sed_rnn_dim'], 
                                   classes_num=args['sed_class'], 
                                   dropout_rate=args['dropout'])
        self.asc_module = ASC_CNN(in_filters=args['encoder_filter'], 
                                  cnn_filters=args['asc_filter'], 
                                  in_fc=args['asc_fc'], class_num=args['asc_class'], 
                                  dropout_rate=args['dropout'])
        # scene-event dictionary
        self.D_se = nn.Parameter(torch.rand(args['asc_class'], args['sed_class']))
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        R_se = self.encoder_layers(x)
        # print(R_se.shape) [16, 64, 200, 128]
        E_e = self.sed_module(R_se)      # [16, 200, 25]
        y_s_hat = self.asc_module(R_se)  # [16, 200, 4]
        # print(y_s_hat[0,0:2,:])
        y_s_hat = self.softmax(y_s_hat)
        # print(y_s_hat[0,0:2,:])
        M_se = torch.matmul(y_s_hat, self.D_se)
        M_se = self.sigmoid(M_se)
        y_e_hat = (E_e * M_se)
        return y_e_hat, y_s_hat, E_e

class SEDM_C(nn.Module):
    # SEDM Model
    # sed module: Transformer
    def __init__(self, args):
        super(SEDM_C, self).__init__()
        self.encoder_layers = Encoder_Layers(in_channel=args['channel'], 
                                             cnn_filters=args['encoder_filter'], 
                                            dropout_rate=args['dropout'])
        self.sed_module = SED_Transformer(in_filters=args['encoder_filter'], 
                                   cnn_filters=args['sed_filter'], 
                                   rnn_hid=args['sed_rnn_dim'], 
                                   classes_num=args['sed_class'], 
                                   dropout_rate=args['dropout'],
                                   head=args['head'],
                                   num_layers=args['num_layers'])
        self.asc_module = ASC_CNN(in_filters=args['encoder_filter'], 
                                  cnn_filters=args['asc_filter'], 
                                  in_fc=args['asc_fc'], class_num=args['asc_class'], 
                                  dropout_rate=args['dropout'])
        # scene-event dictionary
        self.D_se = nn.Parameter(torch.rand(args['asc_class'], args['sed_class']))
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        R_se = self.encoder_layers(x)
        # print(R_se.shape) [16, 64, 200, 128]
        E_e = self.sed_module(R_se)      # [16, 200, 25]
        y_s_hat = self.asc_module(R_se)  # [16, 200, 4]
        M_se = torch.matmul(y_s_hat, self.D_se)
        M_se = self.sigmoid(M_se)
        y_e_hat = E_e * M_se
        return y_e_hat, y_s_hat, E_e
    
if __name__ == '__main__':
    import config
    inputs = torch.rand((2,1,200,128))
    args = config.model_param['SEDM_C']
    model = SEDM_C(args)
    out = model(inputs)
    print(out[0].shape, out[1].shape, out[2].shape)