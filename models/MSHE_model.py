#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 21:03:45 2023

@author: bfzystudent
"""

from models.shared_modules import Encoder_Layers, ASC_CNN, SED_CRNN, SED_Conformer
import torch.nn as nn
import torch


class MSHE_A(nn.Module):
    # MSHE model for SED and ASC (SED module: CRNN)
    def __init__(self, args):
        super(MSHE_A, self).__init__()
        
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
        
    def forward(self, x):
        # x: [B, C, T, F]
        x = self.encoder_layers(x)
        #print(x.shape) 
        sed_out = self.sed_module(x)
        #print(sed_out.shape) [B, T, sed_class]
        asc_out = self.asc_module(x)
        #print(asc_out.shape) [B, T, asc_class]
        return sed_out, asc_out
    
class MSHE_B(nn.Module):
    # MSHE model for SED and ASC (SED module: Conformer)
    def __init__(self, args):
        super(MSHE_B, self).__init__()
        
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
        
    def forward(self, x):
        # x: [B, C, T, F]
        x = self.encoder_layers(x)
        #print(x.shape) 
        sed_out = self.sed_module(x)
        #print(sed_out.shape) [B, T, sed_class]
        asc_out = self.asc_module(x)
        #print(asc_out.shape) [B, T, asc_class]
        return sed_out, asc_out
    
if __name__ == '__main__':
    import config
    inputs = torch.rand((16,1,100,128))
    
    model = MSHE_B(config.model_param['MSHE_B'])
    out = model(inputs)
    print(f'MTL out1:{out[0].shape}, out2:{out[1].shape}')
    
    model = SED_Conformer(in_filters=1, cnn_filters=256, rnn_hid=256, 
                               classes_num=32, dropout_rate=0.1)
    out = model(inputs)
    print('Conformer out shape:', out.shape)



