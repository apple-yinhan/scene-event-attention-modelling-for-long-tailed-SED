#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 23:19:48 2023

@author: bfzystudent
"""

from models.shared_modules import Encoder_Layers, ASC_CNN_cSEM, SED_CRNN_cSEM,\
                                  SED_Conformer_cSEM
import torch.nn as nn
import torch


class cSEM_A(nn.Module):
    # cSEM Model of paper: cooperative scene-event modeling for ASC 
    # sed module: CRNN
    def __init__(self, args):
        super(cSEM_A, self).__init__()
        self.shared_layers = Encoder_Layers(in_channel=args['channel'], 
                                            cnn_filters=args['encoder_filter'], 
                                             dropout_rate=args['dropout'])
        self.sed_embedding = SED_CRNN_cSEM(in_filters=args['encoder_filter'], 
                                           cnn_filters=args['sed_filters'], 
                                           rnn_hid=args['sed_rnn_dim'], 
                                           classes_num=args['sed_class'], 
                                           dropout_rate=args['dropout'])
        
        self.sed_linear1 = nn.Linear(args['sed_rnn_dim']*2, args['sed_rnn_dim'])
        self.sed_linear2 = nn.Linear(args['sed_rnn_dim'], args['sed_class'])
        self.sed_activate = nn.Sigmoid()
        
        self.asc_embedding = ASC_CNN_cSEM(in_filters=args['encoder_filter'], 
                                          cnn_filters=args['asc_filters'], 
                                          class_num=args['asc_class'], 
                                          dropout_rate=args['dropout'])
        self.asc_fc1 = nn.Linear(in_features=args['asc_fc_dim'], 
                                 out_features=args['asc_fc_dim']//2)
        self.asc_fc2 = nn.Linear(in_features=args['asc_fc_dim']//2, 
                                 out_features=args['asc_class'])
        self.act_soft = nn.Softmax(dim=-1)
        
        
    def forward(self, x):
        # x: [B, C, T, F]
        R_se = self.shared_layers(x)
        # print(R_se.shape)  [B, C, T, F]
        E_e = self.sed_embedding(R_se)
        # print(E_e.shape)   [B, T, 2*sed_rnn_dim]
        y_hat_e = self.sed_activate(self.sed_linear2(self.sed_linear1(E_e)))
        # shape: [B, T, sed_class]
        W_e = self.sed_linear2.weight @ self.sed_linear1.weight
        # print(W_e.shape)  [sed_class, 2*sed_rnn_dim]
        
        E_s = self.asc_embedding(R_se)
        #print(E_s.shape) # [B, T, asc_fc_dim]
        y_hat_s = self.act_soft(self.asc_fc2(self.asc_fc1(E_s)))
        # print(y_hat_s.shape) [B, T, asc_calss]
        W_s = self.asc_fc2.weight @ self.asc_fc1.weight
        # print(W_s.shape) [asc_class, asc_fc_dim]
        
        W_se = W_s @ W_e.T
        # print(W_se.shape) [asc_class, sed_class]
        A_e = self.act_soft(W_se)
        K_e2s = A_e @ W_e # [asc_class, 2*sed_rnn_dim]
        y_inf_s = self.act_soft(E_e @ K_e2s.T)
        # print(y_inf_s.shape) [B, T, asc_class]
        
        A_s = self.act_soft(W_se.T) # [sed_class, asc_class]
        K_s2e = A_s @ W_s # [sed_class, asc_fc_dim]
        y_inf_e = self.sed_activate(E_s @ K_s2e.T) # [B, T, sed_class]
        
        return y_hat_e, y_hat_s, y_inf_e, y_inf_s
    
class cSEM_B(nn.Module):
    # cSEM Model of paper: cooperative scene-event modeling for ASC 
    # sed module: Conformer
    def __init__(self, args):
        super(cSEM_B, self).__init__()
        self.shared_layers = Encoder_Layers(in_channel=args['channel'], 
                                            cnn_filters=args['encoder_filter'], 
                                             dropout_rate=args['dropout'])
        self.sed_embedding = SED_Conformer_cSEM(in_filters=args['encoder_filter'], 
                                           cnn_filters=args['sed_filters'], 
                                           rnn_hid=args['sed_rnn_dim'], 
                                           classes_num=args['sed_class'], 
                                           dropout_rate=args['dropout'])
        
        self.sed_linear1 = nn.Linear(args['sed_rnn_dim']*2, args['sed_rnn_dim'])
        self.sed_linear2 = nn.Linear(args['sed_rnn_dim'], args['sed_class'])
        self.sed_activate = nn.Sigmoid()
        
        self.asc_embedding = ASC_CNN_cSEM(in_filters=args['encoder_filter'], 
                                          cnn_filters=args['asc_filters'], 
                                          class_num=args['asc_class'], 
                                          dropout_rate=args['dropout'])
        self.asc_fc1 = nn.Linear(in_features=args['asc_fc_dim'], 
                                 out_features=args['asc_fc_dim']//2)
        self.asc_fc2 = nn.Linear(in_features=args['asc_fc_dim']//2, 
                                 out_features=args['asc_class'])
        self.act_soft = nn.Softmax(dim=-1)
        
        
    def forward(self, x):
        # x: [B, C, T, F]
        R_se = self.shared_layers(x)
        # print(R_se.shape)  [B, C, T, F]
        E_e = self.sed_embedding(R_se)
        # print(E_e.shape)   [B, T, 2*sed_rnn_dim]
        y_hat_e = self.sed_activate(self.sed_linear2(self.sed_linear1(E_e)))
        # shape: [B, T, sed_class]
        W_e = self.sed_linear2.weight @ self.sed_linear1.weight
        # print(W_e.shape)  [sed_class, 2*sed_rnn_dim]
        
        E_s = self.asc_embedding(R_se)
        #print(E_s.shape) # [B, T, asc_fc_dim]
        y_hat_s = self.act_soft(self.asc_fc2(self.asc_fc1(E_s)))
        # print(y_hat_s.shape) [B, T, asc_calss]
        W_s = self.asc_fc2.weight @ self.asc_fc1.weight
        # print(W_s.shape) [asc_class, asc_fc_dim]
        
        W_se = W_s @ W_e.T
        # print(W_se.shape) [asc_class, sed_class]
        A_e = self.act_soft(W_se)
        K_e2s = A_e @ W_e # [asc_class, 2*sed_rnn_dim]
        y_inf_s = self.act_soft(E_e @ K_e2s.T)
        # print(y_inf_s.shape) [B, T, asc_class]
        
        A_s = self.act_soft(W_se.T) # [sed_class, asc_class]
        K_s2e = A_s @ W_s # [sed_class, asc_fc_dim]
        y_inf_e = self.sed_activate(E_s @ K_s2e.T) # [B, T, sed_class]
        
        return y_hat_e, y_hat_s, y_inf_e, y_inf_s

if __name__ == '__main__':
    import config as args
    inputs = torch.rand((16,1,100,128))
    model = cSEM_B(args.model_param['cSEM_B'])
    out = model(inputs)
    print('outputs shape: ',out[0].shape, out[1].shape,
          out[2].shape, out[3].shape)