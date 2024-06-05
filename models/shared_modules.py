#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 20:30:27 2023

@author: bfzystudent
"""

import torch
import torch.nn as nn
from models.conformer.encoder import ConformerBlock

class Encoder_Layers(nn.Module):
    
    def __init__(self, in_channel=1, cnn_filters=64, dropout_rate=0.1):
        super(Encoder_Layers, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=cnn_filters, 
                               kernel_size=(3, 3), padding=(1,1))
        self.batch_norm1 = nn.BatchNorm2d(num_features=cnn_filters)
        
        self.conv2 = nn.Conv2d(in_channels=cnn_filters, out_channels=cnn_filters, 
                               kernel_size=(3, 3), padding=(1,1))
        self.batch_norm2 = nn.BatchNorm2d(num_features=cnn_filters)
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x: [B,C,T,F]
        # print(x.shape) 
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        # print(x.shape) [B, 64, T, F]
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        # print(x.shape) [B, 64, T, F]
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channel, cnn_filters, dropout_rate):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channel, 
                              out_channels=cnn_filters,
                              kernel_size=(3, 3),
                              padding=(1, 1))
                              
        self.conv2 = nn.Conv2d(in_channels=cnn_filters, 
                              out_channels=cnn_filters,
                              kernel_size=(3, 3),
                              padding=(1, 1))
                              
        self.batch_norm1 = nn.BatchNorm2d(cnn_filters)
        self.batch_norm2 = nn.BatchNorm2d(cnn_filters)
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input):
        x = input
        x = self.dropout(torch.relu(self.batch_norm1(self.conv1(x))))
        x = self.dropout(torch.relu(self.batch_norm2(self.conv2(x))))
        return x

class Multiscale_ConvBlock(nn.Module):
    def __init__(self, in_channel, cnn_filters, dropout_rate):
        
        super(Multiscale_ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channel, 
                              out_channels=cnn_filters//2,
                              kernel_size=(3, 3),
                              padding=(1,1))
                              
        
        self.conv2 = nn.Conv2d(in_channels=in_channel, 
                              out_channels=cnn_filters//2,
                              kernel_size=(5, 5),
                              padding=(2, 2))
                              
        
        self.conv3 = nn.Conv2d(in_channels=cnn_filters, 
                              out_channels=cnn_filters,
                              kernel_size=(1, 1))
                              
        
        self.batch_norm1 = nn.BatchNorm2d(cnn_filters//2)
        self.batch_norm2 = nn.BatchNorm2d(cnn_filters//2)
        self.batch_norm3 = nn.BatchNorm2d(cnn_filters)
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input):
        x = input
        x1 = self.dropout(torch.relu(self.batch_norm1(self.conv1(x))))
       
        x2 = self.dropout(torch.relu(self.batch_norm2(self.conv2(x))))
        
        y = torch.cat((x1, x2), dim=1)
        
        y = self.dropout(torch.relu(self.batch_norm3(self.conv3(y))))
        
        return y


class SED_CRNN(nn.Module):
    # CRNN Baseline for DCASE2023 Task4
    def __init__(self, in_filters, cnn_filters, rnn_hid, classes_num, dropout_rate):
        super(SED_CRNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_filters, out_channels=cnn_filters, 
                               kernel_size=(3, 3), padding=(1,1))
        self.batch_norm1 = nn.BatchNorm2d(num_features=cnn_filters)
        
        self.conv2 = nn.Conv2d(in_channels=cnn_filters, out_channels=cnn_filters, 
                               kernel_size=(3, 3), padding=(1,1))
        self.batch_norm2 = nn.BatchNorm2d(num_features=cnn_filters)
        
        self.conv3 = nn.Conv2d(in_channels=cnn_filters, out_channels=cnn_filters, 
                               kernel_size=(3, 3), padding=(1,1))
        self.batch_norm3 = nn.BatchNorm2d(num_features=cnn_filters)
        
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 5))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2))
        
        self.dropout = nn.Dropout(dropout_rate)
        # 3 (64) or 6 (128) 
        self.gru1 = nn.GRU(int(3*cnn_filters), rnn_hid, bidirectional=True, batch_first=True)

        self.linear1 = nn.Linear(rnn_hid*2, rnn_hid)
        
        self.linear2 = nn.Linear(rnn_hid, classes_num)
        self.activate = nn.Sigmoid()
        
    def forward(self, x):
        # x: [B,C,T,F]
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = torch.relu(x)
        x = self.pool3(x)
        x = self.dropout(x)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape((x.shape[0], x.shape[1], -1))
        # print(x.shape)
        # Bidirectional layer
        # print(x.shape)
        recurrent, _ = self.gru1(x)
        # print(recurrent.shape)
        x = self.linear1(recurrent)
        x = self.linear2(x)
        x = self.activate(x)
        return x

class SED_Conformer(nn.Module):
    # Conformer Model for SED
    def __init__(self, in_filters, cnn_filters, rnn_hid, classes_num, dropout_rate):
        super(SED_Conformer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_filters, out_channels=cnn_filters, 
                               kernel_size=(3, 3), padding=(1,1))
        self.batch_norm1 = nn.BatchNorm2d(num_features=cnn_filters)
        
        self.conv2 = nn.Conv2d(in_channels=cnn_filters, out_channels=cnn_filters, 
                               kernel_size=(3, 3), padding=(1,1))
        self.batch_norm2 = nn.BatchNorm2d(num_features=cnn_filters)
        
        self.conv3 = nn.Conv2d(in_channels=cnn_filters, out_channels=cnn_filters, 
                               kernel_size=(3, 3), padding=(1,1))
        self.batch_norm3 = nn.BatchNorm2d(num_features=cnn_filters)
        
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 5))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 3))
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2))
        
        self.dropout = nn.Dropout(dropout_rate)

        self.conformer_block1 = ConformerBlock(512) # 512 256
        self.conformer_block2 = ConformerBlock(512) # 512 256
        
        self.linear1 = nn.Linear(2*rnn_hid, rnn_hid)
        
        self.linear2 = nn.Linear(rnn_hid, classes_num)
        self.activate = nn.Sigmoid()
        
    def forward(self, x):
        # x: [B,C,T,F]
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = torch.relu(x)
        x = self.pool3(x)
        x = self.dropout(x)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape((x.shape[0], x.shape[1], -1))
        
        # print(x.shape)
        
        recurrent = self.conformer_block1(x)
        recurrent = self.conformer_block2(recurrent)
        
        y = self.linear1(recurrent)
        y = self.linear2(y)
        # print(y[0,0,:])
        y = self.activate(y)
        
        return y

class SED_Transformer(nn.Module):
    # Conformer Model for SED
    def __init__(self, in_filters, cnn_filters, rnn_hid, classes_num, dropout_rate,
                 head, num_layers):
        super(SED_Transformer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_filters, out_channels=cnn_filters, 
                               kernel_size=(3, 3), padding=(1,1))
        self.batch_norm1 = nn.BatchNorm2d(num_features=cnn_filters)
        
        self.conv2 = nn.Conv2d(in_channels=cnn_filters, out_channels=cnn_filters, 
                               kernel_size=(3, 3), padding=(1,1))
        self.batch_norm2 = nn.BatchNorm2d(num_features=cnn_filters)
        
        self.conv3 = nn.Conv2d(in_channels=cnn_filters, out_channels=cnn_filters, 
                               kernel_size=(3, 3), padding=(1,1))
        self.batch_norm3 = nn.BatchNorm2d(num_features=cnn_filters)
        
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 5))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 5))
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2))
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=head)
        self.trans_1 = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.trans_2 = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        self.linear1 = nn.Linear(rnn_hid*2, rnn_hid)
        
        self.linear2 = nn.Linear(rnn_hid, classes_num)
        self.activate = nn.Sigmoid()
        # self.relu = nn.ReLU()
        
    def forward(self, x):
        # x: [B,C,T,F]
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = torch.relu(x)
        x = self.pool3(x)
        x = self.dropout(x)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape((x.shape[0], x.shape[1], -1))
        
        # print(x.shape)
        
        recurrent = self.trans_1(x)
        recurrent = self.trans_2(recurrent)
        
        y = self.linear1(recurrent)
        y = self.linear2(y)
        y = self.activate(y)
        
        return y
    
class SED_CRNN_cSEM(nn.Module):
    # CRNN Baseline for DCASE2023 Task4
    def __init__(self, in_filters, cnn_filters, rnn_hid, classes_num, dropout_rate):
        super(SED_CRNN_cSEM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_filters, out_channels=cnn_filters, 
                               kernel_size=(3, 3), padding=(1,1))
        self.batch_norm1 = nn.BatchNorm2d(num_features=cnn_filters)
        
        self.conv2 = nn.Conv2d(in_channels=cnn_filters, out_channels=cnn_filters, 
                               kernel_size=(3, 3), padding=(1,1))
        self.batch_norm2 = nn.BatchNorm2d(num_features=cnn_filters)
        
        self.conv3 = nn.Conv2d(in_channels=cnn_filters, out_channels=cnn_filters, 
                               kernel_size=(3, 3), padding=(1,1))
        self.batch_norm3 = nn.BatchNorm2d(num_features=cnn_filters)
        
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 5))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2))
        
        self.dropout = nn.Dropout(dropout_rate)
        self.gru1 = nn.GRU(int(6*cnn_filters), rnn_hid, bidirectional=True, batch_first=True)

    def forward(self, x):
        # x: [B,C,T,F]
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = torch.relu(x)
        x = self.pool3(x)
        x = self.dropout(x)
        
        x = x.permute(0, 2, 1, 3)
        x = x.reshape((x.shape[0], x.shape[1], -1))
        
        # Bidirectional layer
        recurrent, _ = self.gru1(x)
        return recurrent

class SED_Conformer_cSEM(nn.Module):
    # Conformer model for SED 
    def __init__(self, in_filters, cnn_filters, rnn_hid, classes_num, dropout_rate):
        super(SED_Conformer_cSEM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_filters, out_channels=cnn_filters, 
                               kernel_size=(3, 3), padding=(1,1))
        self.batch_norm1 = nn.BatchNorm2d(num_features=cnn_filters)
        
        self.conv2 = nn.Conv2d(in_channels=cnn_filters, out_channels=cnn_filters, 
                               kernel_size=(3, 3), padding=(1,1))
        self.batch_norm2 = nn.BatchNorm2d(num_features=cnn_filters)
        
        self.conv3 = nn.Conv2d(in_channels=cnn_filters, out_channels=cnn_filters, 
                               kernel_size=(3, 3), padding=(1,1))
        self.batch_norm3 = nn.BatchNorm2d(num_features=cnn_filters)
        
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 5))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 5))
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2))
        
        self.dropout = nn.Dropout(dropout_rate)
        self.conformer_block1 = ConformerBlock(512)
        self.conformer_block2 = ConformerBlock(512)
        
    def forward(self, x):
        # x: [B,C,T,F]
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = torch.relu(x)
        x = self.pool3(x)
        x = self.dropout(x)
        
        x = x.permute(0, 2, 1, 3)
        x = x.reshape((x.shape[0], x.shape[1], -1))
        
        # Conformer layers
        recurrent = self.conformer_block1(x)
        recurrent = self.conformer_block2(recurrent)
        
        return recurrent

class ASC_CNN(nn.Module):
    
    def __init__(self, in_filters, cnn_filters, in_fc, class_num, dropout_rate):
        super(ASC_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_filters, out_channels=cnn_filters, 
                               kernel_size=(3, 3), padding=(1,1))
        self.batch_norm1 = nn.BatchNorm2d(num_features=cnn_filters)
        
        self.conv2 = nn.Conv2d(in_channels=cnn_filters, out_channels=cnn_filters, 
                               kernel_size=(3, 3), padding=(1,1))
        self.batch_norm2 = nn.BatchNorm2d(num_features=cnn_filters)
        
        self.conv3 = nn.Conv2d(in_channels=cnn_filters, out_channels=cnn_filters, 
                               kernel_size=(3, 3), padding=(1,1))
        self.batch_norm3 = nn.BatchNorm2d(num_features=cnn_filters)
        
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 5))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 5))
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2))
        
        self.dropout = nn.Dropout(dropout_rate)
        in_fc = in_fc //2 
        self.fc1 = nn.Linear(in_features=in_fc, out_features=in_fc//2)
        self.fc2 = nn.Linear(in_features=in_fc//2, out_features=class_num)
        # self.act_final = nn.Softmax(dim=-1)
        
        
    def forward(self, x):
        # x: [B,C,T,F]
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = torch.relu(x)
        x = self.pool3(x)
        x = self.dropout(x)
        
        x = x.permute(0, 2, 1, 3)
        x = x.reshape((x.shape[0], x.shape[1], -1))
        x = self.fc2(self.fc1(x))
        # print(x[0,0,:])
        # x = self.act_final(x)
        
        return x
    
class ASC_CNN_cSEM(nn.Module):
    
    def __init__(self, in_filters, cnn_filters, class_num, dropout_rate):
        super(ASC_CNN_cSEM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_filters, out_channels=cnn_filters, 
                               kernel_size=(3, 3), padding=(1,1))
        self.batch_norm1 = nn.BatchNorm2d(num_features=cnn_filters)
        
        self.conv2 = nn.Conv2d(in_channels=cnn_filters, out_channels=cnn_filters, 
                               kernel_size=(3, 3), padding=(1,1))
        self.batch_norm2 = nn.BatchNorm2d(num_features=cnn_filters)
        
        self.conv3 = nn.Conv2d(in_channels=cnn_filters, out_channels=cnn_filters, 
                               kernel_size=(3, 3), padding=(1,1))
        self.batch_norm3 = nn.BatchNorm2d(num_features=cnn_filters)
        
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 5))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 5))
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2))
        
        self.dropout = nn.Dropout(dropout_rate)
        
        
    def forward(self, x):
        # x: [B,C,T,F]
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = torch.relu(x)
        x = self.pool3(x)
        x = self.dropout(x)
        
        x = x.permute(0, 2, 1, 3)
        x = x.reshape((x.shape[0], x.shape[1], -1))
        
        return x