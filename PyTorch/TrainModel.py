# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 18:29:13 2018

@author: Zhiyong
"""

import torch.utils.data as utils
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math
import numpy as np
import pandas as pd
import time

from Models import *


def TrainLSTM(train_dataloader, valid_dataloader, num_epochs = 3):
    
    inputs, labels = next(iter(train_dataloader))
    [batch_size, step_size, fea_size] = inputs.size()
    input_dim = fea_size
    hidden_dim = fea_size
    output_dim = fea_size
    
    lstm = LSTM(input_dim, hidden_dim, output_dim)
    
    lstm.cuda()
    
    loss_MSE = torch.nn.MSELoss()
    loss_L1 = torch.nn.L1Loss()
    
    learning_rate = 1e-5
    optimizer = torch.optim.RMSprop(lstm.parameters(), lr = learning_rate)
    
    use_gpu = torch.cuda.is_available()
    
    interval = 100
    losses_train = []
    losses_interval_train = []
    losses_valid = []
    losses_interval_valid = []
    
    cur_time = time.time()
    pre_time = time.time()
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        trained_number = 0
        
        valid_dataloader_iter = iter(valid_dataloader)

        for data in train_dataloader:
            inputs, labels = data

            if inputs.shape[0] != batch_size:
                continue

            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else: 
                inputs, labels = Variable(inputs), Variable(labels)
                
            lstm.zero_grad()

            outputs = lstm(inputs)
            
            full_labels = torch.cat((inputs[:,1:,:], labels), dim = 1)

            loss_train = loss_MSE(outputs, full_labels)
        
            losses_train.append(loss_train.data)
            
            optimizer.zero_grad()
            
            loss_train.backward()
            
            optimizer.step()
            
             # validation 
            try: 
                inputs_val, labels_val = next(valid_dataloader_iter)
            except StopIteration:
                valid_dataloader_iter = iter(valid_dataloader)
                inputs_val, labels_val = next(valid_dataloader_iter)
            
            if use_gpu:
                inputs_val, labels_val = Variable(inputs_val.cuda()), Variable(labels_val.cuda())
            else: 
                inputs_val, labels_val = Variable(inputs_val), Variable(labels_val)

            full_labels_val = torch.cat((inputs_val[:,1:,:], labels_val), dim = 1)
            
            outputs_val = lstm(inputs_val)
            
            loss_valid = loss_MSE(outputs_val, full_labels_val)
    
            losses_valid.append(loss_valid.data)
            
            # output
            trained_number += 1
            
            if trained_number % interval == 0:
                cur_time = time.time()
                loss_interval_train = np.around(sum(losses_train[-interval:]).cpu().numpy()[0]/interval, decimals=8)
                losses_interval_train.append(loss_interval_train)
                loss_interval_valid = np.around(sum(losses_valid[-interval:]).cpu().numpy()[0]/interval, decimals=8)
                losses_interval_valid.append(loss_interval_valid)
                print('Iteration #: {}, train_loss: {}, valid_loss: {}, time: {}'.format(\
                                                                                         trained_number * batch_size, \
                                                                                         loss_interval_train,\
                                                                                         loss_interval_valid,\
                                                                                         np.around([cur_time - pre_time], decimals=8) ) )
                pre_time = cur_time

    return lstm, [losses_train, losses_interval_train, losses_valid, losses_interval_valid]

def Train_BiLSTM(train_dataloader, valid_dataloader, num_epochs = 3):
    
    inputs, labels = next(iter(train_dataloader))
    [batch_size, step_size, fea_size] = inputs.size()
    input_dim = fea_size
    hidden_dim = fea_size
    output_dim = fea_size
    
    bilstm = BiLSTM(input_dim, hidden_dim, output_dim)
    
    bilstm.cuda()
    
    loss_MSE = torch.nn.MSELoss()
    loss_L1 = torch.nn.L1Loss()
    
    learning_rate = 1e-5
    optimizer = torch.optim.RMSprop(bilstm.parameters(), lr = learning_rate)
    use_gpu = torch.cuda.is_available()
    
    interval = 100
    losses_train = []
    losses_interval_train = []
    losses_valid = []
    losses_interval_valid = []
    
    cur_time = time.time()
    pre_time = time.time()
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        trained_number = 0
        
        valid_dataloader_iter = iter(valid_dataloader)

        for data in train_dataloader:
            inputs, labels = data

            if inputs.shape[0] != batch_size:
                continue

            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else: 
                inputs, labels = Variable(inputs), Variable(labels)
            
            bilstm.zero_grad()

            outputs = bilstm(inputs)
            
            full_labels = torch.cat((inputs[:,1:,:], labels), dim = 1)

            loss_train = loss_MSE(outputs, full_labels)
        
            losses_train.append(loss_train.data)
            
            optimizer.zero_grad()
            
            loss_train.backward()
            
            optimizer.step()
            
             # validation 
            try: 
                inputs_val, labels_val = next(valid_dataloader_iter)
            except StopIteration:
                valid_dataloader_iter = iter(valid_dataloader)
                inputs_val, labels_val = next(valid_dataloader_iter)
            
            if use_gpu:
                inputs_val, labels_val = Variable(inputs_val.cuda()), Variable(labels_val.cuda())
            else: 
                inputs_val, labels_val = Variable(inputs_val), Variable(labels_val)

                
            bilstm.zero_grad()

            full_labels_val = torch.cat((inputs_val[:,1:,:], labels_val), dim = 1)
            
            outputs_val = bilstm(inputs_val)
            
#             Hidden_State, Cell_State = bilstm.loop(inputs_val)

            loss_valid = loss_MSE(outputs_val, full_labels_val)
#             loss_valid = loss_MSE(Hidden_State, labels_val)
    
            losses_valid.append(loss_valid.data)
            
            # output
            trained_number += 1
            
            if trained_number % interval == 0:
                cur_time = time.time()
                loss_interval_train = np.around(sum(losses_train[-interval:]).cpu().numpy()[0]/interval, decimals=8)
                losses_interval_train.append(loss_interval_train)
                loss_interval_valid = np.around(sum(losses_valid[-interval:]).cpu().numpy()[0]/interval, decimals=8)
                losses_interval_valid.append(loss_interval_valid)
                print('Iteration #: {}, train_loss: {}, valid_loss: {}, time: {}'.format(\
                                                                                         trained_number * batch_size, \
                                                                                         loss_interval_train,\
                                                                                         loss_interval_valid,\
                                                                                         np.around([cur_time - pre_time], decimals=8) ) )
                pre_time = cur_time

    return bilstm, [losses_train, losses_interval_train, losses_valid, losses_interval_valid]

def Train_Multi_Bi_LSTM(train_dataloader, valid_dataloader, num_epochs = 3):
    
    inputs, labels = next(iter(train_dataloader))
    [batch_size, step_size, fea_size] = inputs.size()
    input_dim = fea_size
    hidden_dim = fea_size
    output_dim = fea_size
    
#     multiBiLSTM = Multi_Bi_LSTM(input_dim, hidden_dim, output_dim)

    multiBiLSTM = nn.Sequential(BiLSTM(input_dim, hidden_dim, output_dim), LSTM(input_dim, hidden_dim, output_dim))
    
    multiBiLSTM.cuda()
    
    loss_MSE = torch.nn.MSELoss()
    loss_L1 = torch.nn.L1Loss()
    
    learning_rate = 1e-5
    optimizer = torch.optim.RMSprop(multiBiLSTM.parameters(), lr = learning_rate)
    use_gpu = torch.cuda.is_available()
    
    interval = 100
    losses_train = []
    losses_interval_train = []
    losses_valid = []
    losses_interval_valid = []
    
    cur_time = time.time()
    pre_time = time.time()
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        trained_number = 0
        
        valid_dataloader_iter = iter(valid_dataloader)

        for data in train_dataloader:
            inputs, labels = data

            if inputs.shape[0] != batch_size:
                continue

            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else: 
                inputs, labels = Variable(inputs), Variable(labels)
            
            multiBiLSTM.zero_grad()

            outputs = multiBiLSTM(inputs)
            
            full_labels = torch.cat((inputs[:,1:,:], labels), dim = 1)

            loss_train = loss_MSE(outputs, full_labels)
        
            losses_train.append(loss_train.data)
            
            optimizer.zero_grad()
            
            loss_train.backward()
            
            optimizer.step()
            
             # validation 
            try: 
                inputs_val, labels_val = next(valid_dataloader_iter)
            except StopIteration:
                valid_dataloader_iter = iter(valid_dataloader)
                inputs_val, labels_val = next(valid_dataloader_iter)
            
            if use_gpu:
                inputs_val, labels_val = Variable(inputs_val.cuda()), Variable(labels_val.cuda())
            else: 
                inputs_val, labels_val = Variable(inputs_val), Variable(labels_val)

                
            multiBiLSTM.zero_grad()

            full_labels_val = torch.cat((inputs_val[:,1:,:], labels_val), dim = 1)
            
            outputs_val = multiBiLSTM(inputs_val)
            
#             Hidden_State, Cell_State = bilstm.loop(inputs_val)

            loss_valid = loss_MSE(outputs_val, full_labels_val)
#             loss_valid = loss_MSE(Hidden_State, labels_val)
    
            losses_valid.append(loss_valid.data)
            
            # output
            trained_number += 1
            
            if trained_number % interval == 0:
                cur_time = time.time()
                loss_interval_train = np.around(sum(losses_train[-interval:]).cpu().numpy()[0]/interval, decimals=8)
                losses_interval_train.append(loss_interval_train)
                loss_interval_valid = np.around(sum(losses_valid[-interval:]).cpu().numpy()[0]/interval, decimals=8)
                losses_interval_valid.append(loss_interval_valid)
                print('Iteration #: {}, train_loss: {}, valid_loss: {}, time: {}'.format(\
                                                                                         trained_number * batch_size, \
                                                                                         loss_interval_train,\
                                                                                         loss_interval_valid,\
                                                                                         np.around([cur_time - pre_time], decimals=8) ) )
                pre_time = cur_time

    return multiBiLSTM, [losses_train, losses_interval_train, losses_valid, losses_interval_valid]
