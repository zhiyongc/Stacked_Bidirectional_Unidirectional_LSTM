# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 18:00:24 2018

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

class LSTM(nn.Module):
    def __init__(self, input_size, cell_size, hidden_size):
        """
        cell_size is the size of cell_state.
        hidden_size is the size of hidden_state, or say the output_state of each step
        """
        super(LSTM, self).__init__()
        
        self.cell_size = cell_size
        self.hidden_size = hidden_size
        self.fl = nn.Linear(input_size + hidden_size, hidden_size)
        self.il = nn.Linear(input_size + hidden_size, hidden_size)
        self.ol = nn.Linear(input_size + hidden_size, hidden_size)
        self.Cl = nn.Linear(input_size + hidden_size, hidden_size)
        
    def step(self, input, Hidden_State, Cell_State):
        combined = torch.cat((input, Hidden_State), 1)
        f = F.sigmoid(self.fl(combined))
        i = F.sigmoid(self.il(combined))
        o = F.sigmoid(self.ol(combined))
        C = F.tanh(self.Cl(combined))
        Cell_State = f * Cell_State + i * C
        Hidden_State = o * F.tanh(Cell_State)
        
        return Hidden_State, Cell_State
    
    def forward(self, inputs):
        batch_size = inputs.size(0)
        time_step = inputs.size(1)
        Hidden_State, Cell_State = self.initHidden(batch_size)
        outputs = None
        for i in range(time_step):
            Hidden_State, Cell_State = self.step(torch.squeeze(inputs[:,i:i+1,:]), Hidden_State, Cell_State)  
            if outputs is None:
                outputs = Hidden_State.unsqueeze(1)
            else:
                outputs = torch.cat((Hidden_State.unsqueeze(1), outputs), 1)
        return outputs
    
    def initHidden(self, batch_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            return Hidden_State, Cell_State
        else:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size))
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size))
            return Hidden_State, Cell_State
        
class BiLSTM(nn.Module):
    
    def __init__(self, input_size, cell_size, hidden_size):
        """
        cell_size is the size of cell_state.
        hidden_size is the size of hidden_state, or say the output_state of each step
        """
        super(BiLSTM, self).__init__()
        
        self.cell_size = cell_size
        self.hidden_size = hidden_size
        self.fl_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.il_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.ol_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.Cl_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.fl_b = nn.Linear(input_size + hidden_size, hidden_size)
        self.il_b = nn.Linear(input_size + hidden_size, hidden_size)
        self.ol_b = nn.Linear(input_size + hidden_size, hidden_size)
        self.Cl_b = nn.Linear(input_size + hidden_size, hidden_size)
        
        
    
    def step(self, input_f, input_b, Hidden_State_f, Cell_State_f, Hidden_State_b, Cell_State_b):
        batch_size = input_f.size(0)
        
        combined_f = torch.cat((input_f, Hidden_State_f), 1)
        
        f_f = F.sigmoid(self.fl_f(combined_f))
        i_f = F.sigmoid(self.il_f(combined_f))
        o_f = F.sigmoid(self.ol_f(combined_f))
        C_f = F.tanh(self.Cl_f(combined_f))
        Cell_State_f = f_f * Cell_State_f + i_f * C_f
        Hidden_State_f = o_f * F.tanh(Cell_State_f)
        
        combined_b = torch.cat((input_b, Hidden_State_b), 1)

        f_b = F.sigmoid(self.fl_b(combined_b))
        i_b = F.sigmoid(self.il_b(combined_b))
        o_b = F.sigmoid(self.ol_b(combined_b))
        C_b = F.tanh(self.Cl_b(combined_b))
        Cell_State_b = f_b * Cell_State_b + i_b * C_b
        Hidden_State_b = o_b * F.tanh(Cell_State_b)
        
        return Hidden_State_f, Cell_State_f, Hidden_State_b, Cell_State_b
    
    def forward(self, inputs):  
        outputs_f = None
        outputs_b = None
        
        batch_size = inputs.size(0)
        steps = inputs.size(1)
        
        Hidden_State_f, Cell_State_f, Hidden_State_b, Cell_State_b = self.initHidden(batch_size)
        
        for i in range(steps):
            Hidden_State_f, Cell_State_f, Hidden_State_b, Cell_State_b = \
                self.step(torch.squeeze(inputs[:,i:i+1,:]), torch.squeeze(inputs[:,steps-i-1:steps-i,:])\
                          , Hidden_State_f, Cell_State_f, Hidden_State_b, Cell_State_b)  
            
            if outputs_f is None:
                outputs_f = Hidden_State_f.unsqueeze(1)
            else:
                outputs_f = torch.cat((outputs_f, Hidden_State_f.unsqueeze(1)), 1)
            if outputs_b is None:
                outputs_b = Hidden_State_b.unsqueeze(1)
            else:
                outputs_b = torch.cat((Hidden_State_b.unsqueeze(1), outputs_b), 1)
        outputs = (outputs_f + outputs_b) / 2
        return outputs
        
    
    def initHidden(self, batch_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State_f = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            Cell_State_f = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            Hidden_State_b = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            Cell_State_b = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            return Hidden_State_f, Cell_State_f, Hidden_State_b, Cell_State_b
        else:
            Hidden_State_f = Variable(torch.zeros(batch_size, self.hidden_size))
            Cell_State_f = Variable(torch.zeros(batch_size, self.hidden_size))
            Hidden_State_b = Variable(torch.zeros(batch_size, self.hidden_size))
            Cell_State_b = Variable(torch.zeros(batch_size, self.hidden_size))
            return Hidden_State_f, Cell_State_f, Hidden_State_b, Cell_State_b
   