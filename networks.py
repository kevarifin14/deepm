import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DecisionNetwork_CNN(nn.Module):
    """
    An EIIE style decision network implemented with CNN.
    """
    
    def __init__(self):
        super(DecisionNetwork_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, 
                               out_channels=3, 
                               kernel_size=[1,3]) # can also use [1,2]
        self.conv2 = nn.Conv2d(in_channels=3, 
                               out_channels=20, # can also use 10
                               kernel_size=[1, 48])
        self.conv3 = nn.Conv2d(in_channels=21, 
                               out_channels=1, 
                               kernel_size=[1, 1])
        
    def forward(self, obs, prev_pv):
        """
        Compute the forward pass. 
        
        Input:
        - obs: A fresh observation of the market environment at the current time step.
          A tensor of shape [batch_size, feature_size, num_assets, window_size].
        - last_pv: The portfolio weight vector in the previous time step. A tensor of
          shape [batch_size, num_assets].
        
        Returns:
        - new_pv: The new portfolio weight vector for the current time step. A tensor
          of shape [batch_size, num_assets]
        """
        batch_size, num_assets = prev_pv.shape
        
        scores = nn.ReLU()(self.conv1(obs))
        scores = nn.ReLU()(self.conv2(scores))
        scores = torch.cat([scores, prev_pv.view(batch_size, 1, num_assets, 1)], dim=1)
        scores = self.conv3(scores).squeeze()
        
        # This is very hacky, probably need a better way to deal with cash asset
        cash_bias = torch.zeros(batch_size, 1, requires_grad=False) 
        scores = torch.cat([cash_bias, scores], dim=1)
        
        new_pv = F.softmax(scores, dim=1)
        return new_pv


class DecisionNetwork_RNN(nn.Module):
    """
    An EIIE style decision network implemented with vanilla RNN.
    """
    
    def __init__(self):
        super(DecisionNetwork_RNN, self).__init__()
        self.hidden_size = 20
        self.rnn = nn.RNN(input_size=3, 
                          hidden_size=self.hidden_size, 
                          num_layers=1,
                          batch_first=True)
        self.conv = nn.Conv2d(in_channels=21, 
                              out_channels=1, 
                              kernel_size=[1, 1])
    
    def forward(self, obs, prev_pv):
        """
        Compute the forward pass. 
        
        Input:
        - obs: A fresh observation of the market environment at the current time step.
          A tensor of shape [batch_size, feature_size, num_assets, window_size].
        - last_pv: The portfolio weight vector in the previous time step. A tensor of
          shape [batch_size, num_assets].
        
        Returns:
        - new_pv: The new portfolio weight vector for the current time step. A tensor
          of shape [batch_size, num_assets]
        """
        batch_size, num_assets = prev_pv.shape
        
        # pytorch RNN module expect input to be [batch, window, features]
        obs = obs.transpose(2, 3) # [batch, feature, window, assets]
        obs = obs.transpose(1, 2) # [batch, window, feature, assets]
        
        scores_list = []
        for asset in range(num_assets):
            h_0 = torch.zeros(1, batch_size, self.hidden_size) # TODO: try other inits
            output, h_t = self.rnn(obs[:, :, :, asset], h_0)
            scores_list.append(output[:, -1, :])
        scores = torch.stack(scores_list, dim=2)
        scores = scores.view(batch_size, self.hidden_size, num_assets, 1)
        scores = torch.cat([scores, prev_pv.view(batch_size, 1, num_assets, 1)], dim=1)
        scores = self.conv(scores).squeeze()
        
        # This is very hacky, probably need a better way to deal with cash asset
        cash_bias = torch.zeros(batch_size, 1, requires_grad=False) 
        scores = torch.cat([cash_bias, scores], dim=1)
        
        new_pv = F.softmax(scores, dim=1)
        return new_pv


class DecisionNetwork_LSTM(nn.Module):
    """
    An EIIE style decision network implemented with LSTM.
    """
    
    def __init__(self):
        super(DecisionNetwork_LSTM, self).__init__()
        self.hidden_size = 20
        self.lstm = nn.LSTM(input_size=3, 
                            hidden_size=self.hidden_size, 
                            num_layers=1,
                            batch_first=True)
        self.conv = nn.Conv2d(in_channels=21, 
                              out_channels=1, 
                              kernel_size=[1, 1])
    
    def forward(self, obs, prev_pv):
        """
        Compute the forward pass. 
        
        Input:
        - obs: A fresh observation of the market environment at the current time step.
          A tensor of shape [batch_size, feature_size, num_assets, window_size].
        - last_pv: The portfolio weight vector in the previous time step. A tensor of
          shape [batch_size, num_assets].
        
        Returns:
        - new_pv: The new portfolio weight vector for the current time step. A tensor
          of shape [batch_size, num_assets]
        """
        batch_size, num_assets = prev_pv.shape
        
        # pytorch LSTM module expect input to be [batch, window, features]
        obs = obs.transpose(2, 3) # [batch, feature, window, assets]
        obs = obs.transpose(1, 2) # [batch, window, feature, assets]
        
        scores_list = []
        for asset in range(num_assets):
            h_0 = torch.zeros(1, batch_size, self.hidden_size) # TODO: try other inits
            c_0 = torch.zeros(1, batch_size, self.hidden_size) # TODO: try other inits
            output, _ = self.lstm(obs[:, :, :, asset], (h_0, c_0))
            scores_list.append(output[:, -1, :])
        scores = torch.stack(scores_list, dim=2)
        scores = scores.view(batch_size, self.hidden_size, num_assets, 1)
        scores = torch.cat([scores, prev_pv.view(batch_size, 1, num_assets, 1)], dim=1)
        scores = self.conv(scores).squeeze()
        
        # This is very hacky, probably need a better way to deal with cash asset
        cash_bias = torch.zeros(batch_size, 1, requires_grad=False) 
        scores = torch.cat([cash_bias, scores], dim=1)
        
        new_pv = F.softmax(scores, dim=1)
        return new_pv