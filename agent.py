import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import *
from markethistory import *


class Agent:
    def __init__(self, policy, config, data=None):
        self.config = config['agent']
        self.period = self.config['period']
        self.window = self.config['window']
        self.batch_size = self.config['batch_size']
        self.train_iterations = self.config['train_iterations']
        self.txn_fee = self.config['txn_fee']
        self.sampling_bias = self.config['sampling_bias']
        self.num_assets = self.config['num_assets']
        self.lr = self.config['lr']
        
        if data is None:
            self.data_global = self.get_data(config)
        else: 
            self.data_global = data
        self.data_train, self.data_valid, self.data_test = self.split_data()
        self.policy = policy

    def get_data(self, config):
        data_global = MarketHistory(config).data
        num_feature, num_asset, T = data_global.shape
        btc_price_tensor = np.ones((num_feature, 1, T))
        data_global = np.concatenate((btc_price_tensor, data_global), axis=1)
        print("Global data tensor shape:", data_global.shape)
        return data_global

    def split_data(self):
        num_feature, num_asset, T = self.data_global.shape
        T_test = int(0.1 * T)
        T_valid = int(0.2 * T)
        T_train = T - T_test - T_valid

        data_global = torch.from_numpy(self.data_global)
        return data_global[:, :, :T_train], data_global[:, :, T_train:T_train+T_valid], data_global[:, :, T_train+T_valid:]

    def sample(self, start, end, bias):
        """
        Geometrically sample a number in [START, END)
        
        Input:
        - start: the start (inclusive)
        - end: the end (exclusive)
        - bias: a number between 0 to 1. The closer the bias to 1, the more
        likely to generate a sample closer to END.
        """
        offset = np.random.geometric(bias)
        return max(end - offset, start)
    
    def sample_batch(self, batch_size, start, end, bias):
        """
        Sample a batch of numbers geometrically distributed in [START, END)
        """
        return torch.tensor([self.sample(start, end, bias) for _ in range(batch_size)])

    def get_observation(self, end_t_batch, history):
        """
        Get a batch of price history of length OBS_WINDOW, ending at END_T_BATCH (inclusive).
        
        Input:
        - end_t_batch: The end time indices of this observation. Shape: [BATCH_SIZE].
        - history: The price history tensor of shape [NUM_FEATURE, NUM_ASSET, T]
        
        Returns:
        - obs: A torch tensor of shape [BATCH_SIZE, NUM_FEATURE, NUM_ASSET, OBS_WINDOW]
        """
        obs = []
        for offset in range(self.window - 1, -1, -1):
            t_batch = end_t_batch - offset
            observation = history[:3, :, t_batch].permute(2, 0, 1)
            obs.append(observation)
        obs_prices = torch.stack(obs, dim=-1)
        
        # normalize each asset's prices by its lastest closing prices
        last_close_prices = obs_prices[:, 0, :, -1]
        tmp = obs_prices.permute(1, 3, 0, 2) / last_close_prices
        obs_prices = tmp.permute(2, 0, 3, 1)

        obs = []
        for offset in range(self.window - 1, -1, -1):
            t_batch = end_t_batch - offset
            observation = history[3:, :, t_batch].permute(2, 0, 1)
            obs.append(observation)
        obs_features = torch.stack(obs, dim=-1)
        
        obs = torch.cat((obs_prices, obs_features), 1)
        obs = obs.type(torch.float32)
        return obs

    def calculate_shrinkage(self, w, w_prev):
        """
        Calculate the porfolio value shrinkage during a portfolio weight re-allocation due
        to transaction fees.
        This function calculates the shrinkage using an iterative approximation method. See
        equation (14) of the Deep Portfolio Management paper. 
        
        Input:
        - w: Target portfolio weight tensor of shape [BATCH_SIZE, NUM_ASSET]
        - w_prev: Previous portfolio weight tensor of shape [BATCH_SIZE, NUM_ASSET]
        
        Returns:
        - shrinkage: Portfolio value shrinkage multipler tensor of shape [BATCH_SIZE]
        """
        w0_0, w0_m = w_prev[:, 0], w_prev[:, 1:]
        w1_0, w1_m = w[:, 0], w[:, 1:]
        
        const1 = 1 - self.txn_fee * w0_0
        const2 = 2 * self.txn_fee - self.txn_fee ** 2
        const3 = 1 - self.txn_fee * w1_0
        
        u = self.txn_fee * torch.sum(torch.abs(w0_m - w1_m))
        w1_m_T = w1_m.transpose(0, 1)
        while True:
            u_next = (const1 - const2 * torch.sum(F.relu(w0_m - (u*w1_m_T).transpose(0,1)), dim=1)) / const3
            max_diff = torch.max(torch.abs(u - u_next))
            if max_diff <= 1e-10:
                return u_next
            u = u_next

    def train(self, episodes=10000):
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        T = self.data_train.shape[-1]
        
        for i in range(episodes):
            # geometrically sample start times: [batch]
            start_indices = self.sample_batch(self.batch_size, self.window, T-self.window, self.sampling_bias)
            # initialize portfolio weights: [batch, asset]
            pf_w = (torch.ones(self.num_assets) / self.num_assets).repeat(self.batch_size, 1)
            # initialize portfolio values: [batch]
            pf_v = torch.ones(self.batch_size)
            
            # simulate one episode of live trading with the policy
            loss = 0
            price_curr = self.data_train[0, :, start_indices].transpose(0, 1).type(torch.float32) # [batch, asset]
            for t in range(0, self.window):
                price_next = self.data_train[0, :, start_indices+t+1].transpose(0, 1).type(torch.float32) # [batch, asset]
                obs = self.get_observation(start_indices+t, self.data_train)
                
                pf_w_t_start = self.policy.forward(obs, pf_w)
                shrinkage = self.calculate_shrinkage(pf_w_t_start, pf_w)
                pf_v_t_start = (pf_v * shrinkage).type(torch.float32)
                
                w_tmp = (price_next / price_curr) * pf_w_t_start # [batch, asset]
                w_tmp_sum = torch.sum(w_tmp, dim=1) # [batch]
                pf_v_t_end = w_tmp_sum * pf_v_t_start
                pf_w_t_end = w_tmp / w_tmp_sum.view(self.batch_size, 1)
                
                batch_reward = torch.log(pf_v_t_end / pf_v)
                loss -= torch.sum(batch_reward) / self.batch_size
                
                # update variables
                pf_w = pf_w_t_end
                pf_v = pf_v_t_end
                price_curr = price_next
            loss /= self.window
            
            #if i %  == 0:
            print("episode", i, " loss:", float(loss))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()