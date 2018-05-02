import json
import numpy as np

from argparse import ArgumentParser
from markethistory import MarketHistory
from backtest import Backtest
from constants import * 
from agent import Agent
from networks import *

def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--mode",dest="mode",
                        help="start mode, train, generate, load_data"
                             " backtest",
                        metavar="MODE", default="train")
    return parser

def get_data(config):
    data_global = MarketHistory(config).data
    num_feature, num_asset, T = data_global.shape
    btc_price_tensor = np.ones((num_feature, 1, T))
    data_global = np.concatenate((btc_price_tensor, data_global), axis=1)
    print('Global data tensor shape:', data_global.shape)
    return data_global

def main():
    with open('config.json') as file:
        config = json.load(file)
    parser = build_parser()
    options = parser.parse_args()

    if options.mode == 'train':
        print('Setting up agent and training model...')
        policy = DecisionNetwork_CNN()
        data_global = np.load('data_global.npy')
        agent = Agent(policy, config, data=data_global)
        agent.train()
        torch.save(agent.policy, 'agent.pt')
    elif options.mode == 'load_data':
        print('Loading global data...')
        data_global = get_data(config)
        np.save('data_global.npy', data_global)
    elif options.mode == 'backtest':
        print('Preparing backtest on last trained model...')
        policy = torch.load('agent.pt')
        data_global = np.load('data_global.npy')
        agent = Agent(policy, config, data=data_global)
        bt = Backtest(agent, config)

main()


