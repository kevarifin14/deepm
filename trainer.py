import numpy as np

class Trainer():
    def __init__(self, market_history, agent):
        self.market_history = market_history
        self.agent = agent

    def train(self):
        for i in range(self.agent.train_iterations):
            X, Y = self.next_batch()
            self.agent.train(X, Y)
        
    def next_batch(self):
        X = np.zeros((self.agent.batch_size, self.market_history.data.shape[0], self.market_history.data.shape[1], self.market_history.data.shape[2]))
        Y = np.zeros((self.agent.batch_size, self.market_history.data.shape[1]))
        for i in range(self.agent.batch_size):
            index = np.random.geometric(0.1)            
            while index > market_history:
                index = np.random.geometric(0.1)
``          x = self.market_history.data[:, :, -index-self.agent.window:-index]
            y = self.market_history.data[:, :, -index]
            X[i] = x
            Y[i] = y
        return X, Y
        