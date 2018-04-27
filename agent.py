import numpy as np
from constants import *

class Agent:
    def __init__(self, net, market_history, period=1800, window=50, batch_size=50):
        self.period = period
        self.window = window
        self.batch_size = batch_size
        self.net = net
        self.train_iterations = 100
        self.market_history = market_history
        self.w = np.zeros(market_history.data.shape[1])


    def train(self):
        for i in range(self.train_iterations):
            X, Y = self.next_batch()
            print(self.net.forward(X))
            # net.backward(X, Y)

    def next_batch(self):
        X = np.zeros((self.batch_size, self.market_history.data.shape[0], self.market_history.data.shape[1], self.window))
        Y = np.zeros((self.batch_size, self.market_history.data.shape[1]))
        for i in range(self.batch_size):
            index = np.random.geometric(0.1)            
            while index > self.market_history.data.shape[-1] - self.window:
                index = np.random.geometric(0.1)
            x = self.market_history.data[:, :, -index-self.window:-index]
            y = self.market_history.data[0, :, -index]
            X[i] = x
            Y[i] = y
        return X, Y
    
    # def train(self):
    #     total_data_time = 0
    #     total_training_time = 0
    #     for i in range(self.train_config["steps"]):
    #         step_start = time.time()
    #         x, y, last_w, setw = self.next_batch()
    #         finish_data = time.time()
    #         total_data_time += (finish_data - step_start)
    #         self._agent.train(x, y, last_w=last_w, setw=setw)
    #         total_training_time += time.time() - finish_data
    #         if i % 1000 == 0 and log_file_dir:
    #             logging.info("average time for data accessing is %s"%(total_data_time/1000))
    #             logging.info("average time for training is %s"%(total_training_time/1000))
    #             total_training_time = 0
    #             total_data_time = 0
    #             self.log_between_steps(i)

    #     if self.save_path:
    #         self._agent.recycle()
    #         best_agent = NNAgent(self.config, restore_dir=self.save_path)
    #         self._agent = best_agent

    #     pv, log_mean = self._evaluate("test", self._agent.portfolio_value, self._agent.log_mean)
    #     logging.warning('the portfolio value train No.%s is %s log_mean is %s,'
    #                     ' the training time is %d seconds' % (index, pv, log_mean, time.time() - starttime))

    #     return self.__log_result_csv(index, time.time() - starttime)


    # def __set_loss_function(self):
    #     def loss_function4():
    #         return -tf.reduce_mean(tf.log(tf.reduce_sum(self.__net.output[:] * self.__future_price,
    #                                                     reduction_indices=[1])))

    #     def loss_function5():
    #         return -tf.reduce_mean(tf.log(tf.reduce_sum(self.__net.output * self.__future_price, reduction_indices=[1]))) + \
    #                LAMBDA * tf.reduce_mean(tf.reduce_sum(-tf.log(1 + 1e-6 - self.__net.output), reduction_indices=[1]))

    #     def loss_function6():
    #         return -tf.reduce_mean(tf.log(self.pv_vector))

    #     def loss_function7():
    #         return -tf.reduce_mean(tf.log(self.pv_vector)) + \
    #                LAMBDA * tf.reduce_mean(tf.reduce_sum(-tf.log(1 + 1e-6 - self.__net.output), reduction_indices=[1]))

    #     def with_last_w():
    #         return -tf.reduce_mean(tf.log(tf.reduce_sum(self.__net.output[:] * self.__future_price, reduction_indices=[1])
    #                                       -tf.reduce_sum(tf.abs(self.__net.output[:, 1:] - self.__net.previous_w)
    #                                                      *self.__commission_ratio, reduction_indices=[1])))

    #     loss_function = loss_function5
    #     if self.__config["training"]["loss_function"] == "loss_function4":
    #         loss_function = loss_function4
    #     elif self.__config["training"]["loss_function"] == "loss_function5":
    #         loss_function = loss_function5
    #     elif self.__config["training"]["loss_function"] == "loss_function6":
    #         loss_function = loss_function6
    #     elif self.__config["training"]["loss_function"] == "loss_function7":
    #         loss_function = loss_function7
    #     elif self.__config["training"]["loss_function"] == "loss_function8":
    #         loss_function = with_last_w

    #     loss_tensor = loss_function()
    #     regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    #     if regularization_losses:
    #         for regularization_loss in regularization_losses:
    #             loss_tensor += regularization_loss
    #     return loss_tensor

    # def init_train(self, learning_rate, decay_steps, decay_rate, training_method):
    #     learning_rate = tf.train.exponential_decay(learning_rate, self.__global_step,
    #                                                decay_steps, decay_rate, staircase=True)
    #     if training_method == 'GradientDescent':
    #         train_step = tf.train.GradientDescentOptimizer(learning_rate).\
    #                      minimize(self.__loss, global_step=self.__global_step)
    #     elif training_method == 'Adam':
    #         train_step = tf.train.AdamOptimizer(learning_rate).\
    #                      minimize(self.__loss, global_step=self.__global_step)
    #     elif training_method == 'RMSProp':
    #         train_step = tf.train.RMSPropOptimizer(learning_rate).\
    #                      minimize(self.__loss, global_step=self.__global_step)
    #     else:
    #         raise ValueError()
    #     return train_step

    # def train(self, x, y, last_w, setw):
    #     tflearn.is_training(True, self.__net.session)
    #     self.evaluate_tensors(x, y, last_w, setw, [self.__train_operation])

    # def evaluate_tensors(self, x, y, last_w, setw, tensors):
    #     """
    #     :param x:
    #     :param y:
    #     :param last_w:
    #     :param setw: a function, pass the output w to it to fill the PVM
    #     :param tensors:
    #     :return:
    #     """
    #     tensors = list(tensors)
    #     tensors.append(self.__net.output)
    #     assert not np.any(np.isnan(x))
    #     assert not np.any(np.isnan(y))
    #     assert not np.any(np.isnan(last_w)),\
    #         "the last_w is {}".format(last_w)
    #     results = self.__net.session.run(tensors,
    #                                      feed_dict={self.__net.input_tensor: x,
    #                                                 self.__y: y,
    #                                                 self.__net.previous_w: last_w,
    #                                                 self.__net.input_num: x.shape[0]})
    #     setw(results[-1][:, 1:])
    #     return results[:-1]

    # # save the variables path including file name
    # def save_model(self, path):
    #     self.__saver.save(self.__net.session, path)

    # # consumption vector (on each periods)
    # def __pure_pc(self):
    #     c = self.__commission_ratio
    #     w_t = self.__future_omega[:self.__net.input_num-1]  # rebalanced
    #     w_t1 = self.__net.output[1:self.__net.input_num]
    #     mu = 1 - tf.reduce_sum(tf.abs(w_t1[:, 1:]-w_t[:, 1:]), axis=1)*c
    #     """
    #     mu = 1-3*c+c**2

    #     def recurse(mu0):
    #         factor1 = 1/(1 - c*w_t1[:, 0])
    #         if isinstance(mu0, float):
    #             mu0 = mu0
    #         else:
    #             mu0 = mu0[:, None]
    #         factor2 = 1 - c*w_t[:, 0] - (2*c - c**2)*tf.reduce_sum(
    #             tf.nn.relu(w_t[:, 1:] - mu0 * w_t1[:, 1:]), axis=1)
    #         return factor1*factor2

    #     for i in range(20):
    #         mu = recurse(mu)
    #     """
    #     return mu

    # # the history is a 3d matrix, return a asset vector
    # def decide_by_history(self, history, last_w):
    #     assert isinstance(history, np.ndarray),\
    #         "the history should be a numpy array, not %s" % type(history)
    #     assert not np.any(np.isnan(last_w))
    #     assert not np.any(np.isnan(history))
    #     tflearn.is_training(False, self.session)
    #     history = history[np.newaxis, :, :, :]
    #     return np.squeeze(self.session.run(self.__net.output, feed_dict={self.__net.input_tensor: history,
    #                                                                      self.__net.previous_w: last_w[np.newaxis, 1:],
    #                                                                      self.__net.input_num: 1}))
