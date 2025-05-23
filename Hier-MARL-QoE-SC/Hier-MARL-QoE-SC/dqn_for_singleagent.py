""" DQN agent at each base station """

import numpy as np
import random
from neural_network import NeuralNetwork
from collections import deque
from tensorflow.python.keras.optimizers import rmsprop_v2
import os
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import Sequential
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class DQN:
    # hyper params
    def __init__(self,
                 n_actions=NeuralNetwork().output_ports,
                 n_features=NeuralNetwork().input_ports,
                 lr=5e-4,
                 lr_decay=1e-4,
                 reward_decay=0.5,
                 e_greedy=0.6,
                 epsilon_min=1e-2,
                 replace_target_iter=100,
                 memory_size=500,
                 batch_size=8,
                 e_greedy_decay=1e-4):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = lr
        self.lr_decay = lr_decay
        self.gamma = reward_decay
        # epsilon-greedy params
        self.epsilon = e_greedy
        self.epsilon_decay = e_greedy_decay
        self.epsilon_min = epsilon_min
        self.save_path = 'model/'

        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size

        self.loss = []
        self.accuracy = []

        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2))
        self.lstm_obs_history = deque([[0 for y in range(self.n_features)] for x in range(8)], 8)
        self._built_net()
        self.model = Sequential()
    #
    # def clear_history(self):
    #     self.lstm_obs_history = deque([[0 for y in range(self.n_features)] for x in range(8)], 8)
    # #
    # #
    def load_mod(self, indx):
        self.model = load_model('model/DQN_common_PL=5BS_{}.hdf5'.format(indx))
    #
    # def choose_action(self, observation):
    #     # epsilon greedy
    #     observation = observation[np.newaxis, :]
    #     actions_value = self.model.predict(observation)
    #     action = np.argmax(actions_value)
    #     return action

    # def choose_action(self, observation):
    #     self.lstm_obs_history.append(observation)
    #     #choose action from model
    #     q_values = self.model.predict(np.array(self.lstm_obs_history).reshape(1, 8, self.n_features))
    #     action = np.argmax(q_values)
    #     self.clear_history()
    #     return action

    def _built_net(self):

        tar_nn = NeuralNetwork()
        eval_nn = NeuralNetwork()
        self.model1 = tar_nn.get_model(1)
        self.model2 = eval_nn.get_model(1)
        self.target_replace_op()
        # RMSProp optimizer
        optimizer = rmsprop_v2.RMSprop(lr=self.lr, decay=self.lr_decay)

        self.model2.compile(loss='mse', optimizer=optimizer)   # 将字符串编译为字节代码

    def _store_transition_(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, a, r, s_))
        # print(transition, '\n')
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = np.array(transition)
        self.memory_counter += 1

    def save_transition(self, s, a, r, s_):
        self._store_transition_(s, a, r, s_)

    def choose_action(self, observation):
        # epsilon greedy
        if random.uniform(0, 1) > self.epsilon:
            observation = observation[np.newaxis, :]
            actions_value = self.model2.predict(observation)
            action = np.argmax(actions_value)
        else:
            action = random.randint(0, self.n_actions - 1)
        return action

    def save_model(self, file_name):
        file_path = self.save_path + file_name + '.hdf5'
        self.model2.save(file_path, True)

    def target_replace_op(self):
        temp = self.model2.get_weights()
        print('Parameters updated')
        self.model1.set_weights(temp)

    def learn(self):
        # update target network's params
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_replace_op()

        # sample mini-batch from experience replay
        if self.memory_counter > self.memory_size:
            sample_index = random.sample(list(range(self.memory_size)), self.batch_size)
        else:
            sample_index = random.sample(list(range(self.memory_counter)), self.batch_size)

        # mini-batch data
        batch_memory = self.memory[sample_index, :]
        q_next = self.model1.predict(batch_memory[:, -self.n_features:])   # 下个状态传给Q目标值网络
        q_eval = self.model2.predict(batch_memory[:, :self.n_features])    # 当前状态传给Q估计值网络
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        hist = self.model2.fit(batch_memory[:, :self.n_features], q_target, verbose=0)
        self.loss.append(hist.history['loss'][0])

        self.epsilon = max(self.epsilon / (1 + self.epsilon_decay), self.epsilon_min)
        self.learn_step_counter += 1

