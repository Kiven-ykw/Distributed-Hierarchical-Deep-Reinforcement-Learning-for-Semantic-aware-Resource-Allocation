""" the neural network embeded in the DQN agent """

from tensorflow.python.keras import Sequential,Input, Model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM
from config import Config


class NeuralNetwork:

    def __init__(self, input_ports=74,#11*Config().U+7,
                 output_ports=Config().n_high_actions,
                 num_neurons=(64, 32),     # 64 32
                 memory_step=8,
                 activation_function='relu'):

        self.input_ports = input_ports
        self.output_ports = output_ports
        self.num_neurons = num_neurons
        self.activation_function = activation_function
        self.memory_step = memory_step

    def get_model(self, chooseNet):
        if chooseNet == 1:           # DQN
            model = Sequential()
            model.add(Dense(self.num_neurons[0], input_dim=self.input_ports, activation=self.activation_function))
            for j in range(1, len(self.num_neurons)):
                model.add(Dense(self.num_neurons[j], activation=self.activation_function))
            model.add(Dense(self.output_ports))
        return model


class NeuralNetwork_low:

    def __init__(self, input_ports=74,#11*Config().U+7,
                 output_ports=Config().n_low_actions,
                 num_neurons=(64, 32),     # 64 32
                 memory_step=8,
                 activation_function='relu'):

        self.input_ports = input_ports
        self.output_ports = output_ports
        self.num_neurons = num_neurons
        self.activation_function = activation_function
        self.memory_step = memory_step

    def get_model(self, chooseNet):
        if chooseNet == 1:           # DQN
            model = Sequential()
            model.add(Dense(self.num_neurons[0], input_dim=self.input_ports, activation=self.activation_function))
            for j in range(1, len(self.num_neurons)):
                model.add(Dense(self.num_neurons[j], activation=self.activation_function))
            model.add(Dense(self.output_ports))
        return model


