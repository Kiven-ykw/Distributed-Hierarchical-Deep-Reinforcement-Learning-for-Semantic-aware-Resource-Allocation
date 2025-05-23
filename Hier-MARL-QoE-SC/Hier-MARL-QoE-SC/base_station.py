"""simulator for base stations """

import numpy as np
import random
import functions as f
from config import Config
from DRQN import DQN_low
from dqn_for_singleagent import DQN


class BaseStation:

    def __init__(self, location, index):
        """ initialize the attributes of BS """

        self.location = location
        self.index = index
        self.n_antennas = Config().n_antennas
        self.lower_nn = DQN_low()
        self.higher_nn = DQN()
        self.max_power = f.dB2num(Config().bs_power)
        self.codebook = f.get_codebook()
        self.powerbook = np.arange(Config().n_power_levels) * self.max_power / (Config().n_power_levels - 1)
        self.k_u_codebook = np.arange(Config().n_k_level)
        self.code_index = random.randint(0, Config().codebook_size - 1)
        self.power_index = random.randint(0, Config().n_power_levels - 1)
        self.k_u_index = random.randint(0, Config().n_k_level - 1)

        self.code = self.codebook[:, self.code_index]
        self.power = self.powerbook[self.power_index]
        self.k_u = self.k_u_codebook[self.k_u_index]

        #self.interferer_neighbors = None
        self._init_params_()

    def _init_params_(self):
        """ initialize some variables to save the historical actions """

        self.code_index1, self.code_index2 = None, None
        self.power_index1, self.power_index2 = None, None
        self.k_u_index1, self.k_u_index2 = None, None
        self.power1, self.power2 = None, None
        self.k_u1, self.k_u2 = None, None

    def get_k_u(self):
        return self.k_u

    def _save_high_params_(self):

        self.code_index2 = self.code_index1
        self.code_index1 = self.code_index

    def _save_low_params_(self):

        self.power_index2 = self.power_index1
        self.power_index1 = self.power_index

        self.k_u_index2 = self.k_u_index1
        self.k_u_index1 = self.k_u_index

        self.power2 = self.power1
        self.power1 = self.power

        self.k_u2 = self.k_u1
        self.k_u1 = self.k_u

    def take_high_action(self, action=None, weight=None):
        self._save_high_params_()
        if action is not None:
            self.code_index = action
            self.code = self.codebook[:, self.code_index]

        if weight is not None:
            if np.linalg.norm(weight) != 0:
                self.code = weight / np.linalg.norm(weight)
            else:
                self.code = np.zeros(Config().n_antennas, dtype=np.complex)

    def take_low_action(self, action=None, weight=None):
        self._save_low_params_()
        if action is not None:
            self.power_index = action % Config().n_power_levels
            self.k_u_index = action // Config().n_power_levels
            self.k_u = self.k_u_codebook[self.k_u_index]
            self.power = self.powerbook[self.power_index]
        if weight is not None:
            if np.linalg.norm(weight) != 0:
                self.k_u = weight / np.linalg.norm(weight)
                self.power = np.square(np.linalg.norm(weight))
            else:
                self.k_u = np.zeros(Config().n_k_level, dtype=np.complex)
                self.power = 0


