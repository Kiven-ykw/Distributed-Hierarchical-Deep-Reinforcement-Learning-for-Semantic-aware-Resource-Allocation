""" the configuration of the simulation parameters """

import functions as f
import numpy as np


class Config:

    def __init__(self):
        # Base Station
        self.n_k_level = 20
        self.n_antennas = f.get_codebook().shape[0]  # number of transmit antennas
        self.codebook_size = f.get_codebook().shape[1]  # number of codes
        self.n_power_levels = 5  # number of discrete power levels
        self.n_high_actions = self.codebook_size
        self.n_low_actions = self.n_power_levels * self.n_k_level

        self.bs_power = 38  # maximum transmit power of base stations

        self.reward_decay = 0.99

        # Channel
        self.angular_spread = 3 / 180 * np.pi  # angular spread
        self.multi_paths = 4  # number of multi-paths
        self.rho = 0.64  # channel correlation coefficient
        self.noise_power = f.dB2num(-114)  # noise power
        self.U = 6

        # Cellular Network
        self.cell_radius = 200  # cell radius
        self.n_links = 7 # number of simulated direct links in the simulation
        self.inner_cell_radius = 10  # inner cell radius

        # Simulation
        self.slot_interval = 0.02  # interval of one time slot
        self.random_seed = 2022  # random seed to control the simulated cellular network
        self.total_slots = 5000   # total time slots in the simulation
