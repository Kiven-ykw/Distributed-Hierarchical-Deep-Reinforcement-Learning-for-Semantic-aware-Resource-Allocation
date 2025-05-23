""" visualize the simulation results """

import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

def data_visualization():

    r = []
    for i in range(1, 20, 1):
        filename = 'interferenceVSeePL5/DRQN_common_PL=5_i={}.json'.format(i)
        with open(filename, 'r') as f:
            p = np.array(json.load(f))
            r.append(p)
    r = np.array(r)
    plt.plot(r, label='DRQN-cooperative')
    sio.savemat('interferenceVSeePL5/DRQN_commonPL5inter.mat', {'DRQN_commonPL5inter': r})

    r = []
    for i in range(1, 20, 1):
        filename = 'interferenceVSeePL5/DRQN_competitive_PL=5_i={}.json'.format(i)
        with open(filename, 'r') as f:
            p = np.array(json.load(f))
            r.append(p)
    r = np.array(r)
    plt.plot(r, label='DRQN-competitive')
    sio.savemat('interferenceVSeePL5/DRQN_competitivePL5inter.mat', {'DRQN_competitivePL5inter': r})

    r = []
    for i in range(1, 20, 1):
        filename = 'interferenceVSeePL5/DQN_common_PL=5_i={}.json'.format(i)
        with open(filename, 'r') as f:
            p = np.array(json.load(f))
            r.append(p)
    r = np.array(r)
    plt.plot(r, label='DQN-cooperative')
    sio.savemat('interferenceVSeePL5/DQN_commonPL5inter.mat', {'DQN_commonPL5inter': r})

    r = []
    for i in range(1, 20, 1):
        filename = 'interferenceVSeePL5/DQN_competitive_PL=5_i={}.json'.format(i)
        with open(filename, 'r') as f:
            p = np.array(json.load(f))
            r.append(p)
    r = np.array(r)
    plt.plot(r, label='DQN-competitive')
    sio.savemat('interferenceVSeePL5/DQN_competitivePL5inter.mat', {'DQN_competitivePL5inter': r})

    plt.xlabel('Maximum transmit power')
    plt.ylabel('EE')
    plt.legend(loc=0)
    plt.grid(True)
    plt.show()


data_visualization()
