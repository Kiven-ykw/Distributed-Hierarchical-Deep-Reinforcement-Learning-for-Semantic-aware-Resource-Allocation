""" visualize the simulation results """

import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

window = 500
def data_visualization():
    r = []
    filename = 'data/drl_dqn_qoe_global_performance30000.json'
    with open(filename, 'r') as f:
        data = np.array(json.load(f))
        print(len(data))
        for i in range(len(data)-window +1):
            r.append(np.mean(data[i:i+window]))
    r = np.array(r)
    plt.plot(r, label='DQN')
    #sio.savemat('powerVSeePL5/DRQN_commonPL5power.mat', {'DRQN_commonPL5power': r})

    # r = []
    # for i in range(10, 71, 5):
    #     filename = 'powerVSeePL5/DRQN_competitive_PL=5_power={}.json'.format(i)
    #     with open(filename, 'r') as f:
    #         p = np.array(json.load(f))
    #         r.append(p)
    # r = np.array(r)
    # plt.plot(r, label='DRQN-competitive')


    plt.xlabel('Time slots')
    plt.ylabel('QoE')
    plt.legend(loc=0)
    plt.grid(True)
    plt.show()


data_visualization()
