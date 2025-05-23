
"""
------------------------------------------------------------------------------------------------------
Simulation for paper:

             Distributed Hierarchical Deep Reinforcement Learning for Semantic-aware Resource Allocation


Author  : kaiwen yu
Date    : 2024/11/10
------------------------------------------------------------------------------------------------------
"""



from cellular_network import CellularNetwork as WN
import json
import random
import numpy as np
from config import Config
import os
import scipy.io as sio
import warnings
warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

c = Config()
random.seed(c.random_seed)
np.random.seed(c.random_seed)
WN = WN()
mos = []
r_high_last = 0
r_bs_last = 0
s_high = WN.observe_high()
s_high_last = s_high
actions_high = WN.choose_actions(s_high_last, flag=True)
WN.update(ir_change=False, flag=True, actions=actions_high)
r_accumulate = np.zeros(Config().n_links)
s_low = WN.observe_low()
step_per_beamforming = 20
ue_QoE = []
high_reward = []
count_beam_action = []
count_k_action = []
count_power_action = []
WN.draw_topology()
for _ in range(c.total_slots):
    print(_)
    if _ % step_per_beamforming == 0:
        """Performing user beam in higher level"""
        r_discounted = r_accumulate * (Config().reward_decay ** step_per_beamforming)
        WN.save_high_transitions(s_high_last, actions_high, r_discounted, s_high)
        actions_high = WN.choose_actions(s_high, flag=True)
        WN.update(ir_change=False,  flag=True, actions=actions_high)

        s_high_last = s_high
        r_accumulate = np.zeros(Config().n_links)

        count_temp = []
        for link in WN.links:
            count_temp.append(link.bs.code_index)
        count_beam_action.append(count_temp)

    """Performing semantic compress ratio config and power allocation in lower level, conditioned on the association"""
    s_low = WN.observe_low()
    actions = WN.choose_actions(s_low, flag=False)
    WN.update(ir_change=False, flag=False, actions=actions)
    ue_QoE.append(WN.get_ave_qoe())
    high_reward.append(WN.give_rewards())
    WN.update(ir_change=True)

    r_bs = WN.give_rewards()
    r_bs_new = r_bs - r_bs_last

    s_low_next = WN.observe_low()

    WN.save_low_transitions(s_low, actions, r_bs_new, s_low_next)

    s_high = WN.observe_high()

    r_bs_last = r_bs
    r_accumulate += r_bs

    count_k_temp = []
    count_power_temp = []
    for link in WN.links:
        count_k_temp.append(link.bs.k_u)
        count_power_temp.append(link.bs.power_index)
    count_k_action.append(count_k_temp)
    count_power_action.append(count_power_temp)

    if _ > 256:
        WN.train_dqns(True)
        WN.train_dqns(False)
WN.save_models()

# save data
filename = 'data/drl_dqn_qoe_global_performance30000.json'
with open(filename, 'w') as f:
    json.dump(ue_QoE, f)

beam_m = np.array(count_beam_action)
sio.savemat('action_hotmap/beam.mat', {'beam': beam_m})

k_u_m = np.array(count_beam_action)
sio.savemat('action_hotmap/k_u.mat', {'k_u': k_u_m})

power_m = np.array(count_power_action)
sio.savemat('action_hotmap/power.mat', {'power': power_m})





