from cellular_network import CellularNetwork as CN
import json
import random
import numpy as np
import scipy.io as sio
from config import Config
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


c = Config()
random.seed(c.random_seed)
np.random.seed(c.random_seed)
cn = CN()
utility = []
qoe = []
cn.draw_topology()
rate_m = []
for _ in range(c.total_slots):
    print(_)
    s = cn.observe_low()
    #print(s.shape)
    actions = cn.choose_actions(s)
    cn.update(ir_change=False, actions=actions)
    # utility.append(cn.get_ave_utility())
    # rate_m.append(cn.get_all_rates())
    qoe.append(cn.get_ave_qoe())
    cn.update(ir_change=True)
    r = cn.give_rewards()
    s_ = cn.observe_low()
    cn.save_transitions(s, actions, r, s_)

    if _ > 256:
        cn.train_dqns()


# save data
filename = 'data/drl_dqn_qoe_global_performance30000.json'
with open(filename, 'w') as f:
    json.dump(qoe, f)
# rate_m = np.array(rate_m)
# sio.savemat('rates/drl_dqn_global_rates.mat', {'dqn_global_rates': rate_m})
