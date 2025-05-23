from cellular_network import CellularNetwork as CN
from dqn_for_singleagent import DQN
import json
import random
import numpy as np
import scipy.io as sio
from config import Config
import os
import warnings
warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

c = Config()
random.seed(c.random_seed)
np.random.seed(c.random_seed)
cn = CN()
ee = []
r_last = 0
cn.draw_topology()
ee_m = []
R = []
for _ in range(c.total_slots):
    print(_)
    s = cn.observe()
    actions = cn.choose_actions(s)
    cn.update(ir_change=False, actions=actions)
    ee.append(cn.get_ave_ee())
    ee_m.append(cn.get_all_ees())
    cn.update(ir_change=True)
    r = cn.give_rewards(is_cooperation=True)  # the temp reward at current time slot
    r_new = r - r_last                         # the reward at current time slot
    r_last = r                                 # the temp reward at last time slot
    s_ = cn.observe()
    R.append(r_new)
    cn.save_transitions(s, actions, r_new, s_)
    if _ > 256:
        cn.train_dqns()
cn.save_models('DQN_common_PL=10_dft88_oe=0.2')

filename = 'ee/DQN_commonR_PL=10_dft88_oe=0.2_TS=30000.json'
with open(filename, 'w') as f:
    json.dump(ee, f)

