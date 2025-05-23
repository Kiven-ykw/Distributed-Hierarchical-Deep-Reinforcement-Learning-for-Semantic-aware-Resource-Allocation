""" DTDE DRL-based scheme """
from cellular_network import CellularNetwork as CN
#from DRQN import drqn
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
cn.load_models()
ee = []
se = []
se_final = []
ee_final = []
#cn.draw_topology()

for _ in range(0, 100, 1):
    print(_)
    # for i in range(0, 100):
    s = cn.observe()
    actions = cn.choose_actions(s)
    cn.update(ir_change=False, actions=actions)
    ee.append(cn.get_ave_ee())
    se.append(cn.get_ave_utility())
    cn.update(ir_change=True)
ee_final.append(np.sum(ee)/100)
se_final.append(np.sum(se)/100)


filename = 'powerVSsePL5/DQN_common_PL=5_power=70.json'
with open(filename, 'w') as f:
    json.dump(se_final, f)
