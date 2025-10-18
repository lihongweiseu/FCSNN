import matplotlib.pyplot as plt
import torch
from torch import nn
import time
import numpy as np
from scipy.io import savemat, loadmat
import os, sys
import glob
import random
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)

device = torch.device('cpu') # 'cuda:0' 'cpu'
torch.set_default_dtype(torch.float)
bias_status=False

name='HB'
dt=0.02
Nt=2500
N1=15
N2=21
f = loadmat(name+'_data.mat')
u = f['u'].reshape([Nt, N2, 1])
y_ref = f['y_ref'].reshape([Nt, N2, 1])

tend = (Nt - 1) * dt
t = np.linspace(0, tend, Nt).reshape(-1, 1)
del f

current_path = os.path.dirname(__file__)
parent_path = os.path.dirname(current_path)
os.chdir(parent_path)
sys.path.append('.')
from general_tools import time_str