import matplotlib.pyplot as plt
import torch
import math
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

m=1
c=1
k_1=20
k_2=200
A=np.array([[0., 1.],[-k_1/m, -c/m]])
B=np.array([[0.],[-k_2/m]])
E=np.array([[0.],[-1.]])

def SYS(y_dy,y,u):
    dy_ddy = np.dot(A, y_dy) + np.dot(B, np.power(y, 3)) + np.dot(E, u)
    return dy_ddy

def OUT(u, dt):
    y=np.zeros_like(u)
    dy=np.zeros_like(u)
    Nt=u.shape[0]
    for i in range(0, Nt-1):
        y_dy1 = np.concatenate((y[i:i+1,:],dy[i:i+1,:]), axis=0)
        k1 = SYS(y_dy1, y[i:i+1,:], u[i:i+1,:])

        u_mid = (u[i:i+1,:] + u[i+1:i+2,:]) * 0.5
        y_dy2 = y_dy1 + k1 * 0.5 * dt
        k2 = SYS(y_dy2, y_dy2[0:1,:], u_mid)

        y_dy3 = y_dy1 + k2 * 0.5 * dt
        k3 = SYS(y_dy3, y_dy3[0:1,:], u_mid)

        y_dy4 = y_dy1 + k3 * dt
        k4 = SYS(y_dy4, y_dy4[0:1,:], u[i+1:i+2,:])

        y_dy = y_dy1 + (k1 + 2*k2 + 2*k3 + k4) * dt / 6
        y[i+1:i+2,:] = y_dy[0:1,:]
        dy[i+1:i+2,:] = y_dy[1:2,:]
    return y

name='NS'
dt=0.05
Nt=math.floor(50/dt)+1
N1=10
N2=99
f = loadmat(name+'_data_u.mat')
u = f['u'].reshape([Nt, N2, 1])
y_ref = OUT(f['u'], dt).reshape([Nt, N2, 1])

tend = (Nt - 1) * dt
t = np.linspace(0, tend, Nt).reshape(-1, 1)
del f

current_path = os.path.dirname(__file__)
parent_path = os.path.dirname(current_path)
os.chdir(parent_path)
sys.path.append('.')
from general_tools import time_str