# %%
import torch
from torch import nn
import numpy as np
from scipy.io import savemat, loadmat
import os, sys
import glob
from scipy import interpolate
import matplotlib.pyplot as plt

current_path = os.path.dirname(__file__)
parent_path = os.path.dirname(current_path)
os.chdir(parent_path)
sys.path.append('.')
from WNN_tools import WNN
from CSNN_tools import CSNN_create, CSNN_cal
os.chdir(current_path)
sys.path.append('.')

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
N1=10
N2=99
dt = np.array([0.05, 0.125, 0.0625, 0.02, 0.01])
fre = 1./dt
Nt=np.floor(50./dt).astype(int)+1

t = [[] for _ in range(5)]
u = [[] for _ in range(5)]
y_ref = [[] for _ in range(5)]
y_pre = [[[] for _ in range(2)] for _ in range(5)]
# y_pre = [[] for _ in range(5)]

t[0] = np.linspace(0, (Nt[0]-1)*dt[0], Nt[0])
f = loadmat(name+'_data_u.mat')
u[0] = f['u']
y_ref[0] = OUT(u[0], dt[0])
del f

for i in range(1,5):
    t[i] = np.linspace(0, (Nt[i]-1)*dt[i], Nt[i])
    u[i] = np.zeros((Nt[i], N2))
    for j in range(0, N2):
        func = interpolate.interp1d(t[0], u[0][:,j])
        u[i][:,j] = func(t[i])
    y_ref[i] = OUT(u[i], dt[i])

# j = 0
# tend = 10
# for i in range(0,5):
#     plt.plot(t[i][0:np.floor(tend/dt[i]).astype(int)], u[i][0:np.floor(tend/dt[i]).astype(int),j])
# plt.show()

# j = 88
# tend = 50
# for i in range(0,5):
#     plt.plot(t[i][0:np.floor(tend/dt[i]).astype(int)], y_ref[i][0:np.floor(tend/dt[i]).astype(int),j])
# plt.show()

# %%
in_s = 1
out_s = 1

# WNN
non_layer_s = 1
state_s=6
A_ini = -0.01 * torch.ones(state_s, state_s)
B_ini = torch.rand(in_s, state_s)
non_neuron = state_s * np.ones(non_layer_s, dtype=np.int32) + in_s
pt_match = glob.glob(os.path.join('./saved_models', '*_WNN_'+str(state_s)+'.pt'))

for i in range(0,5):
    WNN_model = WNN(in_s, state_s, out_s, non_layer_s, non_neuron, A_ini, B_ini, dt[i], 'cpu', False)
    WNN_model.load_state_dict(torch.load(pt_match[0], map_location='cpu'))
    u_torch = torch.tensor(u[i].reshape([Nt[i], N2, 1]), dtype=torch.float)
    [y_WNN_torch,x] = WNN_model(u_torch)
    y_pre[i][0] = y_WNN_torch.detach().reshape([Nt[i], N2]).numpy()

# CSNN
state_non_layer_s=2
out_non_layer_s=1
state_non_neuron = 2*np.ones(2, dtype=np.int32)  # size of each nonlinear layer
out_non_neuron = np.ones(1, dtype=np.int32)  # size of each nonlinear layer
state_s=2
CSNN_model = CSNN_create(in_s, state_s, out_s, state_non_layer_s, out_non_layer_s, state_non_neuron, out_non_neuron, 'cpu', False)
pt_match = glob.glob(os.path.join('./saved_models', '*_CSNN_'+str(state_s)+'.pt'))
CSNN_model.load_state_dict(torch.load(pt_match[0], map_location='cpu'))

for i in range(0,5):
    y_CSNN_torch = CSNN_cal(N2, Nt[i], dt[i], u[i].reshape([Nt[i], N2, 1]), CSNN_model, state_s, out_s, 'cpu')
    y_pre[i][1] = y_CSNN_torch.detach().reshape([Nt[i], N2]).numpy()


PCC = [[] for _ in range(5)]

for k in range(5):
    PCC[k] = np.zeros((N2, 2))
    for i in range(N2):
        for j in range(2):
            PCC[k][i, j] = np.corrcoef(y_ref[k][:, i], y_pre[k][j][:, i])[0, 1]

PCC_aver = [[] for _ in range(5)]
for k in range(5):
    PCC_aver[k] = np.zeros((3, 2))
    PCC_aver[k][0, :] = np.mean(PCC[k][0:N1, :], axis=0)
    PCC_aver[k][1, :] = np.mean(PCC[k][N1:N2, :], axis=0)
    PCC_aver[k][2, :] = np.mean(PCC[k], axis=0)
    # print(PCC_aver[k])

model_name1=['WNN','CSNN']
for k in range(1, 5):
    print('\midrule')
    print('\multirow{2}{*}{'+str(int(fre[k]))+' Hz}')
    for i in range(2):
        val1, idx1 = min((val, idx) for (idx, val) in enumerate(PCC[k][0:N1, i]))
        val2, idx2 = min((val, idx) for (idx, val) in enumerate(PCC[k][N1:N2, i]))
        print('& '+model_name1[i]+' & '+'{:.5f}'.format(PCC_aver[k][0, i].item())+' & '+'{:.5f}'.format(val1.item())+' (case '+str(idx1+1)+')'
              +' & '+'{:.5f}'.format(PCC_aver[k][1, i].item())+' & '+'{:.5f}'.format(val2.item())+' (case '+str(idx2+N1+1)+')'+r' \\')