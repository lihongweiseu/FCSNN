# %%
import matplotlib.pyplot as plt
import torch
from torch import nn
import time
import numpy as np
from scipy.io import savemat, loadmat
import os, sys
import glob
current_path = os.path.dirname(__file__)
parent_path = os.path.dirname(current_path)
os.chdir(parent_path)
sys.path.append('.')
from FCSNN_tools1 import FCSNN, FCSNN_training
from general_tools import time_str
os.chdir(current_path)
sys.path.append('.')
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

name='MR'
# Prepare data
dt=0.005
Nt1=2001
Nt2=3499
f = loadmat(name+'_data.mat')
u = f['u'].reshape([Nt2, 1, 1])
y_ref = f['y_ref'].reshape([Nt2, 1, 1])
# u_max = np.max(np.abs(u))
# y_max = np.max(np.abs(y_ref))
# u_scale = u/u_max
# y_ref_scale = y_ref/y_max
tend = (Nt2 - 1) * dt
t = np.linspace(0, tend, Nt2).reshape(-1, 1)
del f
bias_status=False

# Define the configuration of FCSNN
in_s = 1
out_s = 1
non_layer_s = 1
Para = torch.load('Para.pt')
# %%
N = 20000 # training num 20000
All_start = time.time()
for i in range(8, 9):
    state_s=i
    Para = torch.load('Para.pt')
    non_neuron = state_s * np.ones(non_layer_s, dtype=np.int32) + in_s
    FCSNN_model = FCSNN(in_s, state_s, out_s, non_layer_s, non_neuron, Para, dt, device, bias_status)
    trainable_num = sum(p.numel() for p in FCSNN_model.parameters() if p.requires_grad)
    print('State size: ',i,', total number of trainable parameters: ', trainable_num,'\n')

    FCSNN_model_m, loss_all, loss_m, cost_time, im, cost_time_str = FCSNN_training(dt, u[0:Nt2,:,:], FCSNN_model, N, y_ref[0:Nt2,:,:], in_s, state_s, out_s, non_layer_s, non_neuron, Para, device, bias_status)
    # loss_m = loss_m * y_max * y_max
    # loss_all = loss_all * y_max * y_max
    torch.save(FCSNN_model_m.state_dict(), './saved_models/'+name+'_FCSNN_'+str(state_s)+'_'+'{:.4e}'.format(loss_m).replace('e+0', 'e').replace('e-0', 'e-')+'_'+str(im)+'_'+cost_time_str+'_2w.pt')
    # savemat('./saved_data/'+name+'_FCSNN_'+str(state_s)+'_'+'{:.4e}'.format(loss_m).replace('e+0', 'e').replace('e-0', 'e-')+'_'+str(im)+'_'+cost_time_str+'_2w.mat', {'loss_all': loss_all,
    #  'cost_time': cost_time, 'trainable_num': trainable_num})

All_end = time.time()
All_cost_time = All_end - All_start
All_cost_time_str = time_str(All_cost_time)
print('Total training time for all cases: ' + All_cost_time_str)
# %%
state_s=8
A_ini = -0.01 * torch.rand(state_s, state_s)
B_ini = torch.rand(in_s, state_s)
non_neuron = state_s * np.ones(non_layer_s, dtype=np.int32)+ in_s
FCSNN_model = FCSNN(in_s, state_s, out_s, non_layer_s, non_neuron, Para, dt, 'cpu', bias_status)
pt_match = glob.glob(os.path.join('./saved_models', '*_FCSNN_'+str(state_s)+'_*2w_all.pt'))
# pt_match = glob.glob(os.path.join('./saved_models', 'MR_FCSNN_20_9.9996e1_19976_2h 7m 34.746s.pt'))
FCSNN_model.load_state_dict(torch.load(pt_match[0], map_location='cpu'))
u_torch = torch.tensor(u, dtype=torch.float)
Nt=Nt2
[y_pre_torch,x] = FCSNN_model(u_torch[0:Nt,:,:])
y_pre = y_pre_torch.reshape([Nt, 1]).detach().numpy()

plt.plot(t[0:Nt,:], y_ref[0:Nt,:,:].reshape([Nt, 1]))
plt.plot(t[0:Nt,:], y_pre)
plt.show()
# %%
# y_ref_torch = torch.tensor(y_ref, dtype=torch.float)
# criterion = nn.MSELoss()
# loss_FCSNN=np.zeros((20, 2))
# for i in range(1,21):
#     mat_match = glob.glob(os.path.join('./saved_data', '*_FCSNN_'+str(i)+'_*.mat'))
#     f = loadmat(mat_match[0])
#     loss_all=f['loss_all']
#     loss_FCSNN[i-1:i, :1]=np.min(loss_all)
#     pt_match=mat_match[0].replace('data', 'models').replace('mat', 'pt')
#     state_s=i
#     FCSNN_model = FCSNN_create(in_s, state_s, out_s, non_layer_s, non_neuron, Para, dt, device, bias_status)
#     FCSNN_model.load_state_dict(torch.load(pt_match, map_location='cpu'))
#     u_torch_scale = torch.tensor(u_scale[0:Nt,:,:], dtype=torch.float)
#     [y_pre_torch_scale,x] = FCSNN_model(u_torch_scale)
#     loss_FCSNN[i-1:i, 1:2] = criterion(y_pre_torch_scale[Nt1:Nt2,:,:], y_ref_torch_scale[Nt1:Nt2,:,:])* y_max * y_max.item()

# del mat_match, pt_match, f, loss_all
# savemat(name+'_loss_FCSNN.mat', {name+'_loss_FCSNN': loss_FCSNN})