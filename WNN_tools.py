# %%
import datetime
from time import strftime
import random
import numpy as np
import torch
from torch import nn, optim
from torch.nn.parameter import Parameter
from general_tools import time_str#, early_stop_check

# To guarantee same results for every running, which might slow down the training speed
torch.set_default_dtype(torch.float)
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)

def WNN(in_s, state_s, out_s, non_layer_s, non_neuron, A_ini, B_ini, dt, device, bias_status):
    # Define WNN
    class WNN(nn.Module):
        def __init__(self):
            super(WNN, self).__init__()
            self.A = Parameter(A_ini.to(device)) #10
            self.B = Parameter(B_ini.to(device))
            
            self.lin = nn.Linear(state_s+in_s, out_s, bias=bias_status)
            
            layer_non = [nn.Linear(in_s + state_s, non_neuron[0], bias=bias_status), torch.nn.Tanh()]
            if non_layer_s > 1:
                for ii in range(non_layer_s - 1):
                    layer_non.append(nn.Linear(non_neuron[ii], non_neuron[ii + 1], bias=bias_status))
                    layer_non.append(torch.nn.Tanh())
            layer_non.append(nn.Linear(non_neuron[-1], out_s, bias=bias_status))
            self.non1 = nn.Sequential(*layer_non)

            layer_non = [nn.Linear(in_s + state_s, non_neuron[0], bias=bias_status), torch.nn.Tanhshrink()]
            if non_layer_s > 1:
                for ii in range(non_layer_s - 1):
                    layer_non.append(nn.Linear(non_neuron[ii], non_neuron[ii + 1], bias=bias_status))
                    layer_non.append(torch.nn.Tanhshrink())
            layer_non.append(nn.Linear(non_neuron[-1], out_s, bias=bias_status))
            self.non2 = nn.Sequential(*layer_non)

        def forward(self, u):
            Nt = u.shape[0]
            Amp = u.shape[1]
            x = torch.zeros(Nt, Amp, state_s).to(device)

            I = torch.eye(state_s).to(device)
            A_dt = self.A*dt
            A_dt0 = A_dt.detach()
            A2_dt2 = torch.matmul(A_dt0, A_dt0)
            A3_dt3 = torch.matmul(A2_dt2, A_dt0)
            A4_dt4 = torch.matmul(A3_dt3, A_dt0)
            # A5_dt5 = torch.matmul(A4_dt4, A_dt0)
            a = I+A_dt+A2_dt2/2+A3_dt3/6+A4_dt4/24 # +A5_dt5/120
            b = (I/2+A_dt/3+A2_dt2/8+A3_dt3/30)*dt # +A4_dt4/144
            c = (I/2+A_dt/6+A2_dt2/24+A3_dt3/120)*dt # +A4_dt4/720

            for i in range(Nt - 1):
                u0 = u[i, :, :]
                u1 = u[i+1, :, :]
                x0 = x[i, :, :].clone()
                x[i+1, :, :] = torch.matmul(x0,a) + torch.matmul(torch.matmul(u0,self.B),b) + torch.matmul(torch.matmul(u1,self.B),c)

            u_x = torch.cat((u, x), dim=2)
            y = self.lin(u_x)+self.non1(u_x)+self.non2(u_x)
            
            return y, x

    WNN_model = WNN().to(device)
    return WNN_model

def WNN_training(dt, u, WNN_model, N, y_ref, in_s, state_s, out_s, non_layer_s, non_neuron, A_ini, B_ini, device, bias_status):
    u_torch = torch.tensor(u, dtype=torch.float).to(device)
    y_ref_torch = torch.tensor(y_ref, dtype=torch.float).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(WNN_model.parameters(), 1e-2) # , weight_decay=1e-8, amsgrad=True

    im=1
    loss_all = np.zeros((N + 1, 1))

    [y_pre_torch,_] = WNN_model(u_torch)
    loss = criterion(y_pre_torch, y_ref_torch)
    loss_all[0:1, :] = loss.item()
    loss_m = loss.item()
    WNN_model_m = WNN(in_s, state_s, out_s, non_layer_s, non_neuron, A_ini, B_ini, dt, 'cpu', bias_status)
    WNN_model_m.load_state_dict(WNN_model.state_dict())

    start = datetime.datetime.now()
    for i in range(N):
        WNN_model.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        [y_pre_torch,_] = WNN_model(u_torch)
        loss = criterion(y_pre_torch, y_ref_torch)
        i1 = i + 1
        loss_all[i1:i1 + 1, :] = loss.item()

        if loss.item() < loss_m:
            loss_m = loss.item()
            WNN_model_m.load_state_dict(WNN_model.state_dict())
            im=i1

        if i1 % 10 == 0 or i == 0:
            print(f'Iteration: {i1}/{N}({i1/N*100:.2f}%), loss: ' + '{:.6e}'.format(loss.item()))
            end = datetime.datetime.now()
            cost_time = (end - start).total_seconds()
            cost_time_str = time_str(cost_time)
            per_time = cost_time / i1
            
            print('Average time per training: '+'{:.4f}'.format(per_time)+'s, Cumulative training time: '+cost_time_str)
            left_time = (N - i1) * per_time
            left_time_str = time_str(left_time)
            print('Executed at ' + strftime('%Y-%m-%d %H:%M:%S', end.timetuple()) +
                  ', left time: ' + left_time_str + '\n')
        
    end = datetime.datetime.now()
    cost_time = (end - start).total_seconds()
    cost_time_str = time_str(cost_time)
    print('Total training time: ' + cost_time_str + ', final loss: ' + '{:.6e}'.format(loss.item()))

    val, idx = min((val, idx) for (idx, val) in enumerate(loss_all))
    print('Minimal loss: ' + '{:.6e}'.format(val.item())+ ' (iteration: ' + str(idx) +')'+ '\n')
    return WNN_model_m, loss_all, loss_m, cost_time, im, cost_time_str