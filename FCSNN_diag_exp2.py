# %%
import datetime
from time import strftime
import random
import numpy as np
import torch
from torch import nn, optim
from torch.nn.parameter import Parameter
from general_tools import time_str#, early_stop_check

# %%
# To guarantee same results for every running, which might slow down the training speed
torch.set_default_dtype(torch.float)
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)

def FCSNN(in_s, state_s, out_s, non_layer_s, non_neuron, Para, dt, device, bias_status):
    # Define FCSNN
    class FCSNN(nn.Module):
        def __init__(self):
            super(FCSNN, self).__init__()
            self.A_diag0 = Parameter(Para[0].clone().to(device))
            
            if in_s>1:
                self.B_down = Parameter(torch.ones(in_s-1, state_s).to(device))
            else:
                self.B_down = torch.tensor([], dtype=torch.float).to(device)
            
            self.lin = nn.Linear(state_s+in_s, out_s, bias=bias_status)
            with torch.no_grad():
                self.lin.weight.copy_(Para[1])
                self.lin.weight.requires_grad = True
            
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

            for i in (0,2):
                with torch.no_grad():
                    self.non1[i].weight.copy_(Para[2+int(i/2)])
                    self.non1[i].weight.requires_grad = True
                    self.non2[i].weight.copy_(Para[4+int(i/2)])
                    self.non2[i].weight.requires_grad = True

        def forward(self, u):
            Nt = u.shape[0]
            Amp = u.shape[1]
            x = torch.zeros(Nt, Amp, state_s).to(device)

            B_up = torch.ones(1, state_s).to(device)
            B = torch.cat((B_up, self.B_down), dim=0)

            A_diag = torch.matmul(-torch.square(self.A_diag0),torch.triu(torch.ones(state_s, state_s).to(device), diagonal=0))
            # A_diag = -torch.square(self.A_diag0)
            A_inverse_diag = 1/A_diag

            a_diag = torch.exp(A_diag*dt)
            temp1 = a_diag * A_inverse_diag
            temp2 = (temp1 - A_inverse_diag)/dt*A_inverse_diag
            b_diag = temp1 - temp2
            c_diag = temp2 - A_inverse_diag

            a = torch.diag(a_diag)
            b = torch.diag(b_diag)
            c = torch.diag(c_diag)

            for i in range(Nt - 1):
                u0 = u[i, :, :]
                u1 = u[i+1, :, :]
                x0 = x[i, :, :].clone()
                x[i+1, :, :] = torch.matmul(x0,a) + torch.matmul(torch.matmul(u0,B),b) + torch.matmul(torch.matmul(u1,B),c)

            u_x = torch.cat((u, x), dim=2)
            y = self.lin(u_x)+self.non1(u_x)+self.non2(u_x)
            
            return y, x

    FCSNN_model = FCSNN().to(device)
    return FCSNN_model

def FCSNN_training(dt, u, FCSNN_model, N, y_ref, in_s, state_s, out_s, non_layer_s, non_neuron, Para, device, bias_status):
    u_torch = torch.tensor(u, dtype=torch.float).to(device)
    y_ref_torch = torch.tensor(y_ref, dtype=torch.float).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(FCSNN_model.parameters(), 1e-2) # 1e-3, weight_decay=1e-8

    im=1
    loss_all = np.zeros((N + 1, 1))

    [y_pre_torch,_] = FCSNN_model(u_torch)
    loss = criterion(y_pre_torch, y_ref_torch)
    loss_all[0:1, :] = loss.item()
    loss_m = loss.item()
    FCSNN_model_m = FCSNN(in_s, state_s, out_s, non_layer_s, non_neuron, Para, dt, 'cpu', bias_status)
    FCSNN_model_m.load_state_dict(FCSNN_model.state_dict())

    start = datetime.datetime.now()
    for i in range(N):
        FCSNN_model.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        [y_pre_torch,_] = FCSNN_model(u_torch)
        loss = criterion(y_pre_torch, y_ref_torch)
        i1 = i + 1
        loss_all[i1:i1 + 1, :] = loss.item()

        if loss.item() < loss_m:
            loss_m = loss.item()
            FCSNN_model_m.load_state_dict(FCSNN_model.state_dict())
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
    return FCSNN_model_m, loss_all, loss_m, cost_time, im, cost_time_str