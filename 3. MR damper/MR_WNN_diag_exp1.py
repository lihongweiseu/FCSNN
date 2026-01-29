# %%
exec(open('MR_exec.py').read())
from WNN_diag_exp1 import WNN, WNN_training
os.chdir(current_path)
sys.path.append('.')

# Define the configuration of WNN
in_s = 1
out_s = 1
non_layer_s = 1

A_diag0 = torch.load('A_diag0_8.pt')
A_diag0 = torch.cat([A_diag0[:6], A_diag0[7:]])
# %%
N = 10000 # training num
All_start = time.time()
for i in range(7, 8):
    # i = 1
    state_s=i
    non_neuron = state_s * np.ones(non_layer_s, dtype=np.int32) + in_s
    WNN_model = WNN(in_s, state_s, out_s, non_layer_s, non_neuron, A_diag0, dt, device, bias_status)
    # for parameters in WNN_model.parameters():
    #     print(parameters,'\n')
    trainable_num = sum(p.numel() for p in WNN_model.parameters() if p.requires_grad)
    print('State size: ',i,', total number of trainable parameters: ', trainable_num,'\n')

    WNN_model_m, loss_all, loss_m, cost_time, im, cost_time_str = WNN_training(dt, u[0:Nt1,:,:], WNN_model, N, y_ref[0:Nt1,:,:], in_s, state_s, out_s, non_layer_s, non_neuron, A_diag0, device, bias_status)
    torch.save(WNN_model_m.state_dict(), './saved_models/'+name+'_WNN_diag_exp1_'+str(state_s)+'.pt')

All_end = time.time()
All_cost_time = All_end - All_start
All_cost_time_str = time_str(All_cost_time)
print('Total training time for all cases: ' + All_cost_time_str)
# %%
state_s=7
non_neuron = state_s * np.ones(non_layer_s, dtype=np.int32) + in_s
WNN_model = WNN(in_s, state_s, out_s, non_layer_s, non_neuron, A_diag0, dt, 'cpu', bias_status)
pt_match = glob.glob(os.path.join('./saved_models', '*_WNN_diag_exp1_'+str(state_s)+'.pt'))
WNN_model.load_state_dict(torch.load(pt_match[0], map_location='cpu'))
Para_diag=[]
Para_diag.append(A_diag0)
for parameters in WNN_model.parameters():
        # print(parameters,'\n')
        Para_diag.append(parameters)
torch.save(Para_diag, 'Para_diag_7.pt')

u_torch = torch.tensor(u, dtype=torch.float)
Nt=Nt2
[y_pre_torch,x] = WNN_model(u_torch[0:Nt,:,:])
y_ref = y_ref.reshape([Nt2, 1])
y_pre = y_pre_torch.reshape([Nt, 1]).reshape([Nt2, 1]).detach().numpy()

Pearson_corr = np.zeros(3)
Pearson_corr[0] = np.corrcoef(y_ref[0:Nt1].reshape(-1), y_pre[0:Nt1].reshape(-1))[0, 1]
Pearson_corr[1] = np.corrcoef(y_ref[Nt1:Nt2].reshape(-1), y_pre[Nt1:Nt2].reshape(-1))[0, 1]
Pearson_corr[2] = np.corrcoef(y_ref.reshape(-1), y_pre.reshape(-1))[0, 1]

plt.plot(t[0:Nt,:], y_ref)
plt.plot(t[0:Nt,:], y_pre)
plt.show()