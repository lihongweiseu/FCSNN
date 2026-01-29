# %%
exec(open('MR_exec.py').read())
from WNN_diag_exp2 import WNN, WNN_training
os.chdir(current_path)
sys.path.append('.')

# Define the configuration of WNN
in_s = 1
out_s = 1
non_layer_s = 1
Para_diag = torch.load('Para_diag_7.pt')
# %%
N = 10000 # training num
All_start = time.time()
for i in range(7, 8):
    # i = 1
    state_s=i
    non_neuron = state_s * np.ones(non_layer_s, dtype=np.int32) + in_s
    WNN_model = WNN(in_s, state_s, out_s, non_layer_s, non_neuron, Para_diag, dt, device, bias_status)
    # for parameters in WNN_model.parameters():
    #     print(parameters,'\n')
    trainable_num = sum(p.numel() for p in WNN_model.parameters() if p.requires_grad)
    print('State size: ',i,', total number of trainable parameters: ', trainable_num,'\n')

    WNN_model_m, loss_all, loss_m, cost_time, im, cost_time_str = WNN_training(dt, u[0:Nt2,:,:], WNN_model, N, y_ref[0:Nt2,:,:], in_s, state_s, out_s, non_layer_s, non_neuron, Para_diag, device, bias_status)
    torch.save(WNN_model_m.state_dict(), './saved_models/'+name+'_WNN_diag_exp2_'+str(state_s)+'.pt')

All_end = time.time()
All_cost_time = All_end - All_start
All_cost_time_str = time_str(All_cost_time)
print('Total training time for all cases: ' + All_cost_time_str)
# %%
state_s=7
A_ini = -0.01 * torch.rand(state_s, state_s)
B_ini = torch.rand(in_s, state_s)
non_neuron = state_s * np.ones(non_layer_s, dtype=np.int32) + in_s
WNN_model = WNN(in_s, state_s, out_s, non_layer_s, non_neuron, Para_diag, dt, 'cpu', bias_status)
pt_match = glob.glob(os.path.join('./saved_models', '*_WNN_diag_exp2_'+str(state_s)+'*.pt'))
# # pt_match = glob.glob(os.path.join('./saved_models', 'MR_WNN_diag_20_9.9996e1_19976_2h 7m 34.746s.pt'))
WNN_model.load_state_dict(torch.load(pt_match[0], map_location='cpu'))
Para=[]
for parameters in WNN_model.parameters():
        # print(parameters,'\n')
        Para.append(parameters)
A_diag0 = Para[0]
A_diag = torch.matmul(torch.square(A_diag0),torch.triu(torch.ones(state_s, state_s), diagonal=0))
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