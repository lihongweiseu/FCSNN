# %%
exec(open('MR_exec.py').read())
from WNN_tools import WNN, WNN_training
os.chdir(current_path)
sys.path.append('.')

# Define the configuration of WNN
in_s = 1
out_s = 1
non_layer_s = 1
# %%
N = 20000 # training num
All_start = time.time()
for i in range(1,21):
    # i = 1
    state_s=i
    A_ini = -0.01 * torch.rand(state_s, state_s)
    B_ini = torch.rand(in_s, state_s)
    non_neuron = state_s * np.ones(non_layer_s, dtype=np.int32) + in_s
    WNN_model = WNN(in_s, state_s, out_s, non_layer_s, non_neuron, A_ini, B_ini, dt, device, bias_status)
    # for parameters in WNN_model.parameters():
    #     print(parameters,'\n')
    trainable_num = sum(p.numel() for p in WNN_model.parameters() if p.requires_grad)
    print('State size: ',i,', total number of trainable parameters: ', trainable_num,'\n')

    WNN_model_m, loss_all, loss_m, cost_time, im, cost_time_str = WNN_training(dt, u[0:Nt1,:,:], WNN_model, N, y_ref[0:Nt1,:,:], in_s, state_s, out_s, non_layer_s, non_neuron, A_ini, B_ini, device, bias_status)
    torch.save(WNN_model_m.state_dict(), './saved_models/'+name+'_WNN_'+str(state_s)+'.pt')

All_end = time.time()
All_cost_time = All_end - All_start
All_cost_time_str = time_str(All_cost_time)
print('Total training time for all cases: ' + All_cost_time_str)
# %%
state_s=8
A_ini = -0.01 * torch.rand(state_s, state_s)
B_ini = torch.rand(in_s, state_s)
non_neuron = state_s * np.ones(non_layer_s, dtype=np.int32) + in_s
WNN_model = WNN(in_s, state_s, out_s, non_layer_s, non_neuron, A_ini, B_ini, dt, 'cpu', bias_status)
pt_match = glob.glob(os.path.join('./saved_models', '*_WNN_'+str(state_s)+'.pt'))
WNN_model.load_state_dict(torch.load(pt_match[0], map_location='cpu'))
u_torch = torch.tensor(u, dtype=torch.float)
Nt=Nt2
[y_pre_torch,x] = WNN_model(u_torch[0:Nt,:,:])
y_pre = y_pre_torch.reshape([Nt, 1]).detach().numpy()

plt.plot(t[0:Nt,:], y_ref[0:Nt,:,:].reshape([Nt, 1]))
plt.plot(t[0:Nt,:], y_pre)
plt.show()