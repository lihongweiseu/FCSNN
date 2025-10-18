# %%
exec(open('NS_exec.py').read())
from FCSNN_tools import FCSNN, FCSNN_training
os.chdir(current_path)
sys.path.append('.')

indices = list(range(0, N1))
u_train = np.take(u, indices, axis=1)
y_ref_train = np.take(y_ref, indices, axis=1)

# Define the configuration of FCSNN
in_s = 1
out_s = 1
non_layer_s = 1
# %%
N = 20000 # training num
All_start = time.time()
for i in range(1,21):
    state_s=i
    A_ini = -0.01 * torch.rand(state_s, state_s)
    B_ini = torch.rand(in_s, state_s)
    non_neuron = state_s * np.ones(non_layer_s, dtype=np.int32) + in_s
    FCSNN_model = FCSNN(in_s, state_s, out_s, non_layer_s, non_neuron, A_ini, B_ini, dt, device, bias_status)
    # for parameters in FCSNN_model.parameters():
    #     print(parameters,'\n')
    trainable_num = sum(p.numel() for p in FCSNN_model.parameters() if p.requires_grad)
    print('State size: ',i,', total number of trainable parameters: ', trainable_num,'\n')

    FCSNN_model_m, loss_all, loss_m, cost_time, im, cost_time_str = FCSNN_training(dt, u_train, FCSNN_model, N, y_ref_train, in_s, state_s, out_s, non_layer_s, non_neuron, A_ini, B_ini, device, bias_status)
    torch.save(FCSNN_model_m.state_dict(), './saved_models/'+name+'_FCSNN_'+str(state_s)+'.pt')

All_end = time.time()
All_cost_time = All_end - All_start
All_cost_time_str = time_str(All_cost_time)
print('Total training time for all cases: ' + All_cost_time_str)
# %%
state_s=6
A_ini = -0.01 * torch.ones(state_s, state_s)
B_ini = torch.rand(in_s, state_s)
non_neuron = state_s * np.ones(non_layer_s, dtype=np.int32) + in_s
FCSNN_model = FCSNN(in_s, state_s, out_s, non_layer_s, non_neuron, A_ini, B_ini, dt, 'cpu', bias_status)
pt_match = glob.glob(os.path.join('./saved_models', '*_FCSNN_'+str(state_s)+'.pt'))
FCSNN_model.load_state_dict(torch.load(pt_match[0], map_location='cpu'))
u_torch = torch.tensor(u, dtype=torch.float)
[y_pre_torch,x] = FCSNN_model(u_torch)
y_pre = y_pre_torch.detach().numpy()

for i in range(29,30):
    plt.plot(t, y_ref[:,i,:].reshape([Nt, 1]))
    plt.plot(t, y_pre[:,i,:].reshape([Nt, 1]))
    plt.show()