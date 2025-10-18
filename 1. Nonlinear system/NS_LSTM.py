# %%
exec(open('NS_exec.py').read())
from LSTM_tools import LSTM_create, LSTM_training
os.chdir(current_path)
sys.path.append('.')

indices = list(range(0, N1))
u_train = np.take(u, indices, axis=1)
y_ref_train = np.take(y_ref, indices, axis=1)

# Define the configuration of LSTM
in_s = 1
out_s = 1
layer_s=1
# %%
N = 20000 # training num
All_start = time.time()
for i in range(1,21):
    hidden_s=i
    LSTM_model = LSTM_create(in_s, layer_s, hidden_s, device, bias_status)
    # for parameters in LSTM_model.parameters():
    #     print(parameters,'\n')
    trainable_num = sum(p.numel() for p in LSTM_model.parameters() if p.requires_grad)
    print('State size: ',i,', total number of trainable parameters: ', trainable_num,'\n')

    LSTM_model_m, loss_all, loss_m, cost_time, im, cost_time_str = LSTM_training(u_train, LSTM_model, in_s, layer_s, hidden_s, N, y_ref_train, device, bias_status)
    torch.save(LSTM_model_m.state_dict(), './saved_models/'+name+'_LSTM_'+str(hidden_s)+'.pt')

All_end = time.time()
All_cost_time = All_end - All_start
All_cost_time_str = time_str(All_cost_time)
print('Total training time for all cases: ' + All_cost_time_str)
# %%
hidden_s=20
LSTM_model = LSTM_create(in_s, layer_s, hidden_s, 'cpu', bias_status)
pt_match = glob.glob(os.path.join('./saved_models', '*_LSTM_'+str(hidden_s)+'.pt'))
LSTM_model.load_state_dict(torch.load(pt_match[0], map_location='cpu'))

criterion = nn.MSELoss()
u_torch = torch.tensor(u, dtype=torch.float).to(device)
y_pre_torch = LSTM_model(u_torch)
y_ref_torch = torch.tensor(y_ref, dtype=torch.float)
loss = criterion(y_pre_torch, y_ref_torch).item()
y_pre = y_pre_torch.detach().numpy()

for i in range(29,30):
    plt.plot(t, y_ref[:,i,:].reshape([Nt, 1]))
    plt.plot(t, y_pre[:,i,:].reshape([Nt, 1]))
    plt.show()