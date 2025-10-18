# %%
exec(open('MR_exec.py').read())
from CSNN_tools import CSNN_create, CSNN_cal, CSNN_training
os.chdir(current_path)
sys.path.append('.')

# Define the configuration of CSNN
in_s = 1
out_s = 1
state_s=8
state_non_layer_s=1
out_non_layer_s=1
state_non_neuron = 8*np.ones(1, dtype=np.int32)  # size of each nonlinear layer
out_non_neuron = np.ones(1, dtype=np.int32)  # size of each nonlinear layer
# %%
state_s=8
CSNN_model = CSNN_create(in_s, state_s, out_s, state_non_layer_s, out_non_layer_s, state_non_neuron, out_non_neuron, 'cpu', bias_status)
pt_match = glob.glob(os.path.join('./saved_models', '*_CSNN_'+str(state_s)+'.pt'))
CSNN_model.load_state_dict(torch.load(pt_match[0], map_location='cpu'))

criterion = nn.MSELoss()
y_pre_torch = CSNN_cal(1, Nt2, dt, u, CSNN_model, state_s, out_s, 'cpu')
y_ref_torch = torch.tensor(y_ref, dtype=torch.float)
loss = criterion(y_pre_torch[0:Nt1,:,:], y_ref_torch[0:Nt1,:,:]).item()
y_pre = y_pre_torch.detach().numpy()

plt.plot(t, y_ref.reshape([Nt2, 1]))
plt.plot(t, y_pre.reshape([Nt2, 1]))
plt.show()