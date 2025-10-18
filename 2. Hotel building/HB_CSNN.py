# %%
exec(open('HB_exec.py').read())
from CSNN_tools import CSNN_create, CSNN_cal, CSNN_training
os.chdir(current_path)
sys.path.append('.')

# Define the configuration of CSNN
in_s = 1
out_s = 1
state_non_layer_s=1
out_non_layer_s=1
state_non_neuron = 3*np.ones(1, dtype=np.int32)  # size of each nonlinear layer
out_non_neuron = np.ones(1, dtype=np.int32)  # size of each nonlinear layer
# %%
state_s=3
CSNN_model = CSNN_create(in_s, state_s, out_s, state_non_layer_s, out_non_layer_s, state_non_neuron, out_non_neuron, 'cpu', bias_status)
pt_match = glob.glob(os.path.join('./saved_models', '*_CSNN_'+str(state_s)+'.pt'))
CSNN_model.load_state_dict(torch.load(pt_match[0], map_location='cpu'))

y_pre_torch = CSNN_cal(N2, Nt, dt, u, CSNN_model, state_s, out_s, 'cpu')
y_pre = y_pre_torch.detach().numpy()

for i in range(0,N2):
    plt.plot(t, y_ref[:,i,:].reshape([Nt, 1]))
    plt.plot(t, y_pre[:,i,:].reshape([Nt, 1]))
    plt.show()