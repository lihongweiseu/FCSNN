# %%
exec(open('NS_exec.py').read())
from WNN_tools import WNN, WNN_training
from CSNN_tools import CSNN_create, CSNN_cal, CSNN_training
from LSTM_tools import LSTM_create, LSTM_training
os.chdir(current_path)
sys.path.append('.')

Nt1= np.arange(101, 1001.1, 100).astype(int).tolist()
u_train=[]
y_ref_train=[]
for i in range(0, 10):
    u_train.append(u[:Nt1[i], 0:N1, :])
    y_ref_train.append(y_ref[:Nt1[i], 0:N1, :])
in_s = 1
out_s = 1
N = 100 # training num

para_num = np.zeros((3, 20))
train_time1 = np.zeros((3, 20))
train_time2 = np.zeros((3, 10))
# %% 
# WNN
non_layer_s = 1
state_non_layer_s=1
out_non_layer_s=1
layer_s=1

for i in range(1,21):
    A_ini = -0.01 * torch.rand(i, i)
    B_ini = torch.rand(in_s, i)
    non_neuron = i * np.ones(non_layer_s, dtype=np.int32) + in_s
    WNN_model = WNN(in_s, i, out_s, non_layer_s, non_neuron, A_ini, B_ini, dt, device, bias_status)
    para_num[0,i-1] = sum(p.numel() for p in WNN_model.parameters() if p.requires_grad)
    print('State size: ',i,', total number of trainable parameters: ', para_num[0,i-1],'\n')
    _, _, _, train_time1[0,i-1], _, _ = WNN_training(dt, u_train[9], WNN_model, N, y_ref_train[9], in_s, i, out_s, non_layer_s, non_neuron, A_ini, B_ini, device, bias_status)

    state_non_neuron = i * np.ones(1, dtype=np.int32) + in_s  # size of each nonlinear layer
    out_non_neuron = np.ones(1, dtype=np.int32)  # size of each nonlinear layer
    CSNN_model = CSNN_create(in_s, i, out_s, state_non_layer_s, out_non_layer_s, state_non_neuron, out_non_neuron, device, bias_status)
    para_num[1,i-1] = sum(p.numel() for p in CSNN_model.parameters() if p.requires_grad)
    print('State size: ',i,', total number of trainable parameters: ', para_num[1,i-1],'\n')
    _, _, _, train_time1[1,i-1], _, _ = CSNN_training(N1, Nt, dt, u_train[9], CSNN_model, N, y_ref_train[9], in_s, i, out_s, state_non_layer_s, out_non_layer_s, state_non_neuron, out_non_neuron, device, bias_status)

    LSTM_model = LSTM_create(in_s, layer_s, i, device, bias_status)
    para_num[2,i-1] = sum(p.numel() for p in LSTM_model.parameters() if p.requires_grad)
    print('State size: ',i,', total number of trainable parameters: ', para_num[2,i-1],'\n')
    _, _, _, train_time1[2,i-1], _, _ = LSTM_training(u_train[9], LSTM_model, in_s, layer_s, i, N, y_ref_train[9], device, bias_status)

for i in range(0,10):
    A_ini = -0.01 * torch.rand(4, 4)
    B_ini = torch.rand(in_s, 4)
    non_neuron = 4 * np.ones(non_layer_s, dtype=np.int32) + in_s
    WNN_model = WNN(in_s, 4, out_s, non_layer_s, non_neuron, A_ini, B_ini, dt, device, bias_status)
    _, _, _, train_time2[0,i], _, _ = WNN_training(dt, u_train[i], WNN_model, N, y_ref_train[i], in_s, 4, out_s, non_layer_s, non_neuron, A_ini, B_ini, device, bias_status)

    state_non_neuron = 4 * np.ones(1, dtype=np.int32) + in_s  # size of each nonlinear layer
    out_non_neuron = np.ones(1, dtype=np.int32)  # size of each nonlinear layer
    CSNN_model = CSNN_create(in_s, 4, out_s, state_non_layer_s, out_non_layer_s, state_non_neuron, out_non_neuron, device, bias_status)
    _, _, _, train_time2[1,i], _, _ = CSNN_training(N1, Nt1[i], dt, u_train[i], CSNN_model, N, y_ref_train[i], in_s, 4, out_s, state_non_layer_s, out_non_layer_s, state_non_neuron, out_non_neuron, device, bias_status)

    LSTM_model = LSTM_create(in_s, layer_s, 4, device, bias_status)
    _, _, _, train_time2[2,i], _, _ = LSTM_training(u_train[i], LSTM_model, in_s, layer_s, 4, N, y_ref_train[i], device, bias_status)

np.savez('Train_time.npz', para_num=para_num, train_time1=train_time1, train_time2=train_time2)
# %%
loaded_data = np.load('Train_time.npz')
para_num = loaded_data['para_num']
train_time1 = loaded_data['train_time1']
train_time2 = loaded_data['train_time2']

params_sw = 1
if params_sw==1:
    params = {'text.usetex': True, 'font.family': 'serif', 'font.serif': 'Times',
          'text.latex.preamble': ''.join([r'\usepackage{fontenc}'
                                          r'\usepackage{newtxmath,amsmath,newtxtext}'
                                        #   r'\usepackage[not1,notextcomp,lcgreekalpha]{stix}'
                                          ])
          }
    plt.rcParams.update(params)
else:
    import matplotlib as mpl
    mpl.use('pgf')
    mpl.rcParams.update({
    'text.usetex': True,  # use default xelatex
    'pgf.texsystem': 'xelatex',
    'font.family': 'sans-serif',
    'font.sans-serif': 'Times New Roman',
    'pgf.preamble': ''.join([
        r'\usepackage{xeCJK}',  # load CJK package
        r'\usepackage{metalogo}',
        # r'\usepackage{unicode-math}',  # unicode math setup
        r'\usepackage{fontspec}',
        r'\usepackage{newtxmath,amsmath}',
        r'\usepackage[not1,notextcomp,lcgreekalpha]{stix}',
        # r'\setmainfont[Mapping=tex-text]{Times New Roman}',
        r'\setCJKmainfont{SimSun}',  # serif font via preamble SimSun SimHei
        r'\makeatletter \newcommand*{\rom}[1]{\expandafter\@slowromancap\romannumeral #1@} \makeatother', ])})
    plt.rcParams['text.latex.preamble'] = r'\makeatletter \newcommand*{\rom}[1]{\expandafter\@slowromancap\romannumeral #1@} \makeatother'

cm = 1 / 2.54
model_name1=['WNN (1-20)','CSNN (1-20)','LSTM (1-18)']
model_name2=['WNN (5)','CSNN (5)','LSTM (5)']
colors = ['r', 'b', 'g']
markers = ['v','^','o']
# %%
NN=[20,20,18]
xl=[[2,14],[5,51]]
yl=[[0.2,1.2],[0.2,1.2]]
fig1 = plt.figure(figsize=(8.8 * cm, 5 * cm))
ax = fig1.add_subplot(111)
for i in range(3):
    ax.plot(para_num[i, 0:NN[i]]/100, train_time1[i, 0:NN[i]]/100, c=colors[i], lw=0.75, label=model_name1[i], marker=markers[i], markersize=4)
ax.set_xlabel(r'Number of parameters ($\times100$)', fontsize=7, labelpad=1)
ax.set_ylabel(r'Training time per epoch (s)', fontsize=7, labelpad=1)
i=0
ax.set_ylim([0, yl[i][1]])
ax.set_yticks(np.arange(0, yl[i][1]+0.01, yl[i][0]))
ax.set_xlim([0, xl[i][1]])
ax.set_xticks(np.arange(0, xl[i][1]+0.01, xl[i][0]))
labels = [item.get_text() for item in ax.get_xticklabels()]
labels_new = [i.replace('$\\mathdefault{', '').replace('}$', '') for i in labels]
ax.set_xticklabels(labels_new)
labels = [item.get_text() for item in ax.get_yticklabels()]
labels_new = [i.replace('$\\mathdefault{', '').replace('}$', '').replace('.0', '') for i in labels]
ax.set_yticklabels(labels_new)

legend = ax.legend(loc='upper right', bbox_to_anchor=(1, 1), borderpad=0.3, borderaxespad=2, handlelength=2,
                edgecolor='black', fontsize=7, ncol=1, columnspacing=0.4, handletextpad=0.2)  # labelspacing=0
legend.get_frame().set_boxstyle('Square', pad=0.0)
legend.get_frame().set_lw(0.75)
legend.get_frame().set_alpha(None)

ax.tick_params(axis='both', direction='in', width=0.75, labelsize=7, length=2.5)
ax.grid(lw=0.5)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_linewidth(0.75)

fig1.tight_layout(pad=0)
# fig1.savefig(fname = 'F_train_time1.pdf', format = 'pdf')
# %%
fig2 = plt.figure(figsize=(8.8 * cm, 5 * cm))
ax = fig2.add_subplot(111)
for i in range(3):
    ax.plot((np.arange(101, 1001.1, 100)-1)*dt, train_time2[i, :]/100, c=colors[i], lw=0.75, label=model_name2[i], marker=markers[i], markersize=4)
ax.set_xlabel(r'Training dataset length (s)', fontsize=7, labelpad=1)
ax.set_ylabel(r'Training time per epoch (s)', fontsize=7, labelpad=1)
i=1
ax.set_ylim([-0.05, yl[i][1]])
ax.set_yticks(np.arange(0, yl[i][1]+0.01, yl[i][0]))
ax.set_xlim([3.9, xl[i][1]])
ax.set_xticks(np.arange(5, xl[i][1]+0.01, xl[i][0]))
labels = [item.get_text() for item in ax.get_xticklabels()]
labels_new = [i.replace('$\\mathdefault{', '').replace('}$', '') for i in labels]
ax.set_xticklabels(labels_new)
labels = [item.get_text() for item in ax.get_yticklabels()]
labels_new = [i.replace('$\\mathdefault{', '').replace('}$', '').replace('.0', '') for i in labels]
ax.set_yticklabels(labels_new)

legend = ax.legend(loc='upper left', bbox_to_anchor=(0, 1), borderpad=0.3, borderaxespad=2, handlelength=2,
                edgecolor='black', fontsize=7, ncol=1, columnspacing=0.4, handletextpad=0.2)  # labelspacing=0
legend.get_frame().set_boxstyle('Square', pad=0.0)
legend.get_frame().set_lw(0.75)
legend.get_frame().set_alpha(None)

ax.tick_params(axis='both', direction='in', width=0.75, labelsize=7, length=2.5)
ax.grid(lw=0.5)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_linewidth(0.75)

fig2.tight_layout(pad=0, rect=(0, 0, 0.998, 1))
# fig2.savefig(fname = 'F_train_time2.pdf', format = 'pdf')