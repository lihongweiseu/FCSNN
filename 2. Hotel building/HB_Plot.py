# %%
exec(open('HB_exec.py').read())
from FCSNN_tools import FCSNN
from CSNN_tools import CSNN_create, CSNN_cal
from LSTM_tools import LSTM_create
os.chdir(current_path)
sys.path.append('.')

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

y_ref=y_ref.reshape([Nt, N2])
y_pre = np.zeros((4, Nt, N2))
in_s = 1
out_s = 1

# FCSNN
non_layer_s = 1
state_s=8
A_ini = -0.01 * torch.ones(state_s, state_s)
B_ini = torch.rand(in_s, state_s)
non_neuron = state_s * np.ones(non_layer_s, dtype=np.int32) + in_s
FCSNN_model = FCSNN(in_s, state_s, out_s, non_layer_s, non_neuron, A_ini, B_ini, dt, 'cpu', bias_status)
pt_match = glob.glob(os.path.join('./saved_models', '*_FCSNN_'+str(state_s)+'.pt'))
FCSNN_model.load_state_dict(torch.load(pt_match[0], map_location='cpu'))
u_torch = torch.tensor(u, dtype=torch.float)
[y_FCSNN_torch,x] = FCSNN_model(u_torch)
y_pre[0,:,:] = y_FCSNN_torch.detach().reshape([Nt, N2]).numpy()

# CSNN
state_non_layer_s=1
out_non_layer_s=1
state_non_neuron = 3*np.ones(1, dtype=np.int32)  # size of each nonlinear layer
out_non_neuron = np.ones(1, dtype=np.int32)  # size of each nonlinear layer
state_s=3
CSNN_model = CSNN_create(in_s, state_s, out_s, state_non_layer_s, out_non_layer_s, state_non_neuron, out_non_neuron, 'cpu', bias_status)
pt_match = glob.glob(os.path.join('./saved_models', '*_CSNN_'+str(state_s)+'.pt'))
CSNN_model.load_state_dict(torch.load(pt_match[0], map_location='cpu'))
y_CSNN_torch = CSNN_cal(N2, Nt, dt, u, CSNN_model, state_s, out_s, 'cpu')
y_pre[1,:,:] = y_CSNN_torch.detach().reshape([Nt, N2]).numpy()

# LSTM
layer_s=1
hidden_s=18
LSTM_model = LSTM_create(in_s, layer_s, hidden_s, 'cpu', bias_status)
pt_match = glob.glob(os.path.join('./saved_models', '*_LSTM_'+str(hidden_s)+'.pt'))
LSTM_model.load_state_dict(torch.load(pt_match[0], map_location='cpu'))
u_torch = torch.tensor(u, dtype=torch.float).to(device)
y_LSTM_torch = LSTM_model(u_torch)
y_pre[2,:,:] = y_LSTM_torch.detach().reshape([Nt, N2]).numpy()

# PhyCNN
f = loadmat(name+'_y_PhyCNN.mat')
y_pre[3,:,:] = f['y_PhyCNN']

PCC = np.zeros((N2, 4))
for i in range(N2):
    for j in range(4):
        PCC[i, j] = np.corrcoef(y_ref[:, i], y_pre[j, :, i])[0, 1]

PCC_aver = np.zeros((3, 4))
PCC_aver[0, :] = np.mean(PCC[0:N1, :], axis=0)
PCC_aver[1, :] = np.mean(PCC[N1:N2, :], axis=0)
PCC_aver[2, :] = np.mean(PCC[:, :], axis=0)
model_name1=['FCSNN','CSNN','LSTM','PhyCNN']
for i in range(4):
    val1, idx1 = min((val, idx) for (idx, val) in enumerate(PCC[0:N1, i]))
    val2, idx2 = min((val, idx) for (idx, val) in enumerate(PCC[N1:N2, i]))
    print(model_name1[i]+' & '+'{:.5f}'.format(PCC_aver[0, i].item())+' & '+'{:.5f}'.format(val1.item())+' (case '+str(idx1+1)+')'
          +' & '+'{:.5f}'.format(PCC_aver[1, i].item())+' & '+'{:.5f}'.format(val2.item())+' (case '+str(idx2+N1+1)+')'+r' \\')

cm = 1 / 2.54
model_name=['FCSNN (8)','CSNN (3)','LSTM (18)','PhyCNN']
# colors = [(0.75,1,0,0.4),(0.75,0.5,0.25,0.4),(0.5,0.5,0.5,0.4),(0,0,1,0.4)]
colors = ['r', 'b', 'g', 'c']
markers = ['v','^','o', 's']
N = np.linspace(1, N2, N2).reshape(-1, 1)
# %%
fig1 = plt.figure(figsize=(14 * cm, 5 * cm))
ax = fig1.add_subplot(111)
ax.plot([N1+0.5, N1+0.5], [0.75, 1.008], 'k', lw=0.5)
for i in range(4):
    ax.scatter(N, PCC[:,i], c=colors[i], marker=markers[i], s=25, alpha=0.8, linewidths=0, label=model_name[i]) #
ax.set_ylim([0.75, 1.008])
ax.set_yticks(np.arange(0.75, 1.01, 0.05))
ax.set_ylabel(r'PCC', fontsize=7, labelpad=1)
ax.set_xlim([0, 22])
ax.set_xticks(np.arange(0, 22.1, 2.0))
ax.set_xlabel(r'Case no.', fontsize=7, labelpad=1)
ax.text(7, 0.775, 'Training', ha='center', va='center', fontsize=7)
ax.text(17, 0.775, 'Testing', ha='center', va='center', fontsize=7)
labels = [item.get_text() for item in ax.get_xticklabels()]
labels_new = [i.replace('$\\mathdefault{', '').replace('}$', '') for i in labels]
ax.set_xticklabels(labels_new)
labels = [item.get_text() for item in ax.get_yticklabels()]
labels_new = [i.replace('$\\mathdefault{', '').replace('}$', '').replace('.00', '').replace('.90', '.9').replace('.80', '.8') for i in labels]
ax.set_yticklabels(labels_new)
legend = ax.legend(loc='lower left', bbox_to_anchor=(0, 0), borderpad=0.3, borderaxespad=1, handlelength=1.8,
                   edgecolor='black', fontsize=7, ncol=1, columnspacing=0.2, handletextpad=0.0, markerscale=1)  # labelspacing=0
legend.get_frame().set_boxstyle('Square', pad=0.0)
legend.get_frame().set_lw(0.75)
legend.get_frame().set_alpha(None)
for obj in legend.legend_handles:
    obj.set_lw(0)
ax.tick_params(axis='both', direction='in', width=0.75, labelsize=7, length=2.5)
ax.grid(lw=0.5)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_linewidth(0.75)
fig1.tight_layout(pad=0, rect=(0, 0, 1, 0.995))

# fig1.savefig(fname = 'F_HB_PCC.pdf', format = 'pdf')
#%%
#val, idx = min((val, idx) for (idx, val) in enumerate(PCC[:,0]))
j=[5,19]
yl= [[40, 20],[16, 8]]
xl = [[22, 38, 2],[22, 38, 2]]
fig2 = plt.figure(figsize=(14 * cm, 7 * cm))
for i in range(2):
    ax = fig2.add_subplot(2,1,i+1)
    ax.plot(t, y_ref[:,j[i]-1:j[i]], color='k', lw=0.5, label='Measurement')
    ax.plot(t, y_pre[0,:,j[i]-1:j[i]], color='r', dashes=[8, 2, 4, 2], lw=0.5, label='FCSNN (8)')
    ax.set_ylim([-yl[i][0], yl[i][0]])
    ax.set_yticks(np.arange(-yl[i][0], yl[i][0]+0.1, yl[i][1]))
    ax.set_xlim([xl[i][0], xl[i][1]])
    ax.set_xticks(np.arange(xl[i][0], xl[i][1]+0.1, xl[i][2]))
    ax.text(30, yl[i][1]*1.5, 'Case '+str(j[i])+', PCC: '+ '{:.5f}'.format(PCC[j[i]-1,0].item()), ha='center', va='center', fontsize=7)
    ax.set_ylabel(r'Acc. ($\rm cm^2$)', fontsize=7, labelpad=-1)
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels_new = [i.replace('$\\mathdefault{', '').replace('}$', '') for i in labels]
    ax.set_xticklabels(labels_new)
    labels = [item.get_text() for item in ax.get_yticklabels()]
    labels_new = [i.replace('$\\mathdefault{', '').replace('}$', '') for i in labels]
    ax.set_yticklabels(labels_new)

    legend = ax.legend(loc='lower right', bbox_to_anchor=(1, 0), borderpad=0.3, borderaxespad=1, handlelength=1.8,
                   edgecolor='black', fontsize=7, ncol=2, columnspacing=0.4, handletextpad=0.2)  # labelspacing=0
    legend.get_frame().set_boxstyle('Square', pad=0.0)
    legend.get_frame().set_lw(0.75)
    legend.get_frame().set_alpha(None)

    ax.tick_params(axis='both', direction='in', width=0.75, labelsize=7, length=2.5)
    ax.grid(lw=0.5)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)

ax.set_xlabel(r'Time (s)', fontsize=7, labelpad=1)

fig2.tight_layout(pad=0)
plt.subplots_adjust(hspace=0.18, wspace=0.1)
# fig2.savefig(fname = 'F_HB_pre.pdf', format = 'pdf')