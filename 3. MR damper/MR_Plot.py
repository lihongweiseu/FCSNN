# %%
exec(open('MR_exec.py').read())
from WNN_tools import WNN
from CSNN_tools import CSNN_create, CSNN_cal
from LSTM_tools import LSTM_create
os.chdir(current_path)
sys.path.append('.')
import matplotlib.patches as mpatches
import matplotlib as mpl

params_sw = 1
if params_sw==1:
    params = {'text.usetex': True, 'font.family': 'serif', 'font.serif': 'Times',
          'text.latex.preamble': ''.join([r'\usepackage{fontenc}'
                                          r'\usepackage{newtxmath,amsmath,newtxtext}'
                                        #   r'\usepackage[not1,notextcomp,lcgreekalpha]{stix}'
                                          ])
          }
    plt.rcParams.update(params)
elif params_sw==2:
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
else:
    plt.rcParams['font.family'] = 'Times New Roman'

mpl.rcParams['hatch.linewidth'] = 0.5
mpl.rcParams['hatch.color'] = 'k'

y_ref=y_ref.reshape([Nt2, 1])
y_pre = np.zeros((Nt2, 4))
in_s = 1
out_s = 1

# WNN
non_layer_s = 1
state_s=8
A_ini = -0.01 * torch.ones(state_s, state_s)
B_ini = torch.rand(in_s, state_s)
non_neuron = state_s * np.ones(non_layer_s, dtype=np.int32) + in_s
WNN_model = WNN(in_s, state_s, out_s, non_layer_s, non_neuron, A_ini, B_ini, dt, 'cpu', bias_status)
pt_match = glob.glob(os.path.join('./saved_models', '*_WNN_'+str(state_s)+'.pt'))
WNN_model.load_state_dict(torch.load(pt_match[0], map_location='cpu'))
u_torch = torch.tensor(u, dtype=torch.float)
[y_WNN_torch,x] = WNN_model(u_torch)
y_pre[:,0:1] = y_WNN_torch.detach().reshape([Nt2, 1]).numpy()

# CSNN
state_non_layer_s=1
out_non_layer_s=1
state_non_neuron = 8*np.ones(1, dtype=np.int32)  # size of each nonlinear layer
out_non_neuron = np.ones(1, dtype=np.int32)  # size of each nonlinear layer
state_s=8
CSNN_model = CSNN_create(in_s, state_s, out_s, state_non_layer_s, out_non_layer_s, state_non_neuron, out_non_neuron, 'cpu', bias_status)
pt_match = glob.glob(os.path.join('./saved_models', '*_CSNN_'+str(state_s)+'.pt'))
CSNN_model.load_state_dict(torch.load(pt_match[0], map_location='cpu'))
y_CSNN_torch = CSNN_cal(1, Nt2, dt, u, CSNN_model, state_s, out_s, 'cpu')
y_pre[:,1:2] = y_CSNN_torch.detach().reshape([Nt2, 1]).numpy()

# LSTM
layer_s=1
hidden_s=20
LSTM_model = LSTM_create(in_s, layer_s, hidden_s, 'cpu', bias_status)
pt_match = glob.glob(os.path.join('./saved_models', '*_LSTM_'+str(hidden_s)+'.pt'))
LSTM_model.load_state_dict(torch.load(pt_match[0], map_location='cpu'))
u_torch = torch.tensor(u, dtype=torch.float).to(device)
y_LSTM_torch = LSTM_model(u_torch)
y_pre[:,2:3] = y_LSTM_torch.detach().reshape([Nt2, 1]).numpy()

# Narx
f = loadmat(name+'_y_Narx.mat')
y_pre[:,3:4] = f['y_Narx']
u = u.reshape([Nt2, 1])

PCC = np.zeros((4,3))
for i in range(4):
    PCC[i,0] = np.corrcoef(y_ref[0:Nt1].reshape(-1), y_pre[0:Nt1, i])[0, 1]
    PCC[i,1] = np.corrcoef(y_ref[Nt1:Nt2].reshape(-1), y_pre[Nt1:Nt2, i])[0, 1]
    PCC[i,2] = np.corrcoef(y_ref.reshape(-1), y_pre[:, i])[0, 1]

cm = 1 / 2.54
model_name=['WNN (8)','CSNN (8)','LSTM (20)','Narx']
# colors = [(0.75,1,0,0.4),(0.75,0.5,0.25,0.4),(0.5,0.5,0.5,0.4),(0,0,1,0.4)]
colors = ['r', 'b', 'g', 'c']
col = [(1,0,0,0.6),(0,0,1,0.6),(0,1,0,0.6),(0,1,1,0.6)]
markers = ['v','^','o', 's']
N = np.linspace(1, 1, 1).reshape(-1, 1)
# %%
fig1 = plt.figure(figsize=(14 * cm, 5 * cm))
patterns = ['//', '\\\\', '..', '']
patterns1 = ['////', '\\\\\\', '....', '']
width = 0.22  # the width of the bars
legend_styles = [[]]*4
x_range=np.linspace(1, 3, 3)
ax = fig1.add_subplot(111)
for i in range(4):
    rects = ax.bar(x_range+(i-1.5)*width, PCC[i,:], width, zorder = 1,
                   color=col[i], edgecolor = 'k', linewidth = 0.5, label=model_name[i])
    ax.bar(x_range+(i-1.5)*width, PCC[i,:], width, hatch=patterns1[i], zorder = 2,
                color='none', edgecolor = 'k', linewidth = 0.5)
    ax.bar_label(rects, padding=1, fmt=lambda x: f'{x :.5f}', fontsize=7) #, rotation=90
    legend_styles[i] = (rects,mpatches.Patch(facecolor='none', linewidth = 0.5, edgecolor = 'k', hatch=patterns1[i])) #, label=model_name[i]

ax.set_ylim([0.97, 1.01])
ax.set_yticks(np.arange(0.97, 1.01, 0.01))
labels = [item.get_text() for item in ax.get_yticklabels()]
labels_new = [i.replace('$\\mathdefault{', '').replace('}$', '').replace('.00', '') for i in labels]
ax.set_yticklabels(labels_new)
ax.set_ylabel(r'PCC', fontsize=7, labelpad=1)
ax.set_xlim([0.5, 3.5])
ax.set_xticks(np.arange(1, 3.1, 1.0))
ax.set_xticklabels(['Training (0-10s)', 'Testing (10-17.49s)', 'Overall (0-17.49s)'])

legend = ax.legend(legend_styles, model_name, loc='upper right', bbox_to_anchor=(1, 1), borderpad=0.3, borderaxespad=1, handlelength=4, handleheight=1.2,
                   edgecolor='black', fontsize=7, ncol=4, columnspacing=0.4, handletextpad=0.2)  # labelspacing=0
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
fig1.tight_layout(pad=0, rect=(0, 0, 0.998, 1))
# fig1.savefig(fname = 'F_MR_PCC.pdf', format = 'pdf')
#%%
fig2 = plt.figure(figsize=(14 * cm, 8 * cm))
y1 = [120, 60]
x1 = [0, 18, 4]

for j in range(4):
    ax = fig2.add_subplot(2,2,j+1)
    ax.plot([10, 10], [-y1[0], y1[0]], 'gray', lw=0.75)
    ax.plot(t, y_ref, color='k', lw=0.5, label='Exp.')
    ax.plot(t, y_pre[:,j:j+1], color=colors[j], dashes=[2, 2], lw=0.5, label=model_name[j])
    ax.text(6, -100, 'Training', ha='center', va='center', fontsize=7)
    ax.text(14, -100, 'Testing', ha='center', va='center', fontsize=7)
    ax.set_ylim([-y1[0], y1[0]])
    ax.set_yticks(np.arange(-y1[0], y1[0]+0.1, y1[1]))
    ax.set_xlim([x1[0], x1[1]])
    ax.set_xticks([0,4,8,10,12,16,18])
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels_new = [i.replace('$\\mathdefault{', '').replace('}$', '') for i in labels]
    ax.set_xticklabels(labels_new)
    labels = [item.get_text() for item in ax.get_yticklabels()]
    labels_new = [i.replace('$\\mathdefault{', '').replace('}$', '') for i in labels]
    ax.set_yticklabels(labels_new)
    ax.tick_params(axis='both', direction='in', width=0.75, labelsize=7, length=2.5)
    ax.grid(lw=0.5)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
    legend = ax.legend(loc='upper left', bbox_to_anchor=(0, 1), borderpad=0.3, borderaxespad=0, handlelength=1.8,
                   edgecolor='black', fontsize=7, ncol=2, columnspacing=0.4, handletextpad=0.2)  # labelspacing=0
    legend.get_frame().set_boxstyle('Square', pad=0.0)
    legend.get_frame().set_lw(0.75)
    legend.get_frame().set_alpha(None)

for j in range(2):
    ax = plt.subplot(2,2,2*j+1)
    ax.set_ylabel(r'Force (N)', fontsize=7, labelpad=-1)
    ax = plt.subplot(2,2,j+3)
    ax.set_xlabel(r'Time (s)', fontsize=7, labelpad=1)

fig2.tight_layout(pad=0)
plt.subplots_adjust(hspace=0.15, wspace=0.14)
# fig2.savefig(fname = 'F_MR_pre.pdf', format = 'pdf')
# %%
fig3 = plt.figure(figsize=(14.5 * cm, 14* cm))
y1 = [100, 50]
x1 = [16, 8]
for j in range(4):
    ax = fig3.add_subplot(1,4,j+1)
    ax.plot(u[Nt1:], y_ref[Nt1:], color='k', lw=0.5, label='Exp.')
    ax.plot(u[Nt1:], y_pre[Nt1:,j:j+1], color=colors[j], dashes=[2, 2], lw=0.5, label=model_name[j])
    ax.set_box_aspect(1)
    ax.set_ylim([-y1[0], y1[0]])
    ax.set_yticks(np.arange(-y1[0], y1[0]+0.1, y1[1]))
    ax.set_xlim([-x1[0], x1[0]])
    ax.set_xticks(np.arange(-x1[0], x1[0]+0.1, x1[1]))
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels_new = [i.replace('$\\mathdefault{', '').replace('}$', '') for i in labels]
    ax.set_xticklabels(labels_new)
    labels = [item.get_text() for item in ax.get_yticklabels()]
    labels_new = [i.replace('$\\mathdefault{', '').replace('}$', '') for i in labels]
    ax.set_yticklabels(labels_new)
    ax.tick_params(axis='both', direction='in', width=0.75, labelsize=7, length=2.5)
    ax.grid(lw=0.5)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
    legend = ax.legend(loc='upper right', bbox_to_anchor=(1, 1), borderpad=0.2, borderaxespad=0, handlelength=0.8,
                   edgecolor='black', fontsize=6, ncol=1, columnspacing=0.2, handletextpad=0.2)  # labelspacing=0
    legend.get_frame().set_boxstyle('Square', pad=0.0)
    legend.get_frame().set_lw(0.75)
    legend.get_frame().set_alpha(None)
    ax.set_xlabel(r'Vel (cm/s)', fontsize=7, labelpad=1)

ax = plt.subplot(1,4,1)
ax.set_ylabel(r'Force (N)', fontsize=7, labelpad=-1)

fig3.tight_layout(pad=0)
# plt.subplots_adjust()
plt.subplots_adjust(wspace=0.3)
# fig3.savefig(fname = 'F_MR_uy.pdf', format = 'pdf', bbox_inches='tight', pad_inches=0)