# %%
exec(open('MR_exec.py').read())
from FCSNN_tools import FCSNN
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
y_pre = np.zeros((Nt2, 1))
in_s = 1
out_s = 1

# FCSNN
non_layer_s = 1
state_s=20
A_ini = -0.01 * torch.ones(state_s, state_s)
B_ini = torch.rand(in_s, state_s)
non_neuron = state_s * np.ones(non_layer_s, dtype=np.int32)
FCSNN_model = FCSNN(in_s, state_s, out_s, non_layer_s, non_neuron, A_ini, B_ini, dt, 'cpu', bias_status)
pt_match = glob.glob(os.path.join('./saved_models', 'MR_FCSNN_20_unstable.pt'))
FCSNN_model.load_state_dict(torch.load(pt_match[0], map_location='cpu'))
u_torch = torch.tensor(u, dtype=torch.float)
[y_FCSNN_torch,x] = FCSNN_model(u_torch)
y_pre = y_FCSNN_torch.detach().reshape([Nt2, 1]).numpy()

cm = 1 / 2.54
#%%
fig1 = plt.figure(figsize=(8.8 * cm, 4 * cm))

ax = fig1.add_subplot(111)
ax.plot([10, 10], [-0.2, 2], 'gray', lw=0.75)
ax.plot(t, y_ref/1000, color='k', lw=0.5, label='Exp.')
ax.plot(t, y_pre/1000, color='r', dashes=[2, 2], lw=0.5, label='FCSNN (20)')
ax.text(9, 0.260, 'Training', ha='center', va='center', fontsize=7)
ax.text(11, 0.260, 'Testing', ha='center', va='center', fontsize=7)
ax.set_ylim([-0.2, 2])
ax.set_yticks([0]+np.arange(0.5, 2.1, 0.5).tolist())
ax.set_xlim([0, 18])
ax.set_xticks([0,4,8,10,12,16,18])
labels = [item.get_text() for item in ax.get_xticklabels()]
labels_new = [i.replace('$\\mathdefault{', '').replace('}$', '') for i in labels]
ax.set_xticklabels(labels_new)
labels = [item.get_text() for item in ax.get_yticklabels()]
labels_new = [i.replace('$\\mathdefault{', '').replace('}$', '').replace('.0', '') for i in labels]
ax.set_yticklabels(labels_new)
ax.tick_params(axis='both', direction='in', width=0.75, labelsize=7, length=2.5)
ax.grid(lw=0.5)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_linewidth(0.75)
legend = ax.legend(loc='upper left', bbox_to_anchor=(0, 1), borderpad=0.3, borderaxespad=1, handlelength=1.8,
                edgecolor='black', fontsize=7, ncol=2, columnspacing=0.4, handletextpad=0.2)  # labelspacing=0
legend.get_frame().set_boxstyle('Square', pad=0.0)
legend.get_frame().set_lw(0.75)
legend.get_frame().set_alpha(None)

ax.set_ylabel(r'Force (kN)', fontsize=7, labelpad=-1)
ax.yaxis.label.set_position((_, 0.55))
ax.set_xlabel(r'Time (s)', fontsize=7, labelpad=1)
fig1.tight_layout(pad=0)
# plt.subplots_adjust(hspace=0.15, wspace=0.14)
# fig1.savefig(fname = 'F_MR_unstable.pdf', format = 'pdf')