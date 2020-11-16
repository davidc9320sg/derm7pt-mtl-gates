import seaborn as sns
import matplotlib.pyplot as plt
from utils.viz import *
from utils.gates import load_alphas_in_dir
from optparse import OptionParser


def make_colors_per_task(n_tasks, total_len=None, palette_name='muted'):
    base_palette = sns.color_palette(palette_name, n_colors=n_tasks)
    new_palette = []
    total_len = 1 if total_len is None else total_len
    el_per_task = total_len // n_tasks
    for t in range(n_tasks):
        for _ in range(el_per_task):
            new_palette.append(base_palette[t])
    return new_palette


def barplot_of_gate(block_data, title=None):
    x = np.arange(block_data.shape[-1])
    y = block_data.sum(axis=0) - 1
    fig = plt.figure(figsize=(8, 5), dpi=100)
    task_palette = make_colors_per_task(block_data.shape[0], block_data.shape[1])
    sns.barplot(x, y, color=['r', 'b'], palette=task_palette)
    if title is not None:
        plt.title(title)
    return fig


def heatmap_of_gates(block_data, title=None):
    tmp = block_data.sum(axis=0) - 1
    fig = plt.figure(figsize=(16, 10), dpi=100)
    sns.heatmap(
        tmp, vmin=0, vmax=7, square=False, cmap='viridis', cbar=True, annot=False, annot_kws={'size': 'x-large'},
        linewidths=0.
    )
    plt.xlabel('feature map')
    plt.ylabel('task')
    if title is not None:
        plt.title(title)
    return fig


def plot_sharing_matrix(block_data, title=None, compute_sum=True):
    if compute_sum:
        tmp = block_data.sum(axis=2)
        tmp = tmp/tmp.max() * 100
    else:
        tmp = block_data
    fig = plt.figure(figsize=(5, 6), dpi=100, frameon=False)
    tasks = [
        'DIAG',
        'PN',
        'BWV',
        'VS',
        'PIG',
        'STR',
        'DaG',
        'RS'
    ]
    sns.heatmap(
        tmp, vmin=0, vmax=tmp.max(), square=True, cmap='RdYlGn', cbar=False,
        annot=True, annot_kws={'size': 'x-large'}, fmt='.0f',
        linewidths=0.5,
        xticklabels=tasks, yticklabels=tasks
    )
    plt.xlabel('task i\ngiving the feature maps')
    plt.ylabel('task t\ntaking the feature maps')
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-p', '--path', dest='path', type='string')
    parser.add_option('-e', '--epoch', dest='epoch', type='int')
    parser.add_option('-t', '--thr', dest='threshold', type='float', default=-999.)
    options, _ = parser.parse_args()

    path_to_alpha_folder = options.path
    epoch_folder = '/e{:04d}'.format(options.epoch)

    threshold = options.threshold if options.threshold != -999. else None
    apply_step = True if threshold is not None else False
    path_to_alpha = path_to_alpha_folder + epoch_folder
    my_gates = load_alphas_in_dir(
        path_to_alpha, by_block=True, flatten=True, apply_step=apply_step, threshold=threshold
    )
    my_gates_unflattened = load_alphas_in_dir(
        path_to_alpha, by_block=True, flatten=False, apply_step=apply_step, threshold=threshold
    )
    fig_idx = 1
    for b, block in enumerate(my_gates):
        fig = barplot_of_gate(block, 'sharing of gates - barplot\n{:02d}'.format(b))
        fname = path_to_alpha + '/fig_{:02d}'.format(fig_idx)
        if threshold is not None:
            fname += '_t{:02d}'.format(int(threshold * 100))
        plt.savefig(fname)
        plt.close(fig)
        fig_idx +=1
        # plt.show()

    # for b, block in enumerate(my_gates_unflattened):
    #     heatmap_of_gates(block, title='sharing of gates - heatmap\n{:02d}'.format(b))
    #     plt.show()

    # change font parameters
    new_params = {
        'axes.labelsize': 'x-large',
        'axes.titlesize': 'x-large',
        'xtick.labelsize': 'x-large',
        'ytick.labelsize': 'x-large'
    }
    default_params = {}  # get defaults
    for k, v in new_params.items():
        default_params[k] = plt.rcParams.get(k)
    plt.rcParams.update(new_params)
    # print(default_params)
    # print(new_params)

    # correlation matrices
    all_sharing_matrices = []
    filters_g_t = []
    fig, axes = plt.subplots(len(my_gates_unflattened), 2, sharey=True, sharex=True)
    # run over all blocks
    for b, block in enumerate(my_gates_unflattened):
        plot_title = 'sharing percentage (%)\nb{:02d}'.format(b)
        if b ==(len(my_gates_unflattened) - 1):
            plot_title = 'sharing percentage (%)\nlast gated block'
        if threshold is not None:
            plot_title +='\nthreshold = {:.2f}'.format(threshold)
        fig = plot_sharing_matrix(block, title=plot_title)
        # plt.show()
        fname = path_to_alpha + '/fig_{:02d}'.format(fig_idx)
        if threshold is not None:
            fname += '_t{:02d}'.format(int(threshold * 100))
        plt.savefig(fname)
        plt.close(fig)
        fig_idx += 1
        # plt.rcParams.update({'font.size': default_fontsize})

        sharing_matrix = block.sum(axis=2)
        all_sharing_matrices.append(sharing_matrix)
        n_filters = sharing_matrix.max()
        n_tasks = sharing_matrix.shape[0]
        filters_given = (sharing_matrix.sum(axis=0) - n_filters) / (n_filters*(n_tasks-1))
        filters_taken = (sharing_matrix.sum(axis=1) - n_filters) / (n_filters*(n_tasks-1))
        filters_g_t.append((filters_taken, filters_given))

        # plot
        # plt.sca(axes[b][0])
        # sns.barplot(x=np.arange(8), y=filters_given, palette=make_colors_per_task(8, 8))
        # plt.title('given')
        # plt.sca(axes[b][1])
        # sns.barplot(x=np.arange(8), y=filters_taken, palette=make_colors_per_task(8, 8))
        # plt.title('taken')
        # plt.ylim((0, 1.))
    # plt.show()

    all_matrix_sum = np.array(all_sharing_matrices).sum(axis=0)
    all_matrix_sum = all_matrix_sum / all_matrix_sum.max() * 100
    plot_title = 'sharing percentage (%)\nconsidering all blocks'
    if threshold is not None:
        plot_title += '\nthreshold = {:.2f}'.format(threshold)
    plot_sharing_matrix(all_matrix_sum, title=plot_title, compute_sum=False)
    fname = path_to_alpha + '/fig_{:02d}'.format(fig_idx)
    if threshold is not None:
        fname += '_t{:02d}'.format(int(threshold * 100))
    plt.savefig(fname)
    plt.close()
    # restore plt parameters
    plt.rcParams.update(default_params)