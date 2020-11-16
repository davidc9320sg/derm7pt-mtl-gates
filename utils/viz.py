import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, StrMethodFormatter
import seaborn as sns
import imageio
import os
from multiprocessing import Pool
from utils.gates import load_alphas_in_dir

def draw_alphas_at_epoch(path_to_alphas, title=None, figsize=(8, 4), dpi=100):
    if os.path.isdir(path_to_alphas):
        # init
        list_of_alphas = []
        block_idx = 0
        alpha_idx = 0
        # load alphas
        list_of_alphas_files = sorted(os.listdir(path_to_alphas))
        list_of_alphas_files = [aname for aname in list_of_alphas_files if aname.endswith('.npy')]
        # print(list_of_alphas_files)
        for block_idx, a_filename in enumerate(list_of_alphas_files):
            alpha_complete = np.load(path_to_alphas + '/' + a_filename)
            for alpha_idx, alpha_arr in enumerate(alpha_complete):
                list_of_alphas.append(alpha_arr)
        # init figure
        n_blocks = block_idx + 1    # rows
        n_alphas = alpha_idx + 1    # cols
        fig, ax = plt.subplots(n_blocks, n_alphas, dpi=dpi, figsize=figsize, facecolor=(223 / 255, 235 / 255, 235 / 255),
                               sharey=True)
        # fig.tight_layout(h_pad=2, rect=(0, 0, .8, .95))
        i = 0
        for b, row in enumerate(ax):
            for c, cell in enumerate(row):
                plt.sca(cell)
                sns.heatmap(list_of_alphas[i], cbar=False, square=False, vmin=0., vmax=1., xticklabels='',
                            yticklabels=list(range(8)),
                            cmap='RdYlGn', linewidths=0., robust=False)
                if c == 0:
                    plt.ylabel('b{:02d}'.format(b))
                if b == 0:
                    plt.title('t{:02d}'.format(c))
                i += 1
        # add common title to subplots
        if title is not None:
            fig.suptitle('{}'.format(title))
        # draw on canvas
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        #return
        return image

def make_gif_of_alphas(
        path_to_all_alphas, output_path, exp_name='', fps=2,
        figsize=(5, 3), dpi=100, interval=1, max_epoch=None
):
    # init
    frames = []
    list_of_epoch_folders = sorted(os.listdir(path_to_all_alphas))
    # loop over folders
    print('Making frames...', end='')

    fun_args = []
    for e, e_name in enumerate(list_of_epoch_folders):
        if e % interval == 0:
            path_to_alpha = os.path.join(path_to_all_alphas, e_name)
            if os.path.isdir(path_to_alpha):
                # make title
                title = '{}\nepoch {}'.format(exp_name, e_name)
                # append args
                fun_args.append(
                    (
                        path_to_alpha,
                        title,
                        figsize, dpi
                    )
                )
        if max_epoch is not None:
            if e >= max_epoch:
                break
    # launch process
    with Pool(10) as p:
        frames = p.starmap(
            draw_alphas_at_epoch,
            fun_args
        )
    print('{:2.2f}%'.format(100), flush=True)
    # save
    print('Saving mosaic.')
    imageio.mimsave(output_path, frames, fps=fps)
    print('Mosaic done.')



def plot_loss_and_scores_over_epochs(
        df,
        metric,
        n_tasks,
        savedir,
        dpi=100,
        figsize=(7, 7)
):
    # plot loss vs epochs
    fig, ax = plt.subplots(1, 1, dpi=dpi, figsize=figsize)
    default_fontsize = plt.rcParams.get('font.size')
    plt.rcParams.update({'font.size': 15})
    plt.sca(ax)
    plt.plot(df['epoch'], df['loss'], label='training')
    plt.plot(df['epoch'], df['loss_val'], label='validation')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    x_ticks = ax.get_xticks()
    x_ticks = [int(x - (x % 5)) for j, x in enumerate(x_ticks)]
    ax.set_xticks(x_ticks)
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:d}'))
    title = 'loss over epochs'
    plt.title(title)
    plt.legend(loc=2)
    fname = savedir + '/' + '_'.join(title.split()) + '.png'
    plt.savefig(fname)
    plt.close(fig)

    # plot average f1 score
    for partition, suffx in zip(['training', 'validation'], ['', '_val']):
        fig, ax = plt.subplots(1, 1, dpi=dpi, figsize=figsize)
        title = '{} {} score over epochs'.format(partition, metric)
        column_format = '{}_t{:02d}{}'
        current_cols = [column_format.format(metric, t, suffx) for t in range(n_tasks)]
        current_avg = df[current_cols].mean(axis=1)
        for tc in current_cols:
            plt.plot(df['epoch'], df[tc], linestyle='dashed', alpha=0.5, label=tc)
        plt.plot(df['epoch'], current_avg, linestyle='-', alpha=1., label='avg', color='black', linewidth=3.)
        ax.set_xticks(x_ticks)
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:d}'))
        plt.title(title)
        plt.xlabel('epoch')
        plt.ylabel(metric)
        plt.legend(loc=2)
        fname = savedir + '/' + '_'.join(title.split()) + '.png'
        plt.savefig(fname)
        plt.close(fig)

    plt.rcParams.update({'font.size': default_fontsize})


def plot_sharing_score_over_epochs(
        path_to_all_alphas, output_path, exp_name='',
        figsize=(5, 3), dpi=100, max_epoch=None
):
    list_of_epoch_folders = sorted(os.listdir(path_to_all_alphas))
    all_alphas_per_epoch = []
    for alpha_folder in list_of_epoch_folders:
        loaded_blocks = load_alphas_in_dir(alpha_folder, by_block=True)
        all_alphas_per_epoch.append()

    # select block
    b = -1 # last block
    for e, all_blocks in enumerate(all_alphas_per_epoch):
        pass