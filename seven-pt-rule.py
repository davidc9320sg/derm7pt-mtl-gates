from derm7pt_dataset import Derm7pt
from utils.metrics import AllMetrics
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from optparse import OptionParser

def apply_threshold(x, threshold):
    if x >= threshold:
        return 1.
    else:
        return 0.


def binary_diag_labels(x):
    if x == 2:
        return 1.
    else:
        return 0.

if __name__ == '__main__':
    default_fontsize = plt.rcParams.get('font.size')
    plt.rcParams.update({'font.size': 20})
    base = Derm7pt()

    parser = OptionParser()
    parser.add_option('-p', '--path', dest='path', type='string')
    parser.add_option('-n', '--name', dest='exp_name', type='string')
    options, _ = parser.parse_args()

    savedir = options.path
    exp_name = options.exp_name
    savedir_test = savedir + '/test'
    savedir_files = sorted(os.listdir(savedir_test))
    test_allmetrics = []
    for fname in savedir_files:
        if fname.endswith('.allmetrics'):
            new = AllMetrics()
            new.load(savedir_test+'/'+fname)
            test_allmetrics.append(
                new
            )

    def eval_score(pred):
        return scores[pred]
    all_scores = []
    for t, scores_tf in enumerate(base.criteria_scores):
        scores = scores_tf.numpy()
        y_pred = test_allmetrics[t+1].y_pred
        vect_fn = np.vectorize(eval_score)
        ans = vect_fn(y_pred)
        all_scores.append(ans)
    all_scores = np.array(all_scores)
    all_scores_sum = all_scores.sum(axis=0)

    diag_pred_t1 = np.vectorize(apply_threshold)(all_scores_sum, threshold=1.)
    diag_pred_t3 = np.vectorize(apply_threshold)(all_scores_sum, threshold=3.)
    #TODO: dinamically identify binary experiments
    if len(np.unique(test_allmetrics[0].y_true)) > 2:
        diag_labels = np.vectorize(binary_diag_labels)(test_allmetrics[0].y_true)
        diag_pred_direct = np.vectorize(binary_diag_labels)(test_allmetrics[0].y_pred)
    else:
        diag_labels = test_allmetrics[0].y_true
        diag_pred_direct = test_allmetrics[0].y_pred
    sevenpt_diag_metrics_t1 = AllMetrics(normalize_cmat=False)
    sevenpt_diag_metrics_t1(diag_labels, diag_pred_t1)
    sevenpt_diag_metrics_t3 = AllMetrics(normalize_cmat=False)
    sevenpt_diag_metrics_t3(diag_labels, diag_pred_t3)
    direct_binary_metrics = AllMetrics(normalize_cmat=False)
    direct_binary_metrics(diag_labels, diag_pred_direct)
    # save as txt
    s = 't=1\t' + str(sevenpt_diag_metrics_t1) + '\n' + str(sevenpt_diag_metrics_t1.confusion_matrix)
    s += '\nt=3\t' + str(sevenpt_diag_metrics_t3) + '\n' + str(sevenpt_diag_metrics_t3.confusion_matrix)
    s += '\ndirect\t' + str(direct_binary_metrics) + '\n' + str(direct_binary_metrics.confusion_matrix)
    print(
        s
    )
    with open(savedir + '/sevenpt_res_gt.txt', 'w') as f:
        f.write(s)
    # save cmat
    sevenpt_diag_metrics_t1.save(savedir+'/sevenpt_test_t1_gt.allmetrics')
    sevenpt_diag_metrics_t3.save(savedir + '/sevenpt_test_t3_gt.allmetrics')
    direct_binary_metrics.save(savedir + '/direct_binary_gt.allmetrics')

    # plot
    for key, mtrcs in zip(['t1', 't3', 'dir'], [sevenpt_diag_metrics_t1, sevenpt_diag_metrics_t3, direct_binary_metrics]):
        data = mtrcs.confusion_matrix
        vmax = max(data.sum(axis=1))
        fig = plt.figure(figsize=(4, 4), dpi=300, frameon=False)
        sns.heatmap(data, annot=True, annot_kws={'size': 'large'}, fmt="d", linewidths=1, cbar=False, cmap='Greens',
                    vmin=0, vmax=vmax)
        plt.xlabel('predicted')
        plt.ylabel('true')
        plt.tight_layout()
        plt.savefig('{}/{}_{}.png'.format(savedir, exp_name, key))
        plt.close()