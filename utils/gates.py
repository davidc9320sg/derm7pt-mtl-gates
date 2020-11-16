import numpy as np
import os

def step_fn(x, threshold):
    if x > threshold:
        return 1
    else:
        return 0

def load_alphas_in_dir(path_to_alphas, by_block=False, by_task=False, flatten=False, apply_step=False, threshold=0.1):
    assert not (by_block and by_task)
    step_vectorized = np.vectorize(step_fn)
    # init
    list_of_alphas = []
    block_idx = 0
    alpha_idx = 0
    # load alphas
    list_of_alphas_files = sorted(os.listdir(path_to_alphas))
    for block_idx, a_filename in enumerate(list_of_alphas_files):
        if a_filename.endswith('.npy'):
            alpha_complete = np.load(path_to_alphas + '/' + a_filename)
            if apply_step:
                alpha_complete = step_vectorized(alpha_complete, threshold=threshold)
            if flatten:
                alpha_complete = alpha_complete.reshape((alpha_complete[0].shape[0], -1))
            if by_block:
                list_of_alphas.append(alpha_complete)
            else:
                for alpha_idx, alpha_arr in enumerate(alpha_complete):
                    list_of_alphas.append(alpha_arr)
    n_blocks, n_tasks = block_idx + 1, alpha_idx + 1
    # group by block
    if by_block:
        list_of_alphas_tmp = []
        for b in range(n_blocks):
            pass
    return list_of_alphas