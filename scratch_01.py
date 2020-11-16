from derm7pt_dataset import Derm7pt
import tensorflow as tf
from utils import make_run_name
import os
import matplotlib.pyplot as plt
from model_library.derm7pt import *
from utils.logging import ReportParams
from utils.metrics import AllMetrics, MyCSVLogger, to_categorical
from data_structures import NewEarlyStopping
from utils.viz import draw_alphas_at_epoch, make_gif_of_alphas, plot_loss_and_scores_over_epochs
import pandas as pd
from subprocess import run


@tf.function
def eval_classification_loss(y, ans, loss_fn, eval_weights_flag=False):
    all_losses = []
    for y_true, logits in zip(y, ans):
        this_loss = focal_cross_entropy(y_true, logits, gamma=FOCAL_LOSS_GAMMA, alpha=FOCAL_LOSS_ALPHA)
        # evaluate mini-batch weights
        if eval_weights_flag:
            _w = eval_batch_weights(y_true)
            w = tf.matmul(y_true, _w)
            w = tf.reduce_sum(w, axis=-1)
            this_loss = this_loss * w
        # eval mean
        this_loss = tf.reduce_mean(this_loss)
        # append to loss list
        all_losses.append(this_loss)
    loss = tf.reduce_sum(all_losses)
    return loss, all_losses


@tf.function
def minimize_variables(this_loss, vars: tf.Tensor, optimizer: tf.optimizers.Optimizer, tape: tf.GradientTape):
    grads = tape.gradient(this_loss, vars)
    optimizer.apply_gradients(zip(grads, vars))


def one_epoch(
        model: GenericGatedModel, loss_fn, tf_ds, training=False,
        loss_log=None, loss_log_per_task=None,
        metrics_log_per_task: (AllMetrics,) = None
):
    # set BN layers to trainable/not
    # model.set_bn_trainable(training)
    # reset metrics loggers
    if loss_log is not None:
        loss_log.reset_states()
    if loss_log_per_task is not None:
        for log in loss_log_per_task: log.reset_states()
    if metrics_log_per_task is not None:
        for log in metrics_log_per_task: log.reset_states()
    consistency_logger = keras.metrics.Mean()
    orthogonality_logger = keras.metrics.Mean()
    # loop over tensorflow dataset
    dots = ''
    for x, y in tf_ds:
        # print dots
        dots += '.'
        print(dots, flush=True, end='')
        # main
        if training:
            with tf.GradientTape(persistent=True) as tape:
                model_ans = model(x, training=True)
                # separate variables if not already
                if model.weights_conv is None and model.weights_alpha is None:
                    model.weights_conv = model.task_specific_models.trainable_weights
                    if model.bn_momentum is not None:
                        model.weights_conv += model.bn.trainable_weights
                    model.weights_alpha = model.all_gates.trainable_weights
                # loss -------------------------
                loss, all_losses = eval_classification_loss(y, model_ans, loss_fn, eval_weights_flag=EVAL_WEIGHTS_TRAIN)
                # l2 regularization -------------
                l2_reg_loss = tf.constant(0., dtype=tf.float32)
                for w in model.trainable_weights:
                    if 'kernel' in w.name:
                        # if w.shape[0] == 1 and w.shape[1] == 1:
                            # only apply to 1x1 conv
                        l2_reg_loss += GAMMA_L2 * tf.nn.l2_loss(w)
                loss += l2_reg_loss
                # l1 reg ------------------------
                if GAMMA_L1 != 0.:
                    all_alphas_vector = gamma_sigmoid(
                        tf.concat(model.all_gates.trainable_weights, axis=-1), GATES_GAMMA
                    )
                    # all_alphas_vector = tf.concat(model.all_gates.trainable_weights, axis=-1) # raw
                    # all_alphas_vector = tf.abs(tf.abs(all_alphas_vector) - 3.)
                    l1_reg_loss = GAMMA_L1 * tf.reduce_sum(all_alphas_vector)
                else:
                    l1_reg_loss = tf.constant(0., dtype=tf.float32)
                loss += l1_reg_loss
            # optimize --------------------------------------------------
            if LEARNING_RATE_0 != LEARNING_RATE_1:
                # optimize - two optimizers
                grads = tape.gradient(loss, model.weights_conv)
                opt_0.apply_gradients(zip(grads, model.weights_conv))
                grads = tape.gradient(loss, model.weights_alpha)
                opt_1.apply_gradients(zip(grads, model.weights_alpha))
            else:
                # optimize - one optimizer for all
                grads = tape.gradient(loss, model.trainable_weights)
                opt_0.apply_gradients(zip(grads, model.trainable_weights))
            del tape
        else:
            model_ans = model(x, training=False)
            loss, all_losses = eval_classification_loss(y, model_ans, loss_fn, eval_weights_flag=EVAL_WEIGHTS_TEST)
        # metrics
        loss_log(loss)
        for l, log in zip(all_losses, loss_log_per_task): log(l)
        for y_true, logits, log in zip(y, model_ans, metrics_log_per_task):
            y_true_cat = to_categorical(y_true).numpy().astype(int)
            logits_cat = to_categorical(tf.nn.softmax(logits)).numpy().astype(int)
            log(y_true_cat, logits_cat)
        print('\b' * len(dots), flush=True, end='')
    # print
    suffix = 't' if not training else ''
    to_print = '{:02d}{}|{:.4f}|'.format(epoch, suffix, loss_log.result().numpy())
    if training:
        to_print += 'L1:{:.4f}|L2:{:.4f}|OT:{:.4f}|7PT:{:.4f}|\t'.format(
            l1_reg_loss, l2_reg_loss, orthogonality_logger.result().numpy(), consistency_logger.result().numpy()
        )
        del consistency_logger
        del orthogonality_logger
    for log in loss_log_per_task:
        to_print += '|{:.3f}'.format(log.result().numpy())
    to_print += '\t|>F1'
    for log in metrics_log_per_task:
        to_print += '|{:.2f}'.format(log.result(to_show=(3,))[0])
    print(to_print, flush=True)


if __name__ == '__main__':
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # main params ------------------
    tf.random.set_seed(0)
    REAL_NAME = 'standard'
    RUN_NAME = make_run_name(REAL_NAME)
    SAVEDIR = '/data/projects/isic-cvpr-mtl-derm7pt/saves/{}'.format(RUN_NAME)
    EPOCHS = 999
    LOAD_MODEL = None
    # dataset parameters -----------
    # TAKE = -1
    # TAKE_TEST = -1
    # TAKE_VAL = -1
    DATASET_K = 6
    BINARY_DIAGNOSIS = False
    BATCH_SIZE = DATASET_K * 24 if not BINARY_DIAGNOSIS else 21 * DATASET_K
    RESHUFFLE_TRAINING = True  # if TAKE >= 5 or TAKE == -1 else False
    # AUGMENT_TRAIN = False
    RANDOM_ROTATIONS = True  # does not impact anything
    MAX_ITER_TRAINING = 5 # 2000 // (DATASET_K * 24)
    print('K: {}\tBS: {}\tMAX_ITER_TRAINING: {}'.format(DATASET_K, BATCH_SIZE, MAX_ITER_TRAINING))
    # hyperparameters -------------
    EARLY_STOPPING_MONITOR = 'loss_val'
    EARLY_STOPPING_PATIENCE = 200
    EARLY_STOPPING_DELTA = 0.
    EARLY_STOPPING_SHOULD_DECREASE = True
    LEARNING_RATE_DECAY_STEPS = MAX_ITER_TRAINING * 10
    LEARNING_RATE_DECAY_RATE = 1.
    LEARNING_RATE_0 = 1e-3
    LEARNING_RATE_1 = LEARNING_RATE_0 * 100
    EVAL_WEIGHTS_TRAIN = True
    EVAL_WEIGHTS_TEST = False
    GATES_OPEN = True
    GATES_GAMMA = 3.0
    # coefficients
    GAMMA_L1 = tf.constant(0., dtype=tf.float32)
    GAMMA_L2 = tf.constant(1e-4, dtype=tf.float32)
    GAMMA_ORTH = tf.constant(0., dtype=tf.float32)  # not used in final experiments
    GAMMA_CONSISTENCY = tf.constant(0., dtype=tf.float32)  # not used in final experiments
    FOCAL_LOSS_GAMMA = tf.constant(2., dtype=tf.float32)
    FOCAL_LOSS_ALPHA = tf.constant(1., dtype=tf.float32)
    # make directories
    os.makedirs(SAVEDIR, exist_ok=True)
    os.makedirs(SAVEDIR+'/alpha', exist_ok=True)
    # report params
    log_params = ReportParams(
        SAVEDIR+'/parameters.txt', run_name=RUN_NAME, savedir=SAVEDIR, max_epochs=EPOCHS, load_model=LOAD_MODEL,
        early_stopping_monitor=EARLY_STOPPING_MONITOR, early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_delta=EARLY_STOPPING_DELTA, learning_rate_0=LEARNING_RATE_0, learning_rate_1=LEARNING_RATE_1,
        eval_mb_weights_train=EVAL_WEIGHTS_TRAIN, eval_mb_weights_test=EVAL_WEIGHTS_TEST,
        gamma_l1=GAMMA_L1, gamma_l2=GAMMA_L2, gamma_orth=GAMMA_ORTH, gamma_consistency=GAMMA_CONSISTENCY,
        focal_loss_gamma=FOCAL_LOSS_GAMMA, focal_loss_alpha=FOCAL_LOSS_ALPHA,
        ds_k=DATASET_K, batch_size=BATCH_SIZE,
        reshuffle_training=RESHUFFLE_TRAINING, max_iter_training=MAX_ITER_TRAINING,
        gates_open=GATES_OPEN, lr_decay=LEARNING_RATE_DECAY_RATE, lr_decay_steps=LEARNING_RATE_DECAY_STEPS,
        gates_gamma=GATES_GAMMA
        # ds_take_train=TAKE, ds_take_test=TAKE_TEST, ds_take_val=TAKE_VAL
    )
    # data
    dataset = Derm7pt('/data/datasets/derm7pt-data', binary_diagnosis=BINARY_DIAGNOSIS)
    train, val, test = dataset.get_balanced_dataset(
        k=DATASET_K,
        reshuffle_training=RESHUFFLE_TRAINING,
        random_rotations=RANDOM_ROTATIONS,
        max_iter_train=MAX_ITER_TRAINING
    )
    output_shapes = []
    for _, y in train:
        for _y in y:
            output_shapes.append(_y.shape[-1])
        break
    log_params.append(output_shapes=output_shapes, target_size=dataset.target_size)
    # model
    my_model = GenericGatedModel(
        output_shapes, base_model=SimpleCNNModel, gates_open=GATES_OPEN, name='my_model',
        input_shape=[*dataset.target_size, 3], gates_gamma=GATES_GAMMA
    )
    n_gated_blocks = my_model.n_blocks
    log_params.append(network_setup=my_model.task_specific_models[0].network_setup, bn_momentum=my_model.bn_momentum)
    # optimizers
    opt_0 = tf.optimizers.Adam(learning_rate=LEARNING_RATE_0, name='opt_0')
    opt_1 = tf.optimizers.Adam(learning_rate=LEARNING_RATE_1, name='opt_1')
    # loss function
    loss_function = softmax_cross_entropy_focal_loss
    # loss log
    epoch_mean = keras.metrics.Mean()
    epoch_mean_task = [keras.metrics.Mean() for _ in output_shapes]
    epoch_mean_val = keras.metrics.Mean()
    epoch_mean_task_val = [keras.metrics.Mean() for _ in output_shapes]
    epoch_mean_test = keras.metrics.Mean()
    epoch_mean_task_test = [keras.metrics.Mean() for _ in output_shapes]
    # metrics log
    train_metrics = [AllMetrics(name='t{:02d}'.format(t)) for t, _ in enumerate(output_shapes)]
    val_metrics = [AllMetrics(name='t{:02d}_val'.format(t)) for t, _ in enumerate(output_shapes)]
    test_metrics = [AllMetrics(name='t{:02d}_test'.format(t)) for t, _ in enumerate(output_shapes)]
    # csv log
    logger = MyCSVLogger('{}/logger.csv'.format(SAVEDIR))
    # checkpoint
    best_val_score = 99999.0 if EARLY_STOPPING_SHOULD_DECREASE else 0.
    best_val_score_checkpoint = tf.Variable(best_val_score, dtype=tf.float32)
    best_val_epoch_checkpoint = tf.Variable(0, dtype=tf.int32)
    ckpt = tf.train.Checkpoint(
        step=tf.Variable(1), best_score=best_val_score_checkpoint,
        best_epoch=best_val_epoch_checkpoint
    )
    manager = tf.train.CheckpointManager(ckpt, '{}/ckpts'.format(SAVEDIR), max_to_keep=1)
    if LOAD_MODEL is not None:
        print('restoring checkpoint {}'.format(LOAD_MODEL))
        print('run once')
        for x_tmp, _ in train.take(1):
            my_model(x_tmp)
            break
        ckpt.restore(LOAD_MODEL).assert_consumed()
        best_val_score = best_val_score_checkpoint.numpy()
        my_model.load_weights('{}/ckpts/save-weights'.format(SAVEDIR))
    # early stopping
    early_stopping = NewEarlyStopping(
        monitor=EARLY_STOPPING_MONITOR, min_delta=EARLY_STOPPING_DELTA, patience=EARLY_STOPPING_PATIENCE
    )
    # MAIN LOOP ---------------------------------------------------------------------
    last_e = best_val_epoch_checkpoint.numpy()
    min_epoch=last_e
    max_epoch = last_e + EPOCHS
    early_stopping.on_train_begin()
    for epoch in range(min_epoch, max_epoch):
        # run one epoch on training set
        one_epoch(
            my_model, tf_ds=train, training=True, loss_fn=loss_function,
            loss_log=epoch_mean, loss_log_per_task=epoch_mean_task,
            metrics_log_per_task=train_metrics
        )
        # run one epoch on validation set
        one_epoch(
            my_model, tf_ds=val, training=False, loss_fn=loss_function,
            loss_log=epoch_mean_val, loss_log_per_task=epoch_mean_task_val,
            metrics_log_per_task=val_metrics
        )
        # log --------------------------------------------------------
        logs = {
            'epoch': epoch,
            'loss': epoch_mean.result().numpy().astype(np.float64),
            'loss_val': epoch_mean_val.result().numpy().astype(np.float64)
        }
        # log all
        log_train_tmp = {'loss_t{:02d}'.format(t) : met.result().numpy().astype(np.float64) for t, met in enumerate(epoch_mean_task)}
        log_test_tmp = {'loss_t{:02d}_val'.format(t) : met.result().numpy().astype(np.float64) for t, met in enumerate(epoch_mean_task_val)}
        logs = {**logs, **log_train_tmp, **log_test_tmp}
        for met in train_metrics + val_metrics:
            logs = {**logs, **met.as_dict()}
        logger.on_epoch_end(logs)
        # save alphas ------------------------------------------------
        ans = my_model.get_all_alpha(flatten=False, numpy=True)
        alpha_epoch_dir = SAVEDIR + '/alpha/e{:04d}'.format(epoch)
        os.makedirs(alpha_epoch_dir, exist_ok=True)
        for b in range(n_gated_blocks):
            i_0, i_1 = b * len(output_shapes), (1+b) * len(output_shapes)
            # print(i_0, i_1)
            to_save = ans[i_0:i_1]
            np.save(alpha_epoch_dir+'/b{:02d}.npy'.format(b), to_save)

        # plot metrics ------------------
        plot_loss_and_scores_over_epochs(
            logger.df, metric='f1', n_tasks=len(dataset.tasks), savedir=SAVEDIR,
            dpi=100, figsize=(10, 7)
        )
        plot_loss_and_scores_over_epochs(
            logger.df, metric='loss', n_tasks=len(dataset.tasks), savedir=SAVEDIR,
            dpi=100, figsize=(10, 7)
        )

        # early stopping + checkpoint -------------------------------------------------
        # check for stopping
        early_stopping.on_epoch_end(epoch, logs=logs)
        if early_stopping.wait == 0:
            # save if best
            ckpt.step.assign_add(1)
            ckpt.best_score.assign(early_stopping.best)
            ckpt.best_epoch.assign(epoch)
            my_model.save_weights('{}/ckpts/save-weights'.format(SAVEDIR), save_format='tf')
        elif early_stopping.stop_training:
            print('stopping training on epoch {}.'.format(epoch))
            break
            # ckpt_save_path = manager.save()

        print('patience: ', early_stopping,
              '| best_score {:.4f} | current score {:.4f}'.format(
                  early_stopping.best, early_stopping.get_monitor_value(logs))
              )

    # log parameters num
    log_params.append(n_params=my_model.count_params())

    # test phase =======================================================================================================
    # load best model
    print('restoring best model (epoch: {:02d} | score: {:.4f})'.format(
        best_val_epoch_checkpoint.value(), best_val_score_checkpoint.value()))
    # ckpt.restore(manager.latest_checkpoint).assert_consumed()
    my_model.load_weights('{}/ckpts/save-weights'.format(SAVEDIR)).assert_consumed()

    print('beginning test phase.')
    one_epoch(
        my_model, tf_ds=test, training=False, loss_fn=loss_function,
        loss_log=epoch_mean_test, loss_log_per_task=epoch_mean_task_test,
        metrics_log_per_task=test_metrics
    )
    # log --------------------------------------------------------
    logs = {
        'epoch': 'TEST',
        'loss_test': epoch_mean_test.result().numpy().astype(np.float64)
    }
    # log all
    log_test_tmp = {'loss_t{:02d}_test'.format(t): met.result().numpy().astype(np.float64) for t, met in
                    enumerate(epoch_mean_task_test)}
    logs = {**logs, **log_test_tmp}
    for met in test_metrics:
        logs = {**logs, **met.as_dict()}
    logger.on_epoch_end(logs)
    print('model params: {:d}'.format(my_model.count_params()))
    # save the actual test outputs
    test_metrics_path = SAVEDIR + '/test'
    os.makedirs(test_metrics_path, exist_ok=True)
    for tm in test_metrics:
        fname = '{}/{}.allmetrics'.format(test_metrics_path, tm.name)
        tm.save(fname)
    ## ADJUST LOGGER COPY ==============================================================================================
    N_TASKS = len(dataset.tasks)
    df = logger.df
    best_val_epoch = best_val_epoch_checkpoint.numpy()
    best_row = df.iloc[best_val_epoch]
    print('best epoch: ', best_val_epoch)
    metrics_list = ['acc', 'rec', 'prec', 'f1']
    suffx = ['', '_val']

    results_set = []
    for sfx in suffx:
        entry = []
        for m in metrics_list:
            l = []
            for t in range(N_TASKS):
                df_key = '{}_t{:02d}{}'.format(m, t, sfx)
                l.append(best_row[df_key])
            entry.append(l)
        entry = pd.DataFrame.from_records(entry, index=metrics_list)
        avg_col = entry[range(1, 8)].mean(axis=1).rename('avg_7pt')
        avg_col_all = entry.mean(axis=1).rename('avg_all')
        results_set.append(
            pd.concat([entry, avg_col, avg_col_all], axis=1)
        )

    # test
    test_row = df.iloc[-1]
    entry = []
    for m in metrics_list:
        l = []
        for t in range(N_TASKS):
            df_key = '{}_t{:02d}_test'.format(m, t)
            l.append(test_row[df_key])
        entry.append(l)
    entry = pd.DataFrame.from_records(entry, index=metrics_list)
    avg_col = entry[range(1, 8)].mean(axis=1).rename('avg_7pt')
    avg_col_all = entry.mean(axis=1).rename('avg_all')
    test_res = pd.concat([entry, avg_col, avg_col_all], axis=1)

    for dataframe, name in zip([*results_set, test_res], ['train', 'val', 'test']):
        dataframe.to_csv(SAVEDIR + '/{}.csv'.format(name))

    report_string = ''
    for m in test_metrics:
        report_string += m.name.ljust(20, '-')  + '\n\n'
        for k, v in m.as_dict().items():
            report_string += '{}:\t{:.2f}\n'.format(k, v)
        report_string += '\n'
        report_string += str(m.confusion_matrix) + '\n\n'
    with open(SAVEDIR+'/test_report.txt', 'w') as f:
        f.write(report_string)

    # run other report functions -----------------------
    list_of_alphas = []
    alpha_dir = SAVEDIR + '/alpha'.format(best_val_epoch)
    best_alpha_dir = alpha_dir + '/e{:04d}/'.format(best_val_epoch)
    last_run_epoch = int(logger.df.epoch[:-1].max())
    last_alpha_dir = alpha_dir + '/e{:04d}/'.format(last_run_epoch)

    # plot best alpha
    im = draw_alphas_at_epoch(best_alpha_dir, 'e{:04d}'.format(best_val_epoch))
    plt.imsave(SAVEDIR+'/gates_best.png', im)
    plt.close()
    # plot alphas at last epoch
    im = draw_alphas_at_epoch(last_alpha_dir, 'e{:04d}'.format(last_run_epoch))
    plt.imsave(SAVEDIR + '/gates_last.png', im)
    plt.close()
    # examine gates on best epoch
    run(['python', 'examine_gates.py', '-p', alpha_dir, '-e', str(best_val_epoch)])
    # examine gates on last epoch
    run(['python', 'examine_gates.py', '-p', alpha_dir, '-e', str(last_run_epoch)])
    # run 7pt rule
    run(['python', 'seven-pt-rule.py', '-p', SAVEDIR, '-n', REAL_NAME])

    # make gif
    make_gif_of_alphas(
        alpha_dir, SAVEDIR+'/gates_all.gif',
        exp_name='',
        fps=4,
        interval=2
    )

    print('end')