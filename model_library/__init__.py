import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np


def to_categorical(y):
    y_cat = tf.argmax(y, axis=-1)
    return y_cat


def check_gradients(g_list, b=None, thr=1e-3):
    # check gradients
    zg = 0  # zero grad
    sg = 0  # small grad
    indeces_list = []
    for j, g in enumerate(g_list):
        if np.any(np.isnan(g)):
            print('nan grad')
        if np.all(np.abs(g) == thr):
            zg += 1
        if np.all(np.abs(g) < thr):
            sg += 1
            indeces_list.append(j)
    if zg > 0 or sg > len(g_list)/2:
        print('b{:03d} | zg: {:d}/{len_gra:d} | sg: {:d}/{len_gra:d}'.format(b, zg, sg, len_gra=len(g_list)))
    return indeces_list

# loss functions
def cosine_loss(one_hot_labels, logits, **kwargs):
    """ Computes the cosine loss as described by Barz adn Denzler (2019).

    :param one_hot_labels:  one-hot encoded vector of true labels [B x L]
    :param logits: unscaled vector of predictions from the netowork [B x L]
    :return: cosine loss [B]
    """
    y_true = tf.nn.l2_normalize(one_hot_labels, axis=-1)    # redundant as ohl already have norm 1
    y_pred = tf.nn.l2_normalize(logits, axis=-1)            # logits shall be normalized to have norm 1
    similarity = tf.reduce_sum(y_true * y_pred, axis=-1)    # similarity will be on unit hypersphere
    loss = 1 - similarity
    return loss

def focal_cross_entropy(one_hot_labels, logits, gamma=2., alpha=1.):
    """ Computes the focal cross entropy loss.

    :param one_hot_labels:
    :param logits:
    :return:
    """
    cat_cross_entr = tf.losses.categorical_crossentropy(one_hot_labels, logits, from_logits=True)
    sftmx_lgts = tf.nn.softmax(logits, axis=-1)
    focal_coeff = tf.reduce_sum(sftmx_lgts * one_hot_labels, axis=-1)
    focal_coeff = tf.pow(1 - focal_coeff, gamma)
    loss = cat_cross_entr * focal_coeff
    return loss


def softmax_cross_entropy_focal_loss(one_hot_labels, logits, gamma=2., alpha=0.25, eval_weights=True):
    # eps = tf.constant(1e-7, dtype=tf.float32)
    # # compute softmax
    # s_p = tf.math.softmax(logits, axis=-1)
    # # apply log, add eps
    # log_s_p = tf.math.log(s_p + eps)
    # # apply coeff
    # gamma_coeff = tf.pow(1 - s_p, gamma)
    # log_s_p = - log_s_p * gamma_coeff
    # log_s_p = tf.losses.categorical_crossentropy(one_hot_labels, logits, from_logits=True)
    log_s_p = 1 + tf.losses.cosine_similarity(one_hot_labels, tf.nn.l2_normalize(tf.nn.softmax(logits, axis=-1), axis=-1), axis=-1)
    # evaluate weights
    if eval_weights:
        _w = eval_batch_weights(one_hot_labels)
        w =  tf.matmul(one_hot_labels, _w)
    else:
        w = one_hot_labels
    # multiply by gt
    # TODO: if going back remove SUM
    ce = log_s_p * tf.reduce_sum(w, axis=-1)
    # ce = tf.reduce_sum(ce, axis=-1)
    # to return
    to_return = tf.reduce_mean(ce)
    return to_return

def eval_batch_weights(true_one_hot_labels):
    eps = tf.constant(1e-7, dtype=tf.float32)   # to avoid division by 0
    labels_sum = tf.reduce_sum(true_one_hot_labels, axis=0)
    labels_max = tf.reduce_max(labels_sum)
    labels_weights = labels_max / (labels_sum + eps)
    labels_weights_diag = tf.linalg.diag(labels_weights)
    return labels_weights_diag

def cross_entropy_segm(one_hot_labels, logits, gamma=1.):
    eps = tf.constant(1e-7, dtype=tf.float32)
    # define weights
    weights = tf.constant([1., 2.], dtype=tf.float32)
    weights = weights * one_hot_labels
    weights = tf.reduce_sum(weights, axis=-1)
    # compute softmax
    s_p = tf.math.softmax(logits, axis=-1)
    # apply log, add eps
    log_s_p = tf.math.log(s_p + eps)
    # apply coeff
    gamma_coeff = tf.pow(1 - s_p, gamma)
    log_s_p = log_s_p * gamma_coeff
    # multiply by gt
    ce = - log_s_p * one_hot_labels
    ce = tf.reduce_sum(ce, axis=-1)
    # multiply by weights
    ce_w = ce * weights
    # sum all
    to_return = tf.reduce_mean(ce_w, axis=-1)
    to_return = tf.reduce_mean(to_return)
    return to_return


def gamma_sigmoid(x, gamma):
    return 1. / (1. + tf.exp(-gamma * x))

def orthogonality_of_features(features_tensor):
    # input tensor has shape N x H x W x C
    # new shape shall be NHW x C
    new_shape = (-1, features_tensor.shape[-1])
    reshaped_features = tf.reshape(features_tensor, new_shape)  # shape = NHW x C
    orth_matrix = tf.matmul(reshaped_features, reshaped_features, transpose_a=True) # [C x NHW][NHW x C]
    # evaluate magnitude of orth matrix
    orth_norm = tf.norm(orth_matrix, ord='euclidean')
    return orth_norm