import functools

import numpy as np
import bottleneck as bn
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow_probability import distributions as tfp


def binary_crossentropy(y_true, y_pred):
    '''
        The tensorflow style binary crossentropy
    '''
    loss = -tf.reduce_mean(
        tf.reduce_sum(
            y_true * tf.math.log(tf.maximum(y_pred, 1e-10)) + (1-y_true) * 
            tf.math.log(tf.maximum(1-y_pred, 1e-10)), axis=-1
        ))
    return loss


def multinomial_crossentropy(y_true, y_pred):
    '''
        The tensorflow style multinomial crossentropy
    '''    
    loss = -tf.reduce_mean(tf.reduce_sum(
        y_true * tf.math.log(tf.maximum(y_pred, 1e-10)), axis=1
    ))
    return loss


def mse(y_true, y_pred):
    '''
        The tensorflow style mean squared error
    '''
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_true - y_pred), axis=-1))
    return loss


def gaussian_kl_Loss(inputs, use_logstd=True):
    '''
        KL divergence between the variational Gaussian 
        distribution and the prior.
    '''
    mu, std  = inputs
    if use_logstd:
        std =  K.exp(std)
    var_dist = tfp.MultivariateNormalDiag(loc=mu, scale_diag=std)
    pri_dist = tfp.MultivariateNormalDiag(loc=K.zeros_like(mu), 
                                          scale_diag=K.ones_like(std))    
    kl_loss  = K.mean(tfp.kl_divergence(var_dist, pri_dist))
    return kl_loss


def remove_empty_users(eval_func):
    @functools.wraps(eval_func)
    def empty_removed_metric(y_true, y_pred, k, num_cold=None):
        if num_cold is None:
            non_empty_users = np.sum(y_true, axis=1) > 0
            y_true = y_true[non_empty_users]
            y_pred = y_pred[non_empty_users]
            metric = eval_func(y_true, y_pred, k, num_cold)
        else:
            non_empty_users = np.sum(y_true[:, -num_cold:], axis=1) > 0
            y_true = y_true[non_empty_users]
            y_pred = y_pred[non_empty_users]
            metric = eval_func(y_true, y_pred, k, num_cold)
        return metric
    return empty_removed_metric


@remove_empty_users
def Recall_at_k(y_true, y_pred, k, num_cold=None):
    '''
        Average recall for top k recommended results.
        The training records should be set to -inf in y_pred
    '''
    batch_size = y_pred.shape[0]
    topk_idxes = bn.argpartition(-y_pred, k, axis=1)[:, :k]
    y_true_bin = (y_true > 0).astype(np.float32)
    y_pred_bin = np.zeros_like(y_pred, dtype=np.bool)
    y_pred_bin[np.arange(batch_size)[:, None], topk_idxes] = True
    #if num_cold:
    #    y_true_bin[:, :-num_cold] = 0
    hits = np.sum(np.logical_and(y_true_bin, y_pred_bin), axis=-1).astype(np.float32)
    recall = np.mean(hits/np.minimum(k, np.sum(y_true_bin, axis=1)))
    return recall


@remove_empty_users
def NDCG_at_k(y_true, y_pred, k, num_cold=None):
    '''
        Average NDCG for top k recommended results. 
        The training records should be set to -inf in y_pred
    '''
    batch_size = y_pred.shape[0]
    topk_idxes_unsort = bn.argpartition(-y_pred, k, axis=1)[:, :k]
    topk_value_unsort = y_pred[np.arange(batch_size)[:, None],topk_idxes_unsort]
    topk_idxes_rel = np.argsort(-topk_value_unsort, axis=1)
    topk_idxes = topk_idxes_unsort[np.arange(batch_size)[:, None], topk_idxes_rel]
    y_true_bin = (y_true > 0).astype(np.float32)
    #if num_cold:
    #    y_true_bin[:, :-num_cold] = 0
    y_true_topk = y_true_bin[np.arange(batch_size)[:, None], topk_idxes]
    weights = 1./np.log2(np.arange(2, k + 2))
    DCG = np.sum(y_true_topk*weights, axis=-1)
    normalizer = np.array([np.sum(weights[:int(n)]) for n in np.minimum(k, np.sum(y_true_bin, axis=-1))])
    NDCG = np.mean(DCG/normalizer)
    return NDCG


def EvaluateModel(eval_model, eval_gen, eval_func, k):
    '''
        Evaluate the trained model.
    '''
    metric_list = []
    num_list = []
    for (obs_records, unk_true) in eval_gen:
        unk_pred = eval_model.predict_on_batch(obs_records)
        unk_pred[obs_records.astype(np.bool)] = -np.inf
        num_list.append(len(unk_true))
        metric_list.append(eval_func(unk_true, unk_pred, k))
    metric = np.sum(np.array(metric_list) * np.array(num_list)) / np.sum(num_list)
    return metric


def EvaluateCold(eval_model, eval_gen, eval_func, k):
    '''
        Evaluate the trained model.
    '''
    metric_list = []
    num_list = []
    assert eval_gen.use_cold, "Fail to generate cold items"
    num_cold = eval_gen.num_cold
    for (obs_records, unk_true) in eval_gen:
        unk_pred = eval_model.predict_on_batch(obs_records)
        unk_true = unk_true[:, -num_cold:]
        unk_pred = unk_pred[:, -num_cold:]
        num_list.append(len(unk_true))
        metric_list.append(eval_func(unk_true, unk_pred, k=k, num_cold=num_cold))
    metric = np.sum(np.array(metric_list) * np.array(num_list)) / np.sum(num_list)
    return metric