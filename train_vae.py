import os
import time
import json
import logging
import argparse

import sys
sys.path.append("libs")
from utils import Init_logging
from utils import PiecewiseSchedule

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.summary import summary as tf_summary

from data import ContentVaeDataGenerator
from data import CollaborativeVAEDataGenerator
from model import UserOrientedCollarboativeVAE
from pretrain_vae import get_content_vae

from evaluate import EvaluateModel, mse
from evaluate import Recall_at_k, NDCG_at_k
from evaluate import binary_crossentropy
from evaluate import multinomial_crossentropy

import warnings
warnings.filterwarnings('ignore')

### Fix the random seeds.
np.random.seed(98765)
tf.set_random_seed(98765)

class Params():
    def __init__(self, W):
        self.lambda_W = W

citeulike_a_args = {
    "hidden_sizes": [], 
    "encoder_activs": [], 
    "decoder_activs": ["softmax"], 
    "latent_size": 150, 
    "lambda_V": 2
}

movielen_10_args = {
    "hidden_sizes":[100], 
    "latent_size":50,
    "encoder_activs" : ["tanh"],
    "decoder_activs" : ["tanh", "softmax"],
    "lambda_V" : 10
}

name_args_dict = {
    "citeulike-a"  : citeulike_a_args,
    "movielen-10"  : movielen_10_args,
}

name_loss_dict = {
    "citeulike-a"  : binary_crossentropy,
    "movielen-10"  : binary_crossentropy
}


def get_collabo_vae(params, input_dim):
    collabo_vae = UserOrientedCollarboativeVAE(
        input_dim = input_dim,
        **params,
    )
    return collabo_vae


def infer(infer_model, inputs, batch_size=2000):
    num_samples = len(inputs)
    z_size = infer_model.output.shape.as_list()[-1]
    z_infer = np.zeros((num_samples, z_size), dtype=np.float32)
    for i in range(num_samples//batch_size+1):
        z_infer[i*batch_size:(i+1)*batch_size] \
             = infer_model.predict_on_batch(inputs[i*batch_size:(i+1)*batch_size])
    return z_infer


def summary(save_root, logs, epoch):
    save_train = os.path.join(save_root, "train")
    save_val = os.path.join(save_root, "val")
    if not os.path.exists(save_train):
        os.makedirs(save_train)
    if not os.path.exists(save_val):
        os.makedirs(save_val)
    writer_train = tf.summary.FileWriter(save_train)
    writer_val = tf.summary.FileWriter(save_val)
    for metric, value in logs.items():
        if isinstance(value, list):
            value = value[0]
        summary = tf_summary.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        if "val" in metric:
            summary_value.tag = metric[4:]
            writer_val.add_summary(summary, epoch)
        else:
            summary_value.tag = metric
            writer_train.add_summary(summary, epoch)
    writer_val.flush(); writer_val.flush()


def train_vae_model():
    ### Parse the console arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="citeulike-a",\
        help="specify the dataset for experiment")
    parser.add_argument("--split", type=int, default=0,
        help="specify the split of dataset for experiment")
    parser.add_argument("--batch_size", type=int, default=256,
        help="specify the batch size for updating vae")
    parser.add_argument("--device" , type=str, default="0",
        help="specify the visible GPU device")
    parser.add_argument("--pretrain_root", type=str, default=None,
        help="specify the root for pretrained model (optional)")
    parser.add_argument("--param_path", type=str, default=None,
        help="specify the path of hyper parameter (if any)")
    parser.add_argument("--save_root", type=str, default=None,
        help="specify the prefix for save root (if any)")
    parser.add_argument("--summary", default=False, action="store_true",
        help="whether or not write summaries to the results")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    ### Set up the tensorflow session.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    K.set_session(sess)

    ### Get the train, val data generator for content vae
    data_root = os.path.join("data", args.dataset, str(args.split))
    dataset = "movielen-10" if "movielen-10" in args.dataset else args.dataset
    tstep_train_gen = ContentVaeDataGenerator(
        data_root = data_root, joint=True,
        batch_size = args.batch_size, 
    )

    ### Get the train, val data generator for vae
    bstep_train_gen = CollaborativeVAEDataGenerator(
        data_root = data_root, phase="train",
        batch_size = args.batch_size, joint=True,
    )
    bstep_valid_gen = CollaborativeVAEDataGenerator(
        data_root = data_root, phase="val",
        batch_size = args.batch_size*8, joint=True,
    )

    blr_schedule = PiecewiseSchedule([[0, 1e-3], [50, 1e-3], [51, 1e-4]], outside_value=1e-4)
    tlr_schedule = PiecewiseSchedule([[0, 1e-3], [50, 1e-3], [51, 1e-4]], outside_value=1e-4)

    ### Build the t and b step vae model
    if not args.pretrain_root:
        pretrain_root = os.path.join("models", args.dataset, str(args.split), "pretrained")
    else:
        pretrain_root = args.pretrain_root

    pretrain_weight_path = os.path.join(pretrain_root, "weights.model")
    pretrain_params_path = os.path.join(pretrain_root, "hyperparams.json")
    with open(pretrain_params_path, "r") as param_file:
        pretrain_params = json.load(param_file)
    content_vae = get_content_vae(pretrain_params, tstep_train_gen.feature_dim)

    if args.param_path is not None:
        try:
            with open(args.param_path, "r") as param_file:
                train_params = json.load(param_file)
        except:
            print("Fail to load hyperparams from file, use default instead!")
            train_params = name_args_dict[dataset]
    else:
        train_params = name_args_dict[dataset]

    collabo_vae = get_collabo_vae(train_params, input_dim=bstep_train_gen.num_items)
    content_vae.load_weights(pretrain_weight_path)
    
    params = Params(W=2e-4)
    vae_bstep = collabo_vae.build_vae_bstep(lambda_W=params.lambda_W)
    vae_tstep = content_vae.build_vae_tstep(lambda_W=params.lambda_W, lambda_V=train_params["lambda_V"])

    sess.run(tf.global_variables_initializer())

    vae_infer_tstep = content_vae.build_vae_infer_tstep()
    vae_infer_bstep = collabo_vae.build_vae_infer_bstep()
    vae_eval = collabo_vae.build_vae_eval()

    ### Some configurations for training
    best_Recall_20, best_Recall_40, best_NDCG_100, best_sum = -np.inf, -np.inf, -np.inf, -np.inf
    if args.save_root:
        save_root = args.save_root
    else:
        save_root = os.path.join("models", args.dataset, str(args.split))
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    with open(os.path.join(save_root, "hyperparams.json"), "w") as f:
        json.dump(train_params, f)
    training_dynamics = os.path.join(save_root, "training_dynamics.csv")
    with open(training_dynamics, "w") as f:
        f.write("Recall@20,Recall@40,NDCG@100\n")

    best_bstep_path = os.path.join(save_root, "best_bstep.model")
    best_tstep_path = os.path.join(save_root, "best_tstep.model")

    lamb_schedule_gauss = PiecewiseSchedule([[0, 0.0], [80, 0.2]], outside_value=0.2)
    vae_bstep.compile(loss=multinomial_crossentropy, optimizer=optimizers.Adam(), metrics=[multinomial_crossentropy])
    vae_tstep.compile(optimizer=optimizers.Adam(), loss=name_loss_dict[dataset] \
        if dataset in name_loss_dict.keys() else binary_crossentropy)

    ### Train the content and collaborative part of vae in an EM-like style
    epochs = 100
    bstep_tsboard = callbacks.TensorBoard(log_dir=save_root)

    for epoch in range(epochs):
        print("-"*10 + "Epoch:{}".format(epoch), "-"*10)

        print("Begin bstep:")
        K.set_value(vae_bstep.optimizer.lr, blr_schedule.value(epoch))
        K.set_value(collabo_vae.gaussian_kl_loss.lamb_kl, lamb_schedule_gauss.value(epoch))
        K.set_value(collabo_vae.mse_loss.targets, infer(vae_infer_tstep, tstep_train_gen.features.A))

        his_bstep = vae_bstep.fit_generator(bstep_train_gen, epochs=1, workers=4, validation_data=bstep_valid_gen)
        Recall_20 = EvaluateModel(vae_eval, bstep_valid_gen, Recall_at_k, k=20)
        Recall_40 = EvaluateModel(vae_eval, bstep_valid_gen, Recall_at_k, k=40)
        NDCG_100 = EvaluateModel(vae_eval, bstep_valid_gen, NDCG_at_k, k=100)
        if args.summary:
            logs = his_bstep.history
            logs.update({"val_recall_20":Recall_20,
                         "val_recall_40":Recall_40,
                         "val_ndcg_100":NDCG_100})
            summary(save_root, logs, epoch)

        if Recall_20 > best_Recall_20:
            best_Recall_20 = Recall_20

        if Recall_40 > best_Recall_40:
            best_Recall_40 = Recall_40

        if NDCG_100 > best_NDCG_100:
            best_NDCG_100 = NDCG_100

        cur_sum = Recall_20 + Recall_40 + NDCG_100
        if cur_sum > best_sum:
            best_sum = cur_sum
            vae_bstep.save_weights(best_bstep_path, save_format="tf")
            vae_tstep.save_weights(best_tstep_path, save_format="tf")

        with open(training_dynamics, "a") as f:
            f.write("{:.4f},{:.4f},{:.4f}\n".\
                format(Recall_20, Recall_40, NDCG_100))

        print("-"*5+"Epoch: {}".format(epoch)+"-"*5)
        print("cur recall@20: {:5f}, best recall@20: {:5f}".format(Recall_20, best_Recall_20))
        print("cur recall@40: {:5f}, best recall@40: {:5f}".format(Recall_40, best_Recall_40))
        print("cur NDCG@100: {:5f}, best NDCG@100: {:5f}".format(NDCG_100, best_NDCG_100))

        print("Begin tstep:")
        K.set_value(vae_tstep.optimizer.lr, tlr_schedule.value(epoch))
        tstep_train_gen.update_previous_bstep(K.get_value(collabo_vae.embedding_weights))
        vae_tstep.fit_generator(tstep_train_gen, workers=4, epochs=1)
    print("Done training!")

if __name__ == '__main__':
    train_vae_model()