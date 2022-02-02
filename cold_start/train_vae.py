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
from model import SymetricUserOrientedCollarboativeVAE
from pretrain_vae import get_content_vae

from evaluate import EvaluateModel, EvaluateCold, mse
from evaluate import Recall_at_k, NDCG_at_k
from evaluate import binary_crossentropy
from evaluate import multinomial_crossentropy

import warnings
warnings.filterwarnings('ignore')

### Fix the random seeds.
np.random.seed(98765)
tf.set_random_seed(98765)

class Params():
    def __init__(self, W, V):
        self.lambda_W = W
        self.lambda_V = V

citeulike_a_args = {
    "hidden_sizes":[], 
    "latent_size":150,
    "encoder_activs" : ["tanh"],
    "decoder_activs" : ["softmax"],
}

movielen_10_args = {
    "hidden_sizes":[100], 
    "latent_size":50,
    "encoder_activs" : ["tanh"],
    "decoder_activs" : ["tanh", "softmax"],
}

name_args_dict = {
    "citeulike-a"  : citeulike_a_args,
    "movielen-10"  : movielen_10_args,
}

name_loss_dict = {
    "citeulike-a"  : binary_crossentropy,
    "movielen-10"  : binary_crossentropy
}

def get_collabo_vae(dataset, input_dim):
    collabo_vae = SymetricUserOrientedCollarboativeVAE(
        input_dim = input_dim,
        **name_args_dict[dataset],
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
    parser.add_argument("--summary", default=False, action="store_true",
        help="whether or not write summaries to the results")
    parser.add_argument("--lambda_V", default=None, type=int,
        help="specify the value of lambda_V for regularization")
    parser.add_argument("--num_cold", default=None, type=int,
        help="specify the number of cold start items")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    ### Set up the tensorflow session.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    K.set_session(sess)

    ### Get the train, val data generator for content vae
    if args.num_cold:
        data_root = os.path.join("data", args.dataset, str(args.split), str(args.num_cold))
    else:
        data_root = os.path.join("data", args.dataset, str(args.split))
    dataset = "movielen-10" if "movielen-10" in args.dataset else args.dataset

    tstep_train_gen = ContentVaeDataGenerator(
        data_root = data_root, joint=True,
        batch_size = args.batch_size, 
    )
    tstep_cold_gen = ContentVaeDataGenerator(
        data_root = data_root, joint=True,
        batch_size = args.batch_size, use_cold=True,
    )

    ### Get the train, val data generator for vae
    bstep_train_gen = CollaborativeVAEDataGenerator(
        data_root = data_root, phase="train",
        batch_size = args.batch_size,
    )
    bstep_valid_gen = CollaborativeVAEDataGenerator(
        data_root = data_root, phase="val",
        batch_size = args.batch_size*8,
    )
    bstep_cold_gen = CollaborativeVAEDataGenerator(
        data_root = data_root, phase="val",
        batch_size = args.batch_size*8, use_cold=True,
    )

    blr_schedule = PiecewiseSchedule([[0, 1e-3], [50, 1e-3], [51, 1e-4]], outside_value=1e-4)
    tlr_schedule = PiecewiseSchedule([[0, 1e-3], [50, 1e-3], [51, 1e-4]], outside_value=1e-4)

    ### Build the t and b step vae model
    if args.num_cold:
        weight_path = os.path.join("models", args.dataset, str(args.split), "num_cold", \
            str(args.num_cold), "pretrained", "weights.model")
    else:
        weight_path = os.path.join("models", args.dataset, str(args.split), "pretrained", "weights.model")
    print("pretrained model load from: {}".format(weight_path))

    content_vae = get_content_vae(dataset, tstep_train_gen.feature_dim)
    collabo_vae = get_collabo_vae(dataset, input_dim=bstep_train_gen.num_items)
    content_vae.load_weights(weight_path)

    if args.lambda_V is not None:
        print("Use user-specified lambda {}".format(args.lambda_V))
        lambda_V = args.lambda_V
        use_default_lambda = False
    else:
        if args.dataset == "citeulike-a":
            lambda_V = 50
        elif "movielen-10" in args.dataset:
            lambda_V = 75
        print("Use default lambda {}".format(lambda_V))
        use_default_lambda = True

    if args.num_cold is None:
        use_default_cold = True
    else:
        use_default_cold = False
    
    params = Params(W=2e-4, V=lambda_V)
    vae_bstep = collabo_vae.build_vae_bstep(lambda_W=params.lambda_W, lambda_V=params.lambda_V)
    vae_tstep = content_vae.build_vae_tstep(lambda_W=params.lambda_W, lambda_V=params.lambda_V)

    sess.run(tf.global_variables_initializer())

    vae_infer_tstep = content_vae.build_vae_infer_tstep()
    vae_infer_bstep = collabo_vae.build_vae_infer_bstep()
    vae_eval = collabo_vae.build_vae_eval()
    vae_eval_cold = collabo_vae.update_vae_coldstart(infer(vae_infer_tstep, tstep_cold_gen.features.A))

    ### Some configurations for training
    best_Recall_20, best_Recall_40, best_NDCG_100, best_sum = -np.inf, -np.inf, -np.inf, -np.inf
    best_Recall_20_cold, best_Recall_40_cold, best_NDCG_100_cold = -np.inf, -np.inf, -np.inf

    if use_default_lambda:
        save_root = os.path.join("models", args.dataset, str(args.split))
    else:
        save_root = os.path.join("models", args.dataset, str(args.split), str(lambda_V))

    if use_default_cold:
        save_root = os.path.join("models", args.dataset, str(args.split))
    else:
        save_root = os.path.join("models", args.dataset, str(args.split), "num_cold", str(args.num_cold))

    if not os.path.exists(save_root):
        os.makedirs(save_root)
    with open(os.path.join(save_root, "hyper.txt"), "w") as f:
        json.dump(name_args_dict[dataset], f)
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    training_dynamics = os.path.join(save_root, "training_dynamics.csv")
    with open(training_dynamics, "w") as f:
        f.write("Recall@20,Recall@40,NDCG@100\n")

    best_bstep_path = os.path.join(save_root, "best_bstep.model")
    best_tstep_path = os.path.join(save_root, "best_tstep.model")

    lamb_schedule_gauss = PiecewiseSchedule([[0, 0.0], [80, 0.2]], outside_value=0.2)
    vae_bstep.compile(loss=multinomial_crossentropy, optimizer=optimizers.Adam(), 
                      metrics=[multinomial_crossentropy])
    vae_tstep.compile(optimizer=optimizers.Adam(), loss=name_loss_dict[dataset])

    ### Train the content and collaborative part of vae in an EM-like style
    epochs = 200
    mix_in_epochs = 30
    bstep_tsboard = callbacks.TensorBoard(log_dir=save_root)

    for epoch in range(epochs):
        print("-"*10 + "Epoch:{}".format(epoch), "-"*10)

        print("Begin bstep:")
        K.set_value(vae_bstep.optimizer.lr, blr_schedule.value(epoch))
        K.set_value(collabo_vae.gaussian_kl_loss.lamb_kl, lamb_schedule_gauss.value(epoch))
        K.set_value(collabo_vae.mse_loss.targets, infer(vae_infer_tstep, tstep_train_gen.features.A))

        his_bstep = vae_bstep.fit_generator(bstep_train_gen, epochs=1, workers=4,
                                            validation_data=bstep_valid_gen)

        Recall_20 = EvaluateModel(vae_eval, bstep_valid_gen, Recall_at_k, k=20)
        Recall_40 = EvaluateModel(vae_eval, bstep_valid_gen, Recall_at_k, k=40)
        NDCG_100 = EvaluateModel(vae_eval, bstep_valid_gen, NDCG_at_k, k=100)

        if epoch > mix_in_epochs:
            Recall_20_cold = EvaluateCold(vae_eval_cold, bstep_cold_gen, Recall_at_k, k=20)
            Recall_40_cold = EvaluateCold(vae_eval_cold, bstep_cold_gen, Recall_at_k, k=40)
            NDCG_100_cold = EvaluateCold(vae_eval_cold, bstep_cold_gen, NDCG_at_k, k=100)
        
            if Recall_20_cold >  best_Recall_20_cold:
                best_Recall_20_cold = Recall_20_cold

            if Recall_40_cold >  best_Recall_40_cold:
                best_Recall_40_cold = Recall_40_cold

            if NDCG_100_cold > best_NDCG_100_cold:
                best_NDCG_100_cold = NDCG_100_cold

        if args.summary:
            logs = his_bstep.history
            logs.update({"val_recall_20":Recall_20,
                         "val_recall_40":Recall_40,
                         "val_ndcg_100":NDCG_100})
            if epoch > mix_in_epochs:
                logs.update({"val_cold_recall_20": Recall_20_cold,
                             "val_cold_recall_40": Recall_40_cold,
                             "val_cold_ndcg_100" : NDCG_100_cold})
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

        if epoch > mix_in_epochs:
            print("-"*5 + "Cold start items" + "-"*5)
            print("cold recall@20: {:5f}, best cold recall@20: {:5f}".format(Recall_20_cold, best_Recall_20_cold))
            print("cold recall@40: {:5f}, best cold recall@40: {:5f}".format(Recall_40_cold, best_Recall_40_cold))
            print("cold NDCG@100: {:5f}, best cold NDCG@100: {:5f}".format(NDCG_100_cold, best_NDCG_100_cold))

        print("Begin tstep:")
        K.set_value(vae_tstep.optimizer.lr, tlr_schedule.value(epoch))
        tstep_train_gen.update_previous_bstep(K.get_value(collabo_vae.embedding_weights))
        vae_tstep.fit_generator(tstep_train_gen, workers=4, epochs=1)
        vae_eval_cold = collabo_vae.update_vae_coldstart(infer(vae_infer_tstep, tstep_cold_gen.features.A))

    print("Done training!")

if __name__ == '__main__':
    train_vae_model()