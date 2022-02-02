import os
import time
import json
import logging
import argparse

import sys
sys.path.append("libs")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K

from data import CollaborativeVAEDataGenerator
from train_vae import get_collabo_vae

from evaluate import EvaluateModel
from evaluate import Recall_at_k, NDCG_at_k


def predict_and_evaluate():
    ### Parse the console arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
        help="specify the dataset for experiment")
    parser.add_argument("--split", type=int,
        help="specify the split of the dataset")
    parser.add_argument("--batch_size", type=int, default=128,
        help="specify the batch size prediction")
    parser.add_argument("--device" , type=str, default="0",
        help="specify the visible GPU device")
    parser.add_argument("--model_root", type=str, default=None,
        help="specify the trained model root (optional)")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    ### Set up the tensorflow session.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    K.set_session(sess)

    ### Fix the random seeds.
    np.random.seed(98765)
    tf.set_random_seed(98765)

    ### Get the test data generator for content vae
    data_root = os.path.join("data", args.dataset, str(args.split))
    if args.model_root:
        model_root = args.model_root
    else:
        model_root = os.path.join("models", args.dataset, str(args.split))
    params_path = os.path.join(model_root, "hyperparams.json")
    with open(params_path, "r") as params_file:
        params = json.load(params_file)

    bstep_test_gen = CollaborativeVAEDataGenerator(
        data_root = data_root, phase = "test", 
        batch_size = args.batch_size, joint=True,
        shuffle=False
    )

    ### Build test model and load trained weights
    collab_vae = get_collabo_vae(params, bstep_test_gen.num_items)
    collab_vae.load_weights(os.path.join(model_root, "best_bstep.model"))
    vae_eval = collab_vae.build_vae_eval()

    ### Evaluate and save the results
    k4recalls = [20, 25, 30, 35, 40, 45, 50]
    k4ndcgs = [50, 100]
    recalls, NDCGs = [], []
    for k in k4recalls:
        recalls.append("{:.4f}".format(EvaluateModel(vae_eval, bstep_test_gen, Recall_at_k, k=k)))
    for k in k4ndcgs:
        NDCGs.append("{:.4f}".format(EvaluateModel(vae_eval, bstep_test_gen, NDCG_at_k, k=k)))

    recall_table = pd.DataFrame({"k":k4recalls, "recalls":recalls}, columns=["k", "recalls"])
    recall_table.to_csv(os.path.join(model_root, "recalls.csv"), index=False)

    ndcg_table = pd.DataFrame({"k":k4ndcgs, "NDCGs": NDCGs}, columns=["k", "NDCGs"])
    ndcg_table.to_csv(os.path.join(model_root, "NDCGs.csv"), index=False)

    print("Done evaluation! Results saved to {}".format(model_root))


if __name__ == '__main__':
    predict_and_evaluate()