import os
import time
import logging
import argparse

import sys
sys.path.append("libs")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K

from data import ContentVaeDataGenerator
from data import CollaborativeVAEDataGenerator
from pretrain_vae import get_content_vae
from train_vae import get_collabo_vae, infer

from evaluate import EvaluateModel
from evaluate import EvaluateCold
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

    ### Fix the random seeds.
    np.random.seed(98765)
    tf.set_random_seed(98765)

    ### Get the train, val data generator for content vae
    if args.lambda_V is not None:
        model_root = os.path.join("models", args.dataset, str(args.split), str(args.lambda_V))
    else:
        model_root = os.path.join("models", args.dataset, str(args.split))

    if args.num_cold is not None:
        data_root = os.path.join("data", args.dataset, str(args.split), str(args.num_cold))
        model_root = os.path.join("models", args.dataset, str(args.split), "num_cold", str(args.num_cold))
    else:
        data_root = os.path.join("data", args.dataset, str(args.split))
        model_root = os.path.join("models", args.dataset, str(args.split))

    dataset = "movielen-10" if "movielen-10" in args.dataset else args.dataset

    tstep_cold_gen = ContentVaeDataGenerator(
        data_root = data_root, joint=True,
        batch_size = args.batch_size, use_cold=True,
    )

    bstep_test_gen = CollaborativeVAEDataGenerator(
        data_root = data_root, phase = "test", 
        batch_size = args.batch_size, shuffle=False
    )

    bstep_cold_gen = CollaborativeVAEDataGenerator(
        data_root = data_root, phase="test",
        batch_size = args.batch_size*8, use_cold=True,
    )

    ### Build test model and load trained weights
    collabo_vae = get_collabo_vae(dataset, bstep_test_gen.num_items)
    collabo_vae.load_weights(os.path.join(model_root, "best_bstep.model"))

    content_vae = get_content_vae(dataset, tstep_cold_gen.feature_dim)
    content_vae.load_weights(os.path.join(model_root, "best_tstep.model")) 
    vae_infer_tstep = content_vae.build_vae_infer_tstep()

    vae_eval = collabo_vae.build_vae_eval()
    vae_eval_cold = collabo_vae.update_vae_coldstart(infer(vae_infer_tstep, tstep_cold_gen.features.A))

    ### Evaluate and save the results
    k4recalls = [10, 20, 25, 30, 35, 40, 45, 50]
    k4ndcgs = [25, 50, 100]
    recalls, NDCGs = [], []
    recalls_cold, NDCGs_cold = [], []
    for k in k4recalls:
        recalls.append("{:.4f}".format(EvaluateModel(vae_eval, bstep_test_gen, Recall_at_k, k=k)))
        recalls_cold.append("{:.4f}".format(EvaluateCold(vae_eval_cold, bstep_cold_gen, Recall_at_k, k=k)))
    for k in k4ndcgs:
        NDCGs.append("{:.4f}".format(EvaluateModel(vae_eval, bstep_test_gen, NDCG_at_k, k=k)))
        NDCGs_cold.append("{:.4f}".format(EvaluateCold(vae_eval_cold, bstep_cold_gen, NDCG_at_k, k=k)))

    recall_table = pd.DataFrame({"k":k4recalls, "recalls":recalls}, columns=["k", "recalls"])
    recall_table.to_csv(os.path.join(model_root, "recalls.csv"), index=False)

    ndcg_table = pd.DataFrame({"k":k4ndcgs, "NDCGs": NDCGs}, columns=["k", "NDCGs"])
    ndcg_table.to_csv(os.path.join(model_root, "NDCGs.csv"), index=False)

    recall_cold_table = pd.DataFrame({"k":k4recalls, "recalls":recalls_cold}, columns=["k", "recalls"])
    recall_cold_table.to_csv(os.path.join(model_root, "recalls_cold.csv"), index=False)

    ndcg_cold_table = pd.DataFrame({"k":k4ndcgs, "NDCGs": NDCGs_cold}, columns=["k", "NDCGs"])
    ndcg_cold_table.to_csv(os.path.join(model_root, "NDCGs_cold.csv"), index=False)

    print("Done evaluation! Results saved to {}".format(model_root))


if __name__ == '__main__':
    predict_and_evaluate()