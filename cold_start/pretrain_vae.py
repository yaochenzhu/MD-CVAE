import os
import time
import json
import logging
import argparse

import sys
sys.path.append("libs")
from utils import Init_logging

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K

from data import ContentVaeDataGenerator
from model import LayerwisePretrainableContentVAE
from evaluate import binary_crossentropy

import warnings
warnings.filterwarnings('ignore')

### Fix the random seeds.
np.random.seed(98765)
tf.set_random_seed(98765)

citeulike_args = {
    "hidden_sizes" : [150, 150],
    "encoder_activs" : ["sigmoid", "sigmoid"],
    "decoder_activs" : ["sigmoid", "sigmoid"],
    "latent_size" : 150,
    "latent_activ" : "sigmoid",
}

movielen_args = {
    "hidden_sizes" : [100, 100],
    "encoder_activs" : ["sigmoid", "sigmoid"],
    "decoder_activs" : ["sigmoid", "sigmoid"],
    "latent_size" : 100,
    "latent_activ" : "sigmoid",
}


name_args_dict = {
    "citeulike-a" : citeulike_args,
    "movielen-10" : movielen_args
}

name_loss_dict = {
    "citeulike-a" : binary_crossentropy,
    "movielen-10" : binary_crossentropy,
}

def get_content_vae(dataset, feature_dim):
    content_vae = LayerwisePretrainableContentVAE(
        input_shape = (None, feature_dim),
        **name_args_dict[dataset]
    )
    return content_vae

if __name__ == '__main__':
    ### Parse the console arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, \
        help="specify the dataset for experiment")
    parser.add_argument("--split", type=int, default=0,
        help="specify the split of dataset for experiment")
    parser.add_argument("--batch_size", type=int, default=500,
        help="specify the batch size for pretraining")
    parser.add_argument("--device", type=str, default="0",
        help="specify the visible GPU device")
    parser.add_argument("--num_cold", type=int, default=None)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    ### Set up the tensorflow session.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    K.set_session(sess)
    
    Init_logging()

    ### Get the initial train, val data generator.
    if args.num_cold:
        data_root = os.path.join("data", args.dataset, str(args.split), str(args.num_cold))
    else:
        data_root = os.path.join("data", args.dataset, str(args.split))

    print("Load data from {}".format(data_root))
    dataset = "movielen-10" if "movielen-10" in args.dataset else args.dataset

    prev_layers = []
    train_gen = ContentVaeDataGenerator(
        data_root = data_root,
        batch_size = args.batch_size,
        prev_layers = prev_layers,
        noise_type = "Mask-0.3", shuffle = True
    )

    ### Get the layerwise pretrainable vae model.
    content_vae = get_content_vae(dataset, train_gen.feature_dim)

    loss = name_loss_dict[dataset]
    hidden_sizes = name_args_dict[dataset]["hidden_sizes"]
    ### Pretrain the peripheral layer pairs.
    pre_epochs, fin_epochs = 50, 100

    for i in range(len(hidden_sizes)):
        logging.info("Pretrain the {}th peripheral layer pair!".format(i+1))
        peri_pretrain = content_vae.build_peri_pretrain(i)
        peri_pretrain.compile(optimizer=optimizers.Adam(lr=0.01), loss=loss)
        peri_pretrain.fit(train_gen, epochs=pre_epochs)

        prev_layer = content_vae.encoder.dense_list[i]
        prev_layers.append(K.function(prev_layer.input, prev_layer.output))

        '''
            Get the new data generator where the inputs are 
            processed by previous pretrained layers.
        '''
        train_gen = ContentVaeDataGenerator(
            data_root = data_root,
            batch_size = args.batch_size,
            prev_layers = prev_layers,
            shuffle = True
        )

    ### Pretrain the latent core layer.
    logging.info("Pretrain the latent core layer!")
    core_pretrain = content_vae.build_core_pretrain()
    core_pretrain.compile(optimizer=optimizers.Adam(lr=0.01), loss=loss)    
    core_pretrain.fit(train_gen, epochs=pre_epochs)

    ### Pretrain the whole vae model.
    logging.info("Pretrain the whole vae model!")
    prev_layers = []
    train_gen = ContentVaeDataGenerator(
        data_root = data_root,
        batch_size = args.batch_size,
        prev_layers = prev_layers,
        shuffle = True
    )

    vae_pretrain = content_vae.build_vae_pretrain()
    vae_pretrain.compile(optimizer=optimizers.Adam(lr=0.01), loss=loss, metrics=[loss])
    vae_pretrain.fit(train_gen, epochs=fin_epochs)

    ### Save the pretrained weights.
    if args.num_cold:
        save_root = os.path.join("models", args.dataset, str(args.split), "num_cold", \
            str(args.num_cold), "pretrained")
    else:
        save_root = os.path.join("models", args.dataset, str(args.split), "pretrained")

    if not os.path.exists(save_root):
        os.makedirs(save_root)
    vae_pretrain.save_weights(os.path.join(save_root, "weights.model"), save_format="tf")

    with open(os.path.join(save_root, "hyper_pretrain.json"), "w") as f:
        json.dump(name_args_dict[dataset], f)
    print("model saved to {}".format(save_root))