import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from tensorflow import keras


class ContentVaeDataGenerator(keras.utils.Sequence):
    '''
        Generate the training and validation data 
        for the content part of vae model.
    '''
    def __init__(self,
                 data_root,
                 batch_size,
                 batch_num=None,
                 prev_layers=[],
                 noise_type=None,
                 joint=False,
                 shuffle=True,
                 use_cold=False):
        feature_path = os.path.join(data_root, "features.npz")
        if use_cold:
            feature_path = os.path.join(data_root, "cold_features.npz")
        self.features = sparse.load_npz(feature_path)
        self.num_items = self.features.shape[0]
        self.batch_size = batch_size
        self.batch_num = batch_num
        if prev_layers != []:
            self.apply_prev_layers(self.features, prev_layers)

        ### Whether or not, or add which type of noise.
        self.noise_type = noise_type

        ### Shuffle the items if necessary.
        self.indexes = np.arange(self.num_items)
        self.shuffle = shuffle 
        if self.shuffle:
            self.on_epoch_end()

        ### Train jointly with the collaborative part
        self.joint = joint

    def __len__(self):
        '''
            The total number of batches.
        '''
        if self.batch_num is None:
            batch_num = self.num_items//self.batch_size
            if self.num_items%self.batch_size != 0:
                batch_num+=1
        else:
            batch_num = self.batch_num
        return batch_num

    def __getitem__(self, i):
        '''
            Return the batch indexed by i.
        '''
        batch_idxes  = self.indexes[i*self.batch_size:(i+1)*self.batch_size]
        batch_target = self.features[batch_idxes].toarray()
        
        if self.noise_type is None:
            batch_input = batch_target
        else:
            batch_input = self.add_noise(self.noise_type, batch_target)

        if self.joint:
            batch_input = [batch_input, self.z_b[batch_idxes]]
            batch_target = batch_target
        return batch_input, batch_target

    def apply_prev_layers(self, features, prev_layers):
        '''
            Apply the previous pretrained layers on the feature
        '''
        batch_num = self.__len__()
        ori_features = features.toarray()
        for prev_layer in prev_layers:
            new_dims = prev_layer.outputs[0].shape.as_list()[-1]
            new_features = np.zeros((self.num_items, new_dims), dtype=np.float32)
            for i in range(batch_num):
                new_features[i*self.batch_size:(i+1)*self.batch_size] = prev_layer(
                    ori_features[i*self.batch_size:(i+1)*self.batch_size]
                )
            ori_features = new_features
        self.features = sparse.csr_matrix(new_features)

    def on_epoch_end(self):
        '''
            Shuffle the item index after each epoch.
        '''
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def add_noise(self, noise_type, contents):
        '''
            corrupt the inputs and train as SDAE style.
        '''
        if 'Mask' in noise_type:
            frac = float(noise_type.split('-')[1])
            masked_contents = np.copy(contents)
            for item in masked_contents:
                zero_pos = np.random.choice(len(item), int(round(
                    frac*len(item))), replace=False)
                item[zero_pos] = 0
            return masked_contents
        else:
            raise NotImplementedError

    def update_previous_bstep(self, z_b):
        self.z_b = z_b

    @property
    def feature_dim(self):
        return self.features.shape[-1]


class CollaborativeVAEDataGenerator(keras.utils.Sequence):
    def __init__(self, 
                 data_root,
                 phase,
                 batch_size,
                 batch_num=None,
                 reuse=True,
                 shuffle=True,
                 use_cold=False):
        '''
            Generate the training and validation data 
            for the collaborative part of vbae model.
        '''
        assert phase in ["train", "val", "test"], "Phase must be [train, val, test]"
        self.phase = phase
        self.batch_size = batch_size
        self.batch_num = batch_num

        self.data_root = data_root
        self._load_data(data_root)
        self.indexes = np.arange(self.num_users)

        self.shuffle = shuffle
        if self.shuffle:
            self.on_epoch_end()

        ### Whether or not to include cold start items
        self.use_cold = use_cold
        if self.use_cold:
            self._load_cold_data(self.data_root)
            self._merge_hot_and_cold()

    def _load_data(self, data_root):
        ### Load the dataset
        X_path = os.path.join(data_root, "{}_X.npz".format(self.phase))
        Y_path = os.path.join(data_root, "{}_Y.npz".format(self.phase))

        self.X = sparse.load_npz(X_path)
        self.Y = sparse.load_npz(Y_path)

        self.num_users = self.X.shape[0]
        self.num_items = self.X.shape[1]

    def _load_cold_data(self, data_root):
        cold_Y_path = os.path.join(data_root, "cold_{}_Y.npz".format(self.phase))
        self.cold_Y = sparse.load_npz(cold_Y_path)
        self.num_cold = self.cold_Y.shape[1]

    def _merge_hot_and_cold(self):
        self.cold_X = sparse.csr_matrix(self.cold_Y.shape)
        self.X = sparse.hstack([self.X, self.cold_X]).tocsr()
        self.Y = sparse.hstack([self.Y, self.cold_Y]).tocsr()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        if self.batch_num is None:
            batch_num = self.num_users//self.batch_size
            if self.num_users%self.batch_size != 0:
                batch_num+=1
        else:
            batch_num = self.batch_num
        return batch_num

    def __getitem__(self, i):
        batch_idxes = self.indexes[i*self.batch_size:(i+1)*self.batch_size]
        batch_X = self.X[batch_idxes].toarray()
        batch_Y =  self.Y[batch_idxes].toarray()
        return (batch_X, batch_Y)

    def update_previous_tstep(self, z_t):
        self.z_t = z_t

    @property
    def target_shape(self):
        return self._target_shape
    

if __name__ == '__main__':
    pass