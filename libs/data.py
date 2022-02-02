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
                 shuffle=True):

        feature_path = os.path.join(data_root, "item_features.npz")
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
                 ragged_x=False,                 
                 reuse=True,
                 joint=True,
                 shuffle=True):
        '''
            Generate the training and validation data 
            for the collaborative part of vbae model.
        '''
        assert phase in ["train", "val", "test"], "Phase must be [train, val, test]"
        self.phase = phase
        self.batch_size = batch_size
        self.batch_num = batch_num
        self.ragged_x = ragged_x

        self.data_root = data_root
        self._load_data(data_root, reuse=reuse, ragged_x=ragged_x)

        self.shuffle = shuffle
        if self.shuffle:
            self.on_epoch_end()

        ### Train jointly with the content part
        self.joint = joint

    def _load_data(self, data_root, reuse, ragged_x):
        ### Load the dataset
        meta_table = pd.read_csv(os.path.join(data_root, "meta.csv"))
        self.num_items = meta_table["num_items"][0]

        if self.phase == "train":
            obs_path = os.path.join(data_root, "train.csv")
            obs_records = pd.read_csv(obs_path)
            obs_group = obs_records.groupby("uid")
            unk_group = obs_group
        else:
            obs_path = os.path.join(data_root, "{}_obs.csv".format(self.phase))
            unk_path = os.path.join(data_root, "{}_unk.csv".format(self.phase))
            obs_records = pd.read_csv(obs_path)
            unk_records = pd.read_csv(unk_path)
            obs_group = obs_records.groupby("uid")
            unk_group = unk_records.groupby("uid")

        ### IDs and corresponding indexes
        self.user_ids = np.array(pd.unique(obs_records["uid"]), dtype=np.int32)
        self.indexes = np.arange(len(self.user_ids))
        self.num_users = len(self.user_ids)

        X_path = os.path.join(data_root, "{}_X.npz".format(self.phase))
        Y_path = os.path.join(data_root, "{}_Y.npz".format(self.phase))

        if reuse and os.path.exists(X_path) and os.path.exists(Y_path):
            self.X = sparse.load_npz(X_path)
            self.Y = sparse.load_npz(Y_path)
        else:
            ### Represent the whole dataset with a huge sparse matrix
            rows_X, cols_X, rows_Y, cols_Y = [], [], [], []
            for i, user_id in enumerate(self.user_ids):
                try:
                    group_X = obs_group.get_group(user_id)
                    group_Y = unk_group.get_group(user_id)
                except:
                    import pdb
                    pdb.set_trace()
                rows_X += [i]*len(group_X); cols_X += list(group_X["vid"]-1)
                rows_Y += [i]*len(group_Y); cols_Y += list(group_Y["vid"]-1)

            self.X = sparse.csr_matrix((np.ones_like(rows_X, dtype=np.float32),
                                       (rows_X, cols_X)), dtype='float32',
                                       shape=(self.num_users, self.num_items))
            
            self.Y = sparse.csr_matrix((np.ones_like(rows_Y, dtype=np.float32),
                                       (rows_Y, cols_Y)), dtype='float32',
                                       shape=(self.num_users, self.num_items))
            if reuse:
                sparse.save_npz(X_path, self.X)
                sparse.save_npz(Y_path, self.Y)

        if ragged_x:
            self.X = self.X.tolil().rows

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
        batch_X = self.X[batch_idxes] if self.ragged_x \
            else self.X[batch_idxes].toarray()
        batch_Y =  self.Y[batch_idxes].toarray()
        return (batch_X, batch_Y)

    def update_previous_tstep(self, z_t):
        self.z_t = z_t

    @property
    def target_shape(self):
        return self._target_shape
    

if __name__ == '__main__':
    pass