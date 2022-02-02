import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras import backend as K
from tensorflow.python.keras.engine import network

from evaluate import mse
from layers import TransposedSharedDense, MSELoss
from layers import GaussianKLLoss, GaussianReparameterization


class MLP(network.Network):
    '''
        Multilayer Perceptron (MLP). 
    '''
    def __init__(self, 
                 hidden_sizes,
                 activations,
                 l2_normalize=False,
                 dropout_rate=None,
                 **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.m_name = kwargs.get("name", None)
        self.l2_normalize = l2_normalize
        self.dropout_rate = dropout_rate
        self.dense_list = []
        for i, (size, activation) in enumerate(zip(hidden_sizes, activations)):
            self.dense_list.append(
                layers.Dense(size, activation=activation,
                    kernel_initializer='glorot_uniform',
                    bias_initializer='zeros',
                    name="mlp_dense_{}".format(i+1)
            ))

    def build(self, input_shapes):
        x_in = layers.Input(input_shapes[1:])
        h_mid = x_in
        if self.l2_normalize:
            h_mid = layers.Lambda(tf.nn.l2_normalize, arguments={"axis":1})(h_mid)
        if self.dropout_rate:
            h_mid = layers.Dropout(self.dropout_rate)(h_mid)
        h_out = self.dense_list[0](h_mid)
        for dense in self.dense_list[1:]:
            h_out = dense(h_out)
        self._init_graph_network(x_in, h_out, self.m_name)
        super(MLP, self).build(input_shapes)


class SymetricMLP(network.Network):
    '''
        The symetric version of an MLP where the transposed 
        weights the corresponding layersof source MLP are reused.
    '''
    def __init__(self,
                 source_mlp,
                 activations,
                 **kwargs):
        super(SymetricMLP, self).__init__(**kwargs)
        self.m_name = kwargs.get("name", None)
        self.dense_list = []
        for i, (dense_W, dense_b) in enumerate(
            zip(source_mlp.dense_list[-1:0:-1], 
                source_mlp.dense_list[-2::-1])):
            weights = [dense_W.weights[0], dense_b.weights[1]]
            self.dense_list.append(
                TransposedSharedDense(weights=weights,
                activation=activations[i],
                name="sym_mlp_dense_{}".format(i+1)
            ))
        weights = [source_mlp.dense_list[0].weights[0]]
        self.dense_list.append(
            TransposedSharedDense(weights=weights, activation=activations[-1]
        ))

    def build(self, input_shapes):
        x_in = layers.Input(input_shapes[1:])
        h_out = self.dense_list[0](x_in)
        for dense in self.dense_list[1:]:
            h_out = dense(h_out)
        self._init_graph_network(x_in, h_out, name=self.m_name)
        super(SymetricMLP, self).build(input_shapes)


class ContentLatentCore(network.Network):
    '''
        The latent core for the content network.
    '''
    def __init__(self,  
                 latent_size, 
                 out_size, 
                 activation, 
                 **kwargs):
        super(ContentLatentCore, self).__init__(**kwargs)
        self.m_name = kwargs.get("name", None)
        self.dense_mean = layers.Dense(latent_size, name="mean")
        self.dense_std = layers.Dense(latent_size, name="logstd")
        self.z_sampler = GaussianReparameterization(use_logstd=True, name="z_sampler")
        self.dense_out = layers.Dense(out_size, activation=activation, name="latent_out")

    def build(self, input_shapes):
        x_in = layers.Input(input_shapes[1:])
        mean, logstd = self.dense_mean(x_in), self.dense_std(x_in)
        z_t = self.z_sampler([mean, logstd]) 
        y_out = self.dense_out(z_t)
        self._init_graph_network(x_in, [y_out, mean, logstd, z_t], name=self.m_name)
        super(ContentLatentCore, self).build(input_shapes)


class CollaborativeLatentCore(models.Model):
    '''
        The latent core for the collaborative network
    '''
    def __init__(self, 
                 latent_size,
                 **kwargs):
        super(CollaborativeLatentCore, self).__init__(**kwargs)
        self.m_name = kwargs.get("name", None)
        self.dense_mean = layers.Dense(latent_size, name="mean")
        self.dense_std  = layers.Dense(latent_size, name="logstd")
        self.z_sampler = GaussianReparameterization(use_logstd=True, name="z_sampler")

    def build(self, input_shapes):
        x_in = layers.Input(input_shapes[1:])
        mean, logstd = self.dense_mean(x_in), self.dense_std(x_in)
        z_b = self.z_sampler([mean, logstd])
        self._init_graph_network(x_in, [mean, logstd, z_b], name=self.m_name)
        super(CollaborativeLatentCore, self).build(input_shapes)


class LayerwisePretrainableContentVAE():
    '''
        The Layerwise Pretrainable Content VAE
    '''
    def __init__(self, 
                 input_shape,
                 hidden_sizes,
                 encoder_activs,
                 decoder_activs,
                 latent_size,
                 latent_activ):
        self.input_shape = input_shape
        self.latent_size = latent_size
        self.encoder = MLP(hidden_sizes, encoder_activs, name="encoder")
        self.encoder.build(input_shapes=input_shape)
        self.latent_core = ContentLatentCore(latent_size, hidden_sizes[-1], 
                                             latent_activ, name="latent_core")
        self.decoder = SymetricMLP(self.encoder, decoder_activs, name="decoder")
        self.latent_core.build(input_shapes=(None, hidden_sizes[-1]))
        self.decoder.build(input_shapes=(None, hidden_sizes[-1]))

    def build_peri_pretrain(self, layer_idx):
        '''
            Pair the ith layer of encoder and the L-i+1th 
            layer of decoder as an auto-encoder for pretraining
        '''
        depth = len(self.encoder.dense_list)
        assert layer_idx < depth, "index out {} of range {}!".format(layer_idx, depth)
        if not hasattr(self, "peri_pretrains"):
            self.peri_pretrains = {}

        if not layer_idx in self.peri_pretrains.keys():
            src_dense = self.encoder.dense_list[layer_idx]
            sym_dense = self.decoder.dense_list[depth-layer_idx-1]
            x_in = layers.Input(shape=(src_dense.weights[0].shape[0],),
                                name="peri_pretrain_{}_input".format(layer_idx))
            x_rec = sym_dense(src_dense(x_in))
            self.peri_pretrains[layer_idx] = models.Model(inputs=x_in, outputs=x_rec)
        return self.peri_pretrains[layer_idx]

    def build_core_pretrain(self):
        '''
            Get the latent core for pretraining
        '''
        if not hasattr(self, "core_pretrain"):
            x_in = layers.Input(shape=(self.latent_core.weights[0].shape[0],),
                                name="core_pretrain_input")
            x_rec, mean, logstd, _ = self.latent_core(x_in)
            self.core_pretrain = models.Model(inputs=x_in, outputs=x_rec)
            kl_loss = GaussianKLLoss(use_logstd=True)([mean, logstd])
            self.core_pretrain.add_loss(kl_loss)
        return self.core_pretrain

    def build_vae_pretrain(self):
        '''
            Get the whole vae model
        '''
        if not hasattr(self, "vae_pretrain"):
            x_in = layers.Input(shape=self.input_shape[1:], name="contents")
            h_mid = self.encoder(x_in)
            h_mid, mean, logstd, _ = self.latent_core(h_mid)
            x_rec = self.decoder(h_mid)
            self.vae_pretrain = models.Model(inputs=x_in, outputs=x_rec)
            kl_loss = GaussianKLLoss(use_logstd=True)([mean, logstd])
            self.vae_pretrain.add_loss(kl_loss)
        return self.vae_pretrain

    def build_vae_tstep(self, lambda_W, lambda_V, lambda_K=1):
        '''
            Get the content module for the vae model
        '''
        if not hasattr(self, "vae_tstep"):
            x_in = layers.Input(shape=self.input_shape[1:], name="contents")
            z_b = layers.Input(shape=self.latent_size, name="collabo_embeds")
            h_mid = self.encoder(x_in)
            h_mid, mean, logstd, z_t = self.latent_core(h_mid)
            x_rec = self.decoder(h_mid)
            self.vae_tstep = models.Model(inputs=[x_in, z_b], outputs=x_rec)
            kl_loss = GaussianKLLoss(use_logstd=True)([mean, logstd])
            self.vae_tstep.add_loss(kl_loss)
            self.vae_tstep.add_metric(kl_loss, name='kl_loss', aggregation='mean')
            weights_reg_loss = tf.nn.l2_loss(self.vae_tstep.layers[1].dense_list[0].weights[0]) + \
                       tf.nn.l2_loss(self.vae_tstep.layers[1].dense_list[1].weights[0]) + \
                       tf.nn.l2_loss(tf.transpose(self.vae_tstep.layers[1].dense_list[0].weights[0])) + \
                       tf.nn.l2_loss(tf.transpose(self.vae_tstep.layers[1].dense_list[1].weights[0]))
            self.vae_tstep.add_loss(lambda: lambda_W*weights_reg_loss)
            self.vae_tstep.add_metric(lambda_W*weights_reg_loss, name='reg_loss', aggregation='mean')
            #collabo_reg_loss = mse(z_t, z_b)
            collabo_reg_loss = mse(mean, z_b)
            self.vae_tstep.add_metric(lambda_V*collabo_reg_loss, name='collabo_loss', aggregation='mean')
            self.vae_tstep.add_loss(lambda: lambda_V*collabo_reg_loss)
        return self.vae_tstep

    def build_vae_infer_tstep(self):
        '''
            Get the inference part of the vae model
        '''
        if not hasattr(self, "vae_infer_tstep"):
            x_in = layers.Input(shape=self.input_shape[1:], name="contents")
            h_mid = self.encoder(x_in)
            _, mu_t, _, _ = self.latent_core(h_mid)
            self.vae_infer_tstep = models.Model(inputs=x_in, outputs=mu_t)
        return self.vae_infer_tstep

    def load_weights(self, weight_path):
        '''
            Load weights from pretrained vae
        '''
        vae = self.build_vae_pretrain()
        vae.load_weights(weight_path)


class UserOrientedCollarboativeVAE():
    '''
       	The collaborative VAE
    '''
    def __init__(self, 
                 input_dim,
                 hidden_sizes,
                 latent_size,
                 encoder_activs,
                 decoder_activs,
                 lambda_V=1):
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.latent_size = latent_size
        if hidden_sizes:
            self.encoder = MLP(hidden_sizes, activations=encoder_activs, name="Encoder")
        else:
            self.encoder = lambda x:x
        self.latent_core = CollaborativeLatentCore(latent_size, name="LatentCore")
        self.decoder = MLP(hidden_sizes[::-1]+[input_dim], activations=decoder_activs, name="Decoder")
        self.lambda_V = lambda_V

    def build_vae_bstep(self, lambda_W):
        '''
            Get the collaborative module for the vae model
        '''
        self.lambda_W = lambda_W
        if not hasattr(self, "vae_bstep") or self.lambda_W != lambda_W:
            r_in = layers.Input(shape=[self.input_dim,], name="ratings")
            h_mid = self.encoder(r_in)
            mean, logstd, z_u = self.latent_core(h_mid)
            r_rec = self.decoder(z_u)
            self.vae_bstep = models.Model(inputs=r_in, outputs=r_rec)
            self.gaussian_kl_loss = GaussianKLLoss(use_logstd=True)
            kl_loss = self.gaussian_kl_loss([mean, logstd])
            self.vae_bstep.add_loss(kl_loss)
            self.vae_bstep.add_metric(kl_loss, name='kl_loss', aggregation='mean')
            if self.hidden_sizes:
                weights_reg_loss = tf.nn.l2_loss(self.vae_bstep.layers[1].dense_list[0].weights[0]) + \
                                   tf.nn.l2_loss(self.vae_bstep.layers[3].dense_list[0].weights[0])
                self.vae_bstep.add_metric(lambda_W*weights_reg_loss, name='reg_loss', aggregation='mean')
                self.vae_bstep.add_loss(lambda: lambda_W*weights_reg_loss)
            self.mse_loss = MSELoss(shape=(self.input_dim, self.hidden_sizes[0] 
                                           if self.hidden_sizes else self.latent_size))

            if self.hidden_sizes:
                z = self.vae_bstep.layers[1].dense_list[0].weights[0]
            else:
                z = self.vae_bstep.layers[1].dense_mean.weights[0]
            content_reg_loss = self.mse_loss(z)
            self.vae_bstep.add_metric(self.lambda_V*content_reg_loss, name='content_loss', aggregation='mean')
            self.vae_bstep.add_loss(lambda: self.lambda_V*content_reg_loss)
        return self.vae_bstep

    def build_vae_eval(self):
        '''
            For evaluation, use the mean deterministically
        '''
        if not hasattr(self, "vae_eval"):
            r_in = layers.Input(shape=[self.input_dim,], name="ratings")
            h_mid = self.encoder(r_in)
            mu_u = self.latent_core.dense_mean(h_mid)
            r_rec = self.decoder(mu_u)
            self.vae_eval = models.Model(inputs=r_in, outputs=r_rec)
        return self.vae_eval        

    def build_vae_infer_bstep(self):
        '''
            Get the inference part of the vae model
        '''
        if not hasattr(self, "vae_infer_bstep"):
            r_in = layers.Input(shape=[self.input_dim,], name="ratings")
            h_mid = self.encoder(r_in)
            mu_u = self.latent_core.dense_mean(h_mid)
            self.vae_infer_bstep = models.Model(inputs=r_in, outputs=mu_u)
        return self.vae_infer_bstep

    def load_weights(self, weight_path):
        '''
            Load weights from pretrained vae
        '''
        if not hasattr(self, "lambda_W"):
            self.lambda_W = 0
        if not hasattr(self, "lambda_V"):
            self.lambda_V = 0
        vae_model = self.build_vae_bstep(self.lambda_W)
        vae_model.load_weights(weight_path)


    @property
    def embedding_weights(self):
        if self.hidden_sizes != []:
            return self.encoder.dense_list[0].weights[0]
        else:
            return self.latent_core.dense_mean.weights[0]


if __name__ == "__main__":
    pass