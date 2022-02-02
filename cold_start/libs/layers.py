import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import backend as K

from tensorflow_probability import distributions as tfp


class TransposedSharedDense(layers.Layer):
    '''
        Dense layer that shares weights (transposed) and bias 
        with another dense layer.
    '''

    def __init__(self, weights, activation=None, use_bias=False, **kwargs):
        super(TransposedSharedDense, self).__init__(**kwargs)
        assert(len(weights) in [1, 2]), \
            "Specify the [kernel] or the [kernel] and [bias]."
        self.W = weights[0]
        self.use_bias = use_bias
        if self.use_bias:
            if len(weights) == 1:
                b_shape = self.W.shape.as_list()[0]
                self.b = self.add_weight(shape=(b_shape),
                    name="bias",
                    trainable=True,
                    initializer="zeros")
            else:
                self.b = weights[1]
        self.activate = activations.get(activation)

    def call(self, inputs):
        if self.use_bias:
            return self.activate(K.dot(inputs, K.transpose(self.W))+self.b)
        else:
            return self.activate(K.dot(inputs, K.transpose(self.W)))


class GaussianReparameterization(layers.Layer):
    '''
        Rearameterization trick for Gaussian
    '''
    def __init__(self, use_logstd=True, **kwargs):
        super(GaussianReparameterization, self).__init__(**kwargs)
        self.use_logstd = use_logstd
        if self.use_logstd:
            self.clip = layers.Lambda(lambda x:K.clip(x, -20, 2), name="clip")
            self.exp = layers.Lambda(lambda x:K.exp(x), name="exp")

    def call(self, stats):
        mu, std = stats
        if self.use_logstd:
            std = self.exp(self.clip(std))
        dist = tfp.MultivariateNormalDiag(loc=mu, scale_diag=std)
        return dist.sample()


class GaussianKLLoss(layers.Layer):
    '''
        Add the KL divergence between the variational 
        Gaussian distribution and the prior to loss.
    '''
    def __init__(self, use_logstd=True, **kwargs):
        super(GaussianKLLoss, self).__init__(**kwargs)
        self.lamb_kl = self.add_weight(shape=(), 
                                       name="lamb_kl", 
                                       initializer="ones", 
                                       trainable=False)
        self.use_logstd = use_logstd
        if self.use_logstd:
            self.clip = layers.Lambda(lambda x:K.clip(x, -20, 2), name="clip")
            self.exp = layers.Lambda(lambda x:K.exp(x), name="exp")

    def call(self, inputs):
        mu, std  = inputs
        if self.use_logstd:
            std = self.exp(self.clip(std))
        var_dist = tfp.MultivariateNormalDiag(loc=mu, scale_diag=std)
        pri_dist = tfp.MultivariateNormalDiag(loc=K.zeros_like(mu), 
                                              scale_diag=K.ones_like(std))    
        kl_loss  = self.lamb_kl*K.mean(tfp.kl_divergence(var_dist, pri_dist))
        return kl_loss


class MSELoss(layers.Layer):
    '''
        Add the mse loss to the weights of a layer
    '''
    def __init__(self, shape, **kwargs):
        super(MSELoss, self).__init__(**kwargs)
        self.targets = self.add_weight(shape=shape, 
                                       name="mse_targets", 
                                       initializer="zeros", 
                                       trainable=False)

    def call(self, inputs):
        mse_loss = tf.reduce_mean(tf.reduce_sum(
            tf.square(inputs - self.targets), axis=-1))
        return mse_loss