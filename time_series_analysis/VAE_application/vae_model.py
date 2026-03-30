# vae_model.py
##### Code to build a Variational Autoencoder #####

import tensorflow as tf
from tensorflow.keras import layers, metrics
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback

##################################################
# Callback to store weights every few epochs

class WeightsSaver(Callback):
    
    def __init__(self, N, title):
        self.N = N
        self.title = title
        
    def on_epoch_end(self, epoch, logs={}):
        if (epoch+1)%self.N == 0:
            name = f'{self.title}_{epoch+1}epochs'
            print("Saving weights ",name)
            self.model.save_weights(name)

##################################################

##########
# SAMPLING LAYER
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    
##########
# ENCODER
def build_encoder(original_dimension, latent_dimension, n_filters, 
                  kernel_initializer=None, bias_initializer=None):

    encoder_inputs = layers.Input((original_dimension,1), name="input")

    # Three layers of 1D convolution + average pooling

    x = layers.Conv1D(name="conv1", filters=n_filters[0], kernel_size=32, strides=2, 
                      activation='relu', padding='same', use_bias=True, 
                      kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(encoder_inputs)
    x = layers.AveragePooling1D(pool_size=2, strides=2, padding='valid')(x)

    x = layers.Conv1D(name="conv2", filters=n_filters[1], kernel_size=32, strides=2, 
                      activation='relu', padding='same', use_bias=True,
                      kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(x)
    x = layers.AveragePooling1D(pool_size=2, strides=2, padding='valid')(x)
   
    x = layers.Conv1D(name="conv3", filters=n_filters[2], kernel_size=32, strides=2, 
                      activation='relu', padding='same', use_bias=True,
                      kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(x)    
    x = layers.AveragePooling1D(pool_size=2, strides=2, padding='valid')(x)    

    # Inner part of the encoder

    x = layers.Flatten(name="flatten")(x)

    z_mean = layers.Dense(latent_dimension, name="z_mean", 
                          kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(x)
    z_log_var = layers.Dense(latent_dimension, name="z_log_var",
                             kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(x)
    z = Sampling()([z_mean, z_log_var])

    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    return encoder


##########
# DECODER 
def build_decoder(original_dimension, latent_dimension, n_filters, 
                  kernel_initializer=None, bias_initializer=None):

    latent_inputs = layers.Input(shape=(latent_dimension,))

    dimension_reduction_pooling = 2*2*2
    dimension_reduction_conv_stride = 2*2*2
    dimension_after_convolutions = original_dimension // (dimension_reduction_pooling * dimension_reduction_conv_stride)

    x = layers.Dense(dimension_after_convolutions * n_filters[2], name="dense_decoded", 
                     activation='relu', use_bias=True, 
                     kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(latent_inputs)
    x = layers.Reshape((dimension_after_convolutions, n_filters[2]), name="reshape")(x)

    # Deconding part, with three transpose 1D convolutional + up-sampling layers

    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1DTranspose(name="deconv3", filters=n_filters[1], kernel_size=32, strides=2, 
                               activation='relu', padding='same', use_bias=True,
                               kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(x)
     
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1DTranspose(name="deconv2", filters=n_filters[0], kernel_size=32, strides=2, 
                               activation='relu', padding='same', use_bias=True, 
                               kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(x)

    x = layers.UpSampling1D(size=2)(x)
    decoder_outputs = layers.Conv1DTranspose(name="deconv1", filters=1, kernel_size=32, strides=2, 
                                             activation='relu', padding='same', use_bias=True, 
                                             kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(x)

    decoder = Model(latent_inputs, decoder_outputs, name="decoder")

    return decoder


##########
# Define the VAE as a `Model` with a custom  `sampling_layer` & `train_step`

class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")

    def call(self, x):
        _, _, z = self.encoder(x)
        return self.decoder(z)
        
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            reconstruction_loss = tf.square(data - reconstruction)
            reconstruction_loss = tf.reduce_sum(reconstruction_loss, axis=[-2,-1])
            reconstruction_loss = tf.reduce_mean(reconstruction_loss, axis=-1)

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_sum(kl_loss, axis=-1)
            kl_loss = tf.reduce_mean(kl_loss, axis=-1)

            total_loss = reconstruction_loss + kl_loss
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.square(data - reconstruction)
            reconstruction_loss = tf.reduce_sum(reconstruction_loss, axis=[-2,-1])
            reconstruction_loss = tf.reduce_mean(reconstruction_loss, axis=-1)

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_sum(kl_loss, axis=-1)
            kl_loss = tf.reduce_mean(kl_loss, axis=-1)

            total_loss = reconstruction_loss + kl_loss
            
        #grads = tape.gradient(total_loss, self.trainable_weights)
        #self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }