"""
vae_model.py

Architecture and training logic for a 1D Convolutional Variational Autoencoder (VAE)
designed for latent space studies in time-series waveforms.

The model components:
    - Sampling      : reparameterisation-trick layer
    - build_encoder : returns a Keras Model (input --> z_mean, z_log_var, z)
    - build_decoder : returns a Keras Model (z --> reconstructed waveform)
    - VAE           : full model with custom train_step and test_step

The encoder architecture (mirrored in the decoder) comprises:
three blocks of Conv1D (stride 2) + AveragePooling1D (stride 2), 
followed by a Flatten and two Dense layers (z_mean and z_log_var).
The input dimension must be divisible by 64.
"""

import tensorflow as tf
from tensorflow.keras import layers, metrics
from tensorflow.keras.models import Model


##########
# SAMPLING LAYER
class Sampling(layers.Layer):
    def call(self, inputs):
        """
        Apply the reparameterisation trick: z = z_mean + eps * exp(0.5 * z_log_var),
        where eps ~ N(0, I).

        Parameters
        ----------
        inputs : tuple of tf.Tensor
                (z_mean, z_log_var), both of shape (batch, latent_dim).

        Returns
        -------
        tf.Tensor
            Sampled latent vector z, shape (batch, latent_dim).
        """
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    
##########
# ENCODER
def build_encoder(original_dimension, latent_dimension, n_filters, 
                  kernel_initializer=None, bias_initializer=None):
    """
    Build the encoder component of the VAE model.

    Three blocks of Conv1D (stride 2) + AveragePooling1D (stride 2) progressively
    downsample the input, followed by a Flatten and two parallel Dense layers
    that output z_mean and z_log_var. A Sampling layer then draws z.

    Parameters
    ----------
    original_dimension   : int
        Number of time bins in each input waveform. Must be divisible by 64.
    latent_dimension     : int
        Size of the latent space.
    n_filters            : list of int, length 3
        Number of filters for conv1, conv2, conv3 respectively.
    kernel_initializer   : tf.keras.initializers or None
        Initializer for convolutional and dense kernels. Defaults to Keras default (GlorotUniform).
    bias_initializer     : tf.keras.initializers or None
        Initializer for biases. Defaults to Keras default (Zeros).

    Returns
    -------
    encoder : tf.keras.Model
        Maps input (batch, original_dimension, 1) into [z_mean, z_log_var, z],
        each of shape (batch, latent_dimension).
    """

    assert len(n_filters) == 3, f"n_filters must have length 3, got {len(n_filters)}"
    assert original_dimension % 64 == 0, f"original_dimension must be divisible by 64, got {original_dimension}"

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
    """
    Build the decoder component of the VAE model, mirroring the encoder structure.

    A Dense layer expands the latent vector, followed by three blocks of
    UpSampling1D + Conv1DTranspose that progressively reconstruct the original
    waveform length.

    Parameters
    ----------
    original_dimension   : int
        Number of time bins in the reconstructed output. Must match the value
        used in build_encoder.
    latent_dimension     : int
        Size of the latent space (must match build_encoder).
    n_filters            : list of int, length 3
        Number of filters for deconv3, deconv2, deconv1 respectively.
        It is used in reverse w.r.t. the encoder filter list.
    kernel_initializer   : tf.keras.initializers or None
        Initializer for convolutional and dense kernels.
    bias_initializer     : tf.keras.initializers or None
        Initializer for biases.

    Returns
    -------
    decoder : tf.keras.Model
        Maps latent vector (batch, latent_dimension) into reconstructed waveform
        (batch, original_dimension, 1).
    """

    assert len(n_filters) == 3, f"n_filters must have length 3, got {len(n_filters)}"
    assert original_dimension % 64 == 0, f"original_dimension must be divisible by 64, got {original_dimension}"

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
# VAE model
class VAE(Model):
    """
    Full Variational Autoencoder with a custom train_step and test_step.

    The total loss is the sum of:
        - reconstruction_loss : mean over the batch of sum-of-squared differences
          between input and reconstruction, computed over time bins and filter axis.
        - kl_loss             : mean over the batch of the KL divergence.

    Parameters
    ----------
    encoder : tf.keras.Model
        Encoder sub-model, as returned by build_encoder.
    decoder : tf.keras.Model
        Decoder sub-model, as returned by build_decoder.

    Attributes
    ----------
    total_loss_tracker          : tf.keras.metrics.Mean
    reconstruction_loss_tracker : tf.keras.metrics.Mean
    kl_loss_tracker             : tf.keras.metrics.Mean
    """

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

    def _compute_losses(self, data):
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.square(data - reconstruction)
        reconstruction_loss = tf.reduce_sum(reconstruction_loss, axis=[-2, -1])
        reconstruction_loss = tf.reduce_mean(reconstruction_loss, axis=-1)
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_sum(kl_loss, axis=-1)
        kl_loss = tf.reduce_mean(kl_loss, axis=-1)
        return reconstruction_loss + kl_loss, reconstruction_loss, kl_loss

    def train_step(self, data):
        with tf.GradientTape() as tape:
            total_loss, reconstruction_loss, kl_loss = self._compute_losses(data)
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
        total_loss, reconstruction_loss, kl_loss = self._compute_losses(data)
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
