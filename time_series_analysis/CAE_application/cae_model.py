"""
CAE_model.py

Shared definitions for the CAE model architecture and custom loss function.
Import this module in CAE_training.py and analysis notebooks to ensure
consistency across training and inference.
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

###########################################################################
def my_loss_func(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    # Sum over time bins (axis=-2) and filter dimension (axis=-1),
    # returning shape (batch,). Keras averages over the batch internally.
    return tf.reduce_sum(squared_difference, axis=[-2, -1])

###########################################################################
def convolutional_autoencoder(original_dimension, encoded_dimension):
    """
    Build and compile a 1D Convolutional Autoencoder (CAE).

    Encoder: three blocks of Conv1D (stride 2) + AveragePooling1D (stride 2),
             followed by a Flatten and a Dense layer ('dense_encoded') that
             projects the representation into the latent space.

    Decoder: a Dense layer that expands the latent vector, followed by three
             blocks of UpSampling1D + Conv1DTranspose, mirroring the encoder.

    The input dimension must be divisible by 64 (= 2^3 pooling * 2^3 conv stride)
    for the decoder reshape to be consistent.

    Parameters
    ----------
    original_dimension : int
        Number of time bins in each input waveform.
    encoded_dimension : int
        Size of the latent space (bottleneck).

    Returns
    -------
    model : tf.keras.Sequential
        Uncompiled CAE model. Compile separately with the desired
        optimizer and loss before training.
    """

    n_filters1 = 4
    n_filters2 = 8
    n_filters3 = 16

    model = Sequential()

    model.add(layers.InputLayer((original_dimension,1), name="input"))
    
    # Three layers of 1D convolution + average pooling

    model.add(layers.Conv1D(name="conv1", filters=n_filters1, 
                            kernel_size=32, activation='relu', strides=2, padding='same', use_bias=True))
    model.add(layers.AveragePooling1D(pool_size=2, strides=2, padding='valid'))

    model.add(layers.Conv1D(name="conv2", filters=n_filters2,
                            kernel_size=32, activation='relu', strides=2, padding='same', use_bias=True))
    model.add(layers.AveragePooling1D(pool_size=2, strides=2, padding='valid'))
   
    model.add(layers.Conv1D(name="conv3", filters=n_filters3,
                            kernel_size=32, activation='relu', strides=2, padding='same', use_bias=True))    
    model.add(layers.AveragePooling1D(pool_size=2, strides=2, padding='valid'))    
    
    # Inner part of the autoencoder
    # The result of the convulution (+ pooling) layers is flattened.
    # Then 2 dense layers (one for encoding, one for deconding) follows
    # Finally, the output is again put in the form of (trace, n filters)

    dimension_reduction_pooling = 2*2*2
    dimension_reduction_conv_stride = 2*2*2
    dimension_after_convolutions = original_dimension // (dimension_reduction_pooling * dimension_reduction_conv_stride)

    model.add(layers.Flatten(name="flatten"))
    model.add(layers.Dense(encoded_dimension, name="dense_encoded", activation='linear', use_bias=True))
    model.add(layers.Dense(dimension_after_convolutions * n_filters3, name="dense_decoded", activation='relu', use_bias=True))
    model.add(layers.Reshape((dimension_after_convolutions, n_filters3), name="reshape") )
    
    # Deconding part, with three transpose 1D convolutional + up-sampling layers

    model.add(layers.UpSampling1D(size=2))
    model.add(layers.Conv1DTranspose(name="deconv3", filters=n_filters2, 
                                     kernel_size=32, activation='relu', strides=2, padding='same', use_bias=True))
     
    model.add(layers.UpSampling1D(size=2))
    model.add(layers.Conv1DTranspose(name="deconv2", filters=n_filters1, 
                                     kernel_size=32, activation='relu', strides=2, padding='same', use_bias=True))

    model.add(layers.UpSampling1D(size=2))
    model.add(layers.Conv1DTranspose(name="deconv1", filters=1,
                                     kernel_size=32, activation='relu', strides=2, padding='same', use_bias=True))

    return model
