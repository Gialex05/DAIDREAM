#!/usr/bin/env python
# coding: utf-8

"""
CAE_training.py

Training script for a Convolutional Autoencoder (CAE) designed to identify
rare signals embedded in noise waveforms in the context of astroparticle physics.

Input:
    Waveform files in NumPy compressed format (.npz), located in
    ../synthetic_waveforms/synthetic_waveforms_XX.npz
    Each file must contain the arrays:
        - 'noise'  : noise-only waveforms, shape (N, n_bins)
        - 'signal' : signal-only waveforms, shape (N, n_bins)

Output:
    - Model weights: ./checkpoints/CAE_roundXX.weights.h5
    - Training history (epochs, loss, val_loss, ...): CAE.csv

Training strategy:
    Round 0    : n. of random initializations, best model (lowest val_loss) is kept.
    Rounds 1-4 : the best model from the previous round is re-trained with
                 multiple random seeds to escape potential local minima.
"""

import numpy as np
import tensorflow as tf

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

from tensorflow.keras import callbacks
from tensorflow.keras.optimizers import Adam

###########################################################################
# First file, initializing the numpy arrays
loaded = np.load("../synthetic_waveforms/synthetic_waveforms_00.npz")
noise_waves = loaded['noise']
signal_waves = loaded['signal']

# Load the other files and concatenate
for i in range(1,10):
    loaded = np.load(f"../synthetic_waveforms/synthetic_waveforms_0{i}.npz")
    noise_waves = np.concatenate((noise_waves,loaded['noise']),axis=0)
    signal_waves = np.concatenate((signal_waves,loaded['signal']),axis=0)

# Create the final array of fake traces and delete the loaded things to save memory
waves = noise_waves + signal_waves
print(waves.shape)
del noise_waves, signal_waves, loaded

# In the real data, the first and last 125 are not used because there are instabilities in the digitazer
# Then, additional 22 values are removed since the traces must have a number of bins multiple of 16 
# to be correcly reconstructed in the decoder of the current model
waves = waves[:,125:-125-22]

# Select the training (+ validation) dataset
train_samples = 7500
train_waves = waves[0:train_samples,:]
train_waves = train_waves[..., np.newaxis]

original_dim = train_waves.shape[1]
print(f"Dataset selected for training (and validation): {train_samples} traces of {original_dim} time bins")

# Delete original waves array to save memory 
del waves


###########################################################################

from cae_model import convolutional_autoencoder, my_loss_func

###################################
# FUNDAMENTAL PARAMETERs !

# DIMENSION OF THE LATENT SPACE
encoded_dim = 4

# Title, used in the history and checkpoints files
title = "CAE"

# Initial random seed for the first round of training
tf.keras.utils.set_random_seed(8888)

# Following random seeds
seeds = np.array([20845, 568234, 9182])

# Number of randomly-initialized models for the first round of training
n_random_models = 10

# Epochs for each round of training 
epochs_per_training = 5


#####################################
# Initial round of training

# Define callbacks

# Reduce L.R. on plateau
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, cooldown=5, 
                                        min_delta=1e-4, min_lr=1e-5, restore_best_weights=True)
                                                 
# Stream training results to a CSV file
csv_logger = callbacks.CSVLogger(f"{title}.csv", separator=',', append=True)

# Store model after training
ckpt_name = f"./checkpoints/{title}_round00.weights.h5"
checkpoint = callbacks.ModelCheckpoint(filepath=ckpt_name, 
                                       save_weights_only=True, 
                                       monitor='val_loss', 
                                       mode='min', 
                                       save_best_only=True)


# Arrays to store values of L.R. and val. loss after the training
lr_values = np.zeros(n_random_models)
val_loss_values = np.zeros(n_random_models)

for i in range (0,n_random_models) :
    
    print("\n Round 00 - model %s" % i )

    model = convolutional_autoencoder(original_dim, encoded_dim)

    model.compile(optimizer=Adam(learning_rate=1e-3), loss=my_loss_func, metrics=['mean_absolute_error'])

    history = model.fit(train_waves, train_waves, verbose=2,
    					epochs=epochs_per_training, batch_size=100, 
                        callbacks=[checkpoint, reduce_lr, csv_logger],
                        validation_split=0.2, shuffle=True)
        
    # Get the last value of the validation loss
    val_loss_values[i] = history.history['val_loss'][-1]
              
    # Get the last value of the learning rate
    lr_values[i] = model.optimizer.learning_rate.numpy()


#########################
# At the end of the first round, select the "best model"
# as the one with the lowest validation loss
best_model_index = np.argmin(val_loss_values)
print("The best model (lowest val. loss) at the end of Round 00 is n.",best_model_index)
lr_best_model = lr_values[best_model_index]


####################################
# Following rounds of training

for i in range (1,5) :

    print("\n\n Round %s" % i)
    
    ckpt_name = (f"./checkpoints/{title}_round0{i}.weights.h5")
    checkpoint = callbacks.ModelCheckpoint(filepath=ckpt_name,
                                            save_weights_only=True,
                                            monitor='val_loss',
                                            mode='min',
                                            save_best_only=True)

    # Arrays to store values of L.R. and val. loss after the training
    lr_values = np.zeros(seeds.size)
    val_loss_values = np.zeros(seeds.size)

    for j in range (0,int(seeds.size)) :
    
        print("\n Round %s - model %s" % (i,j) )
        tf.keras.utils.set_random_seed(int(seeds[j]))
        
        # Load best model weights from previous round of training (without compiling)    
        model_second_step = convolutional_autoencoder(original_dim, encoded_dim)
        model_second_step.load_weights(f"./checkpoints/{title}_round0{i-1}.weights.h5")    
        
        # Compile the model (required because of the custom loss)
        model_second_step.compile(optimizer=Adam(learning_rate=lr_best_model), loss=my_loss_func, metrics=['mean_absolute_error']) 

        history = model_second_step.fit(train_waves, train_waves, verbose=2,
                                        initial_epoch=i*epochs_per_training, 
                                        epochs=(i+1)*epochs_per_training, 
                                        batch_size=100, 
                                        callbacks=[checkpoint, reduce_lr, csv_logger],
                                        validation_split=0.2, shuffle=True)

        # Get the last value of the validation loss
        val_loss_values[j] = history.history['val_loss'][-1]
              
        # Get the last value of the learning rate
        lr_values[j] = model_second_step.optimizer.learning_rate.numpy()
 

    #########################
    # At the end of each round of training, select the "best model"
    # as the one with the lowest validation loss
    best_model_index = np.argmin(val_loss_values)
    print(f"The best model (lowest val. loss) at the end of Round 0{i} is n. {best_model_index}")
    lr_best_model = lr_values[best_model_index]
