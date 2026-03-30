### Variational (Convolutional) AutoEncoder

## Setup

import math
import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras import metrics
from tensorflow.keras import callbacks
from tensorflow.keras import initializers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

##################################################
## Dataset

data_dir = "../synthetic_waveforms"

# First file, initializing the numpy arrays
data_title = f"{data_dir}/synthetic_waveforms_00.npz"
print(f"Opening file {data_title}")
loaded = np.load(data_title)
noise_waves = loaded['noise']
signal_waves = loaded['signal']

# Load the other files and concatenate
for i in range(1,10):
    data_title = f"{data_dir}/synthetic_waveforms_0{i}.npz"
    print(f"Opening file {data_title}")
    loaded = np.load(data_title)
    noise_waves = np.concatenate((noise_waves,loaded['noise']),axis=0)
    signal_waves = np.concatenate((signal_waves,loaded['signal']),axis=0)
    
# Create the final array of fake traces and delete the loaded things to save memory
waves = noise_waves + signal_waves
print(f"Raw dataset shape: {waves.shape}")

del noise_waves, signal_waves, loaded

# The traces must have a number of bins multiple of 16 to be used in the current model
# In the real data, the first and last 125 are not used because there are instabilities in the digitazer
waves = waves[:,125:-22-125]

# Select the training (+ validation) dataset
train_samples = 7500
train_waves = waves[0:train_samples,:]
train_waves = train_waves[..., np.newaxis]
print(f"Training (+ validation) dataset shape: {train_waves.shape}")

original_dimension = train_waves.shape[1]

# Delete original waves array to save memory 
del waves
##################################################

##################################################
### Train the VAE

from vae_model import *

# Create directory to store history files
os.makedirs("./history_files", exist_ok=True)

# Initialize random seeds
tf.keras.utils.set_random_seed(18245)

# Choose the latent dimension
latent_dim = 2

# Other parameters of the architecture
n_filters = [4, 8, 16]

# Training
cooldown_models = 10
cooldown_epochs = 20
total_epochs = 150
epochs_per_save = 10

# Create an instance of the model just to print the model summary
vae = VAE(build_encoder(original_dimension, latent_dim, n_filters), 
          build_decoder(original_dimension, latent_dim, n_filters))
vae.compile(optimizer=optimizers.Adam())
vae.build(train_waves.shape)
vae.summary(expand_nested=True, show_trainable=True)
del vae


# Train 5 different models with a manual k-folding

for i in range(5) :
    
    print("\n\n Fold %s" % i)
    
    # Manually define the training set for this fold
    frac = int(train_waves.shape[0]/5)
    val_set = train_waves[i*frac:(i+1)*frac]        
    train1 = train_waves[:i*frac]
    train2 = train_waves[(i+1)*frac:]
    train_set = np.concatenate([train1, train2], axis=0)
    print(val_set.shape," - ",train1.shape," - ",train2.shape," - ",train_set.shape) 
    
    csv_logger_best = callbacks.CSVLogger(f"./history_files/VAE_init_fold{i}.csv", separator=',', append=True)

    # Checkpoint with only the best weights
    checkpoint_best = callbacks.ModelCheckpoint(filepath=f"./checkpoints/VAE_init_fold{i}",
                                                    save_weights_only=True,
                                                    monitor='val_loss',
                                                    mode='min',
                                                    save_best_only=True)
    for j in range(cooldown_models) :

        print("\n\n Model %s" % j)

        vae = VAE(build_encoder(original_dimension, latent_dim, n_filters), 
                  build_decoder(original_dimension, latent_dim, n_filters))
        vae.compile(optimizer=optimizers.Adam())

        vae.fit(train_set, validation_data=[val_set], shuffle=True, batch_size=100, verbose=2,
                epochs=cooldown_epochs,
                callbacks=[csv_logger_best, checkpoint_best])

        del vae
    
    del csv_logger_best, checkpoint_best
  
    
    # Training from the resulting best model
    
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, cooldown=10, verbose=1,
                                            min_delta=0.01, min_lr=1e-6, restore_best_weights=True)
   
    csv_logger = callbacks.CSVLogger(f"./history_files/VAE_fold{i}.csv", separator=',', append=True)

    weights_saver = WeightsSaver(N=epochs_per_save, title=f"./checkpoints/VAE_fold{i}")
    
    # Create an instance of the model, compile & fit
    
    vae = VAE(build_encoder(original_dimension, latent_dim, n_filters), 
              build_decoder(original_dimension, latent_dim, n_filters))
    vae.load_weights(f"./checkpoints/VAE_init_fold{i}")
    vae.compile(optimizer=optimizers.Adam())

    K.set_value(vae.optimizer.lr, 5e-4)
    print(vae.get_compile_config())
    
    vae.fit(train_set, validation_data=[val_set], shuffle=True, batch_size=100, verbose=2,
            initial_epoch = cooldown_epochs, epochs = total_epochs,
            callbacks=[reduce_lr, csv_logger, weights_saver])
    
    del vae
    del reduce_lr, csv_logger, weights_saver
    