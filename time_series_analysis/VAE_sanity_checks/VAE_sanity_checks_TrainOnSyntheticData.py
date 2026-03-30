#!/usr/bin/env python
# coding: utf-8

"""
VAE_sanity_checks_TrainOnSyntheticData.py

Server-side training script for the VAE convergence study on synthetic waveforms.
Equivalent to VAE_sanity_checks_TrainOnSyntheticData.ipynb, designed to be run
on a server or cluster without a Jupyter environment.

Workflow
--------
1. Load and concatenate synthetic waveform .npz files from ../synthetic_waveforms/.
2. Preprocess waveforms (slicing, selection of true pulses by integral quantile).
3. Build a controlled-mixing dataset: each batch contains a fixed number of
   signal waveforms and noise-only waveforms, in a reproducible order.
4. Train one independent VAE instance per seed. For each seed:
   - initialise a new VAE with the given random seed
   - save the initial weights to ./weights/
   - train for a fixed number of epochs
   - save the trained weights to ./weights/
   - stream the training history to ./history_files/

Input
-----
    ../synthetic_waveforms/synthetic_waveforms_00.npz
    ...
    ../synthetic_waveforms/synthetic_waveforms_09.npz

Output
------
    ./history_files/{config_name}_{seed}.csv     training history per seed
    ./weights/{config_name}_{seed}_init.npz      weights before training
    ./weights/{config_name}_{seed}_trained.npz   weights after training

Dependencies
------------
    vae_model.py, waveforms_tools.py, weights_tools.py

Notes
-----
- This script is a direct equivalent of VAE_sanity_checks_TrainOnSyntheticData.ipynb.
  Any logic change must be applied to both files manually until a shared
  module is introduced.
- shuffle=False in model.fit() is intentional: the dataset is pre-mixed with a
  controlled signal/noise ratio per batch, and reproducibility requires that
  the batch structure is preserved across runs.
"""

import os
import numpy as np

import keras
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks


########################
##### Load dataset #####

data_dir = "../synthetic_waveforms"

data_title = f"{data_dir}/synthetic_waveforms_00.npz"
print(f"Opening file {data_title}")
loaded = np.load(data_title)

noise_wave = loaded['noise']
signal_wave = loaded['signal']
true_peaks = loaded['peak']
true_peak_positions = loaded['peak_position']
true_integrals = loaded['integral']

for i in range(1,10):
    data_title = f"{data_dir}/synthetic_waveforms_0{i}.npz"
    print(f"Opening file {data_title}")
    loaded = np.load(data_title)
    
    noise_wave = np.concatenate((noise_wave,loaded['noise']),axis=0)
    signal_wave = np.concatenate((signal_wave,loaded['signal']),axis=0)
    true_peaks = np.concatenate((true_peaks,loaded['peak']),axis=0)
    true_peak_positions = np.concatenate((true_peak_positions,loaded['peak_position']),axis=0)
    true_integrals = np.concatenate((true_integrals,loaded['integral']),axis=0)

wave = noise_wave + signal_wave


###############################
##### Dataset Preparation #####

from waveforms_tools import *

## remove time samples from the beginning and ending of each WF as done in real ReD data
wfs        =       wave[:,125:-22-125]
noise_wave = noise_wave[:,125:-22-125]

### fix original_dimension varible as the n. time samples in each event
original_dimension = wfs.shape[1]

# Get basic properties for all WFs
integrals  = get_integral(wfs, whole=True)
peaks      = get_peak(wfs)
sel_integral_indexes = get_quantile_indices(integrals)
sel_peak_indexes     = get_quantile_indices(peaks)

## Apply a threshold selection on the integral 
good_wfs = wfs[sel_integral_indexes]

print(f'Selecting {len(good_wfs)} events considered as true pulses')

true_pulses = np.expand_dims(good_wfs, axis=-1)
pure_noise  = np.expand_dims(noise_wave, axis=-1)
print(true_pulses.shape)
print(pure_noise.shape)


##########################
##### Dataset Mixing #####

# Configuration
batch_size = 100
total_batches = 70
n_good_evts_per_batch = 10

assert n_good_evts_per_batch < batch_size, "Requested too many good waveforms per batch."

config_name = f'VAE_4dim_{total_batches*batch_size}evts_{n_good_evts_per_batch}good-per-batch'

### To make sure that the mixed dataset have 
### the events in the same order
rng = np.random.default_rng(seed=42) 

dataset = []

# Configuration
n_random = batch_size - n_good_evts_per_batch
final_size = total_batches * batch_size

# --- Validation ---
assert true_pulses.shape[0] >= total_batches * n_good_evts_per_batch, "Not enough unique waveforms in true pulses."

# --- Step 1: Random sampling ---
# Select the required number of unique good waveforms
good_indices = rng.permutation(true_pulses.shape[0])[:total_batches * n_good_evts_per_batch]
good_waveforms = true_pulses[good_indices]  # Shape: (n. good, 9728, 1)

# Select the required number of random traces (with replacement) from pure_noise
random_indices = rng.choice(pure_noise.shape[0], size=total_batches * n_random, replace=True)
random_waveforms = pure_noise[random_indices]  # Shape: (n. random, 9728, 1)

# Step 2: Interleave exactly 10 good + 90 noise per batch
good_waveforms = good_waveforms.reshape(total_batches, n_good_evts_per_batch, *good_waveforms.shape[1:])
random_waveforms = random_waveforms.reshape(total_batches, n_random, *random_waveforms.shape[1:])

# Combine per batch
batched = np.concatenate([good_waveforms, random_waveforms], axis=1)  

# --- Step 3: Shuffle each batch independently (no loop) ---
shuffle_indices = np.argsort(rng.random((total_batches, batch_size)), axis=1)
shuffled = batched[np.arange(total_batches)[:, None], shuffle_indices]

# --- Step 4: Flatten back to final output ---
dataset = shuffled.reshape(final_size, *batched.shape[2:])  # (n. tot, 9728, 1)
    

#########################
##### Train the VAE #####

from vae_model import VAE, build_encoder, build_decoder
from weights_tools import *

os.makedirs("./history_files", exist_ok=True)

# Parameters of the architecture

# Latent Dimension
latent_dim = 4

# n. of filters
n_filters = [2, 4, 8]

# Seeds for obtaining different models' training
seeds = np.array([32467, 75410, 87278, 34987, 38956, 26012, 67418, 47842, 66012, 87346])

print(f"Dataset name : {config_name}")
print(f"Dataset shape: {dataset.shape}")
print(f'{n_good_evts_per_batch} good events in each batch of {batch_size} events.') 
    
for j in range (0,int(seeds.size)):

    print(f"\nInitializing model {j} (seed {seeds[j]})")

    keras.utils.set_random_seed(int(seeds[j]))
    tf.keras.utils.set_random_seed(int(seeds[j]))

    # Stream history into a CSV file
    csv_logger = callbacks.CSVLogger(f'./history_files/{config_name}_{seeds[j]}.csv', 
                                     separator=',', append=False)

    vae = VAE(build_encoder(original_dimension, latent_dim, n_filters), 
              build_decoder(original_dimension, latent_dim, n_filters))
    vae.compile(optimizer=optimizers.Adam(learning_rate=1e-3))

    save_weights_npz(vae, config_name, "init", seeds[j], overwrite=True)

    vae.fit(dataset, epochs=15, validation_split=0.2, shuffle=False,
            batch_size=batch_size, callbacks=[csv_logger], verbose=2)

    save_weights_npz(vae, config_name, "trained", seeds[j], overwrite=True)
