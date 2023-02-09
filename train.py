"""Main module to load and train the model. This should be the program entry point."""
#generic imports
import os
import pathlib
import random
from datetime import datetime
import numpy as np
import math

#import constants
from EnformerCelltyping.constants import (
    CHROM_LEN, 
    CHROMOSOMES, 
    SAMPLES,
    SAMPLE_NAMES,
    SRC_PATH,
    HIST_MARKS,
    DATA_PATH,
    TRAIN_DATA_PATH)

#enformer imports
import tensorflow as tf
from IPython.display import clear_output
import pandas as pd
import time
from EnformerCelltyping.utils import(gelu, create_enf_model, 
                                     pearsonR,
                                     train_valid_split)
from os import listdir
from itertools import compress
import glob

"""Train model with Enformer and log with Wandb."""
# Set random seeds.
np.random.seed(101)
tf.random.set_seed(101)
random.seed(101)

SAVE_PATH = pathlib.Path("./model_results")
SAVE_PATH.mkdir(parents=True, exist_ok=True)

run_name = "enformer_celltyping"
features = ["A", "C", "G", "T","chrom_access_embed"] # cell type representation
labels = HIST_MARKS  # prediction targets
pred_resolution = 128  # window size must be divisible by prediciton resolution
window_size_dna = 196_608 #Enformer input size
batch_size = 128
n_epochs = 100 #lower than in manuscript for speed
learning_rate = 0.0002 #target learning rate,matches enformer

#remove dnase from pred if using in training
if 'dnase' in features or 'chrom_access_embed' in features:
    #labels.remove('dnase')
    labels.remove('atac')
    
# 1. --- SETUP PARAMETERS ------------------------------------------------
# 1.1 Dataset parameters
#test fraction proportion (equalling Leopards approach)
valid_frac = 0.2

# Train test split over chromosomes, samples or both
split = "SAMPLE"

# Exclude datasets from cells to be used for test set (3 immune cells)
excl = ['Monocyte','Neutrophil','T-Cell',]
train_valid_samples = np.delete(SAMPLES, np.isin(SAMPLES,excl))
# Don't exclude chromosomes when predicting across cell types
train_len = CHROM_LEN
train_chrom = CHROMOSOMES
test_len = CHROM_LEN
test_chrom = CHROMOSOMES
#Split the data into training and validation set - split by mix chrom and sample
#set seed so get the same split
(s_train_index, s_valid_index, c_train_index, c_valid_index, s_train_dist,
 s_valid_dist, c_train_dist, c_valid_dist) = train_valid_split(train_chrom,
                                                            train_len,
                                                            train_valid_samples,
                                                            valid_frac,
                                                            split)
# Training
train_cells = train_valid_samples[np.ix_(s_train_index)]
train_chromosomes = CHROMOSOMES[np.ix_(c_train_index)]
train_cell_probs = s_train_dist # equal probabilities
train_chromosome_probs = c_train_dist #weighted by chrom size
#get train cell IDs
train_ids = [list(SAMPLE_NAMES)[SAMPLES.index(cell_i)] for cell_i in train_cells]

# Validation
valid_cells = train_valid_samples[np.ix_(s_valid_index)]
valid_chromosomes = CHROMOSOMES[np.ix_(c_valid_index)]
valid_cell_probs = s_valid_dist
valid_chromosome_probs = c_valid_dist
#get valid cell IDs
valid_ids = [list(SAMPLE_NAMES)[SAMPLES.index(cell_i)] for cell_i in valid_cells]

#load datasets
train_valid_data = glob.glob(str(TRAIN_DATA_PATH/'*[0-9].npz'))
#identify the saved files for ATAC chromatin access
chrom_access = glob.glob(str(TRAIN_DATA_PATH/'*_ATAC.npz'))

#data loading imports
from EnformerCelltyping.utils import(
    PreSavedDataGen,
    train_valid_split)

#split by train and valid
#don't need to split by chrom as all chrom in train and valid
#split by cells
train_data = list(compress(train_valid_data, 
                           [any(x in dat for x in train_ids) for dat in train_valid_data]))
valid_data = list(compress(train_valid_data, 
                           [any(x in dat for x in valid_ids) for dat in train_valid_data]))
#pass in full paths
train_data = [str(TRAIN_DATA_PATH / i) for i in train_data]
valid_data = [str(TRAIN_DATA_PATH / i) for i in valid_data]
#remove reverse compliment and random permutation sequences from validation set
valid_data = [x for x in valid_data if x.endswith("0.npz")]

# 2. --- Data loaders ---------------------------------------------------
train_dataloader = PreSavedDataGen(files=train_data,
                                   batch_size=batch_size)
valid_dataloader = PreSavedDataGen(files=valid_data,
                                   batch_size=batch_size)

# 3. --- Model ----------------------------------------------------------


#Make sure save directories exist
checkpoint_path = f"{SAVE_PATH}/checkpoints/{run_name}"
save_path = f"{SAVE_PATH}/final_models/{run_name}"
pathlib.Path(f"{SAVE_PATH}/checkpoints/").mkdir(parents=True, exist_ok=True)
pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

#create Enformer Celltyping - for training
from EnformerCelltyping.enf_celltyping import Enformer_Celltyping

#don't add in layers for enformer since the dna is already 
#passed through enformer layers when precomputing
model = Enformer_Celltyping(use_prebuilt_model=False)

#set seed reproducibility
np.random.seed(102)
tf.random.set_seed(102)
random.seed(102)
    
print(model.summary())
#trainable params 1,681,985,712

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
          loss={'avg':tf.keras.losses.poisson,
                'delta':tf.keras.losses.mean_squared_error},
          metrics=['mse',pearsonR])

#checkpoint to rerun where left off
checkpoint = tf.keras.callbacks.ModelCheckpoint(
     filepath=checkpoint_path+'.{epoch:02d}.h5',
     save_freq='epoch', verbose=1, 
     save_weights_only=True,
     period=10
)

import datetime
strt = datetime.datetime.now()

# Train the model
model.fit(
        train_dataloader,
        epochs=n_epochs,
        verbose=2,
        validation_data=valid_dataloader,
        callbacks=[checkpoint]
    )


end = datetime.datetime.now()
print(end-strt)

# 7. --- Save the model ------------------------------------------------
model.save(save_path, save_format="tf") #Enformer tf.keras.Module