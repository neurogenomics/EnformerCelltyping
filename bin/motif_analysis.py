#for each chromosome, find peaks in norm pred and find top 10% of regions 
#that rely on global signal based on gradients


#model specifics
from EnformerCelltyping.utils import(
    pearsonR, 
    generate_data,
    initiate_bigwigs,
    create_enformer_celltyping,
    gelu,
    build_ec_embed_atac,
    build_ec_aftr_embed
)
from tensorflow.keras.metrics import mean_squared_error
from pathlib import Path
#import constants
from generate_data.constants import (
    CHROM_LEN,
    CHROMOSOMES,
    DATA_PATH,
    HIST_MARKS)

import os
import pathlib
import numpy as np
import math
import pandas as pd
import pyBigWig
#enformer imports
import tensorflow as tf

test_len = CHROM_LEN
test_chrom = CHROMOSOMES

pred_samples_pretty = ['Keratinocyte','Epimap_Astrocyte','Nott19_Astrocyte','Monocyte','Nott19_Microglia','Nott19_Neuron','Heart']
pred_samples = ['KERATINOCYTE','ASTROCYTE','Nott19_Astrocyte','Monocyte','Nott19_Microglia','Nott19_Neuron','HEART LEFT VENTRICLE']
labs = ['h3k27ac','h3k4me3']
avg_peaks = 128*8

#initiate data connection
data_conn = initiate_bigwigs(cells=pred_samples,
                             cell_probs=np.repeat(1, len(pred_samples)),
                             chromosomes=test_chrom,
                             chromosome_probs=np.repeat(1, len(test_chrom)),
                             features=["A", "C", "G", "T","chrom_access_embed"],
                             training=False,
                             labels=labs,
                             pred_res=128,
                             labels_for_all_cells=False)

## --------------------------
## Model loading

#embedding layers do not give gradients so can't get gradients on input:
# https://github.com/keras-team/keras/issues/12270
# need to split model
# and get gradient on gbl after embedding

#load EC
SAVE_PATH = pathlib.Path("./model_results")
from dna_hist_mark_pred.enf_celltyping_test import Enformer_Celltyping
assays = ['h3k27ac', 'h3k4me1', 'h3k4me3', 'h3k9me3', 'h3k27me3', 'h3k36me3']
model = Enformer_Celltyping(assays=assays,
                            enf_path=str(DATA_PATH / "enformer_model"),
                            use_prebuilt_model=True,
                            freeze = False,
                            enf_celltyping_pth = f"{SAVE_PATH}/final_models/EC_quant_combn_full")

#compile loaded model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
                  loss=tf.keras.losses.mean_squared_error)

#also load enf since pre-computing positions for gradients
#Transform input data by passing through Enformer
#less memory intensive than using map() on data generator
from dna_hist_mark_pred.utils import create_enf_chopped_model
enf = create_enf_chopped_model(str(DATA_PATH / "enformer_model"))

#get weights into two surrogate models
EC_mod = model.get_model()
EC_mod_embed = build_ec_embed_atac()
EC_mod_embed.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
                     loss=tf.keras.losses.mean_squared_error)
EC_mod_embed_aftr = build_ec_aftr_embed()
EC_mod_embed_aftr.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
                          loss=tf.keras.losses.mean_squared_error)

#make everything trainable
for layer in EC_mod.layers:
    layer.trainable = True    
for layer in EC_mod_embed.layers:
    layer.trainable = True
for layer in EC_mod_embed_aftr.layers:
    layer.trainable = True    
#move over layer weights by name
EC_mod_embed_lyr_nmes = [i.name for i in EC_mod_embed.layers]
EC_mod_embed_a_lyr_nmes = [i.name for i in EC_mod_embed_aftr.layers]
EC_mod_lyr_nmes = [i.name for i in EC_mod.layers]

for ind, nme_i in enumerate(EC_mod_embed_lyr_nmes):
    if nme_i in EC_mod_lyr_nmes:
        ind_lyr = EC_mod_lyr_nmes.index(nme_i)
        trained_weights_i = EC_mod.layers[ind_lyr].get_weights()
        EC_mod_embed.layers[ind].set_weights(trained_weights_i)
    else:
        print("EC before Embed - Didn't find: ",nme_i)    

for ind, nme_i in enumerate(EC_mod_embed_a_lyr_nmes):
    if nme_i in EC_mod_lyr_nmes:
        ind_lyr = EC_mod_lyr_nmes.index(nme_i)
        trained_weights_i = EC_mod.layers[ind_lyr].get_weights()
        EC_mod_embed_aftr.layers[ind].set_weights(trained_weights_i)
    else:
        print("EC aftr Embed - Didn't find: ",nme_i)     
        
        
#make everything non-trainable now
for layer in EC_mod.layers:
    layer.trainable = False   
for layer in EC_mod_embed.layers:
    layer.trainable = False   
for layer in EC_mod_embed_aftr.layers:
    layer.trainable = False            
    
#-----------------------------------

#get inputs - cell and hist mark to check
labels = HIST_MARKS 
#remove atac from hist 
labels.remove('atac')    

#input mark
import argparse
# argv
def get_args():
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument('-cell', '--cell', default='Monocyte', type=str, help='Cell type')
    parser.add_argument('-hist', '--hist', default='h3k27ac', type=str, help='Hist mark')
    args = parser.parse_args()
    return args

args=get_args()

cell = args.cell #'Monocyte'
hist = args.hist #'h3k27ac'
#leading and trailing whitespace
cell = cell.strip()
hist = hist.strip()

#get index of hist in model outputs
hist_ind = labels.index(hist)
#get cell name for data laoder
cell_nme_i=pred_samples[pred_samples_pretty.index(cell)]

mono_hist = pyBigWig.open(str(SAVE_PATH /"predictions"/f"{cell}_{hist}.bigWig"))
#save to bigwig, convert to bed after
bw_top = pyBigWig.open(str(SAVE_PATH /"predictions"/f"{cell}_{hist}_grad_top.bigWig"), "w")
bw_top.addHeader(list(zip(CHROMOSOMES , CHROM_LEN)), maxZooms=0) # zip two turples
ind_i=0
for ind,the_chr in enumerate(CHROMOSOMES):
    print("***"*5)
    print(the_chr)
    all_grads = []
    #get len divis by 128
    remainder = CHROM_LEN[ind]%(avg_peaks)
    mono_hist_vals = np.nan_to_num(mono_hist.values(the_chr, 0, CHROM_LEN[ind]-remainder,numpy=True))
    mono_hist_vals_pred_res = np.mean(mono_hist_vals.reshape(-1, avg_peaks),axis=1)
    del mono_hist_vals
    #restrict positions to where there was a peak with gbl then wasn't with none
    peak_act_mask = mono_hist_vals_pred_res>2
    #get index of where these peaks are
    peak_ind = np.where(mono_hist_vals_pred_res>2)[0]
    del mono_hist_vals_pred_res
    print(f"{len(peak_ind)} peaks found at resolution {avg_peaks} for {the_chr}")
    #get all possible start/end/chr values
    starts = np.arange(0,CHROM_LEN[ind])
    ends = np.arange(1,CHROM_LEN[ind]+1)
    chroms = np.array([the_chr] * len(starts))
    #now loop through these peak positions to get grad of gbl for hist mark of interest for each
    for peak_i in peak_ind:
        strt = peak_i*avg_peaks - (196_608//2) #get chromosonal pos of peak
        #1562*128 => CA recep field
        if (strt>0+(((1562*128)//2)-(196_608//2))) and ((strt+((1562*128)//2)+((196_608//2)))<=CHROM_LEN[ind]):
            #load the X values for the position
            X = next(generate_data(cells=cell_nme_i,chromosomes=test_chrom,
                            cell_probs=np.repeat(1, len(pred_samples)),
                            chromosome_probs=np.repeat(1, len(test_chrom)),
                            features=["A", "C", "G", "T","chrom_access_embed"],
                            labels=labs,
                            data=data_conn,
                            rand_pos=False,
                            labels_for_all_cells=False,
                            chro=the_chr,pos=strt,
                            data_trans = enf,
                            return_y = False,
                            training=False))
            #pass X to pre Embed layers
            pre_embed_out = EC_mod_embed(X)
            #update X vals
            X_update = {'dna':X['dna'],
                        'chrom_access_lcl1':pre_embed_out[0],
                        'chrom_access_lcl2':pre_embed_out[1],
                        'chrom_access_lcl3':pre_embed_out[2],
                        'chrom_access_gbl':pre_embed_out[3]}
            #test to ensure surrogate models match actual model on first peak
            if ind_i==0:
                #past to post embed layers to get pred
                pred_surr = EC_mod_embed_aftr(X_update) 
                #check perf against act model
                pred = EC_mod(X)
                assert pearsonR(pred,pred_surr)>.99, f"Surrogate models do not make same pred as original, Pearson R: {pearsonR(pred,pred_surr)}"
                ind_i+=1
            
            #now work out gradient
            X2 = {k:tf.Variable(tf.identity(v)) for k, v in X_update.items()}
            with tf.GradientTape() as g:
                #get gradient with respect to specific hist mark output
                output_tensor = EC_mod_embed_aftr(X2)[:,:,hist_ind]

            gradients = g.gradient(output_tensor, X2)
            #aggregate the gradient on the gbl
            all_grads.append(np.mean(np.absolute(gradients['chrom_access_gbl'][0])))
        else:
            #get abs value of gradients so no negatives
            all_grads.append(0)
    #get indexes of top 10% of values (peak gradients)
    n = int(len(all_grads)*.1)
    top_n_ind = sorted(range(len(all_grads)), key=lambda i: all_grads[i])[-n:]
    #now get genomic positions for top X%
    top_n_peak = peak_ind[top_n_ind]
    top_n_genomic_pos = [i*avg_peaks for i in top_n_peak]
    #now get genomic positions for top X%
    #save to bigwig
    #create array of zeros and add 1 for chosen positions
    vals = np.zeros(CHROM_LEN[ind])
    #going to add 1's for full avg_peaks length
    for pos_i in top_n_genomic_pos: 
        for j in range(avg_peaks):
            vals[pos_i+j]=1.0
    bw_top.addEntries(chroms, starts,
                      ends=ends, values=vals)
    #help with memory usage, clear mem every chrom
    import gc
    tf.keras.backend.clear_session()
    gc.collect()
    del vals
bw_top.close()