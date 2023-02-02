#----------------------------------------------------
#Precompute genome wide training regions
#This saves time by loading the data from the bigwigs
#for each training location once and then storing these
#as npy objects which then can be used with a normal 
#data loader. Loading data on the fly is time consuming.

#Steps to generate training regions:
#1. Bin genome based on predictive window
#2. Filter bins to select training set based on
#   DNA and cell type filters. DNA filters:
#     1. Leave buffer at start/end chrom
#     2. Not in blacklist regions
#   Cell type filters:
#     1. Coverage for the histone mark > 12.5% -
#     Then down sampled to the lowest count of marks
#     so each hist mark has equal representation
#     at 11,903 positions to avoid model prioritising
#     training on one mark
#This results in 67,007 unique training & validation positions
# - this is 14,188 unique DNA positions split train/valid
# - similar to num basenji & enformer = 14,533 sequences
#This approach ensures model sees peaks for all histone marks
#Get validation set positons by randomly sampling DNA
#start position to avoid overfitting to training bins
#Model paramters
#1. 1562*128 bp of ATAC data (approx same as DNA window) for local
#   cell type info
#2. 1216*3_000 (3.6m) bp of ATAC data (PanglaoDB marker genes) for 
#   global cell type info
#Finally save the data as npy objects to use with a data loader
#----------------------------------------------------

#model specifics
from EnformerCelltyping.utils import(
    create_buffer,
    load_bigwig,
    generate_data,
    initiate_bigwigs,
    train_valid_split,
    load_y
)
from pathlib import Path
#import constants
from EnformerCelltyping.constants import (
    CHROM_LEN,
    CHROMOSOMES,
    SAMPLES,
    SAMPLE_NAMES,
    CHROMOSOME_DATA,
    SRC_PATH,
    HIST_MARKS,
    DATA_PATH,
    TRAIN_DATA_PATH,
    PROJECT_PATH,
    BLACKLIST_PATH)
import os
import pathlib
import numpy as np
import random
import math
import pandas as pd
import os.path
import itertools

import argparse
#run in parallel
#pass inputs
# argv
def get_args():
    parser = argparse.ArgumentParser(description="mkeData")
    parser.add_argument('-s', '--start', default=0, type=int, help='start indicator')
    parser.add_argument('-e', '--end', default=0, type=int, help='end indicator')
    args = parser.parse_args()
    return args

args=get_args()

start=args.start
end=args.end

# Exclude datasets from cells to be used for test set (3 immune cells)
train_valid_samples = SAMPLES
features = ["A", "C", "G", "T","chrom_access_embed"]
labels = HIST_MARKS  # prediction targets
#remove dnase, don't want to predict this
#remove dnase from pred if using in training
if 'chrom_access_embed' in features and 'atac' in labels:
    labels.remove('atac')

#following values should be the same as when building the model
#pre-trained Enformer takes wider DNA input and then just cuts it so this prop reflects that
#this value gives the same amount of output values as enformer
pred_prop = (128*896)/196_608 # centre proportion of base-pairs from input to return preds for
window_size = 196_608
#embedding
window_size_CA=1562*128 #local Chromatin Accessibility size
#why so large? - https://www.pnas.org/doi/full/10.1073/pnas.0909344107 used 4k around TSS for promoter
up_dwn_stream_bp=3_000#at every prot coding TSS in genome
pred_resolution = 128#25
coverage_threshold = 0.125

#Transform input data by passing through Enformer
#less memory intensive than using map() on data generator
from EnformerCelltyping.utils import create_enf_chopped_model
enf = create_enf_chopped_model(str(DATA_PATH / "enformer_model"))

#initiate data connection
data_conn = initiate_bigwigs(cells=train_valid_samples,
                             cell_probs=np.repeat(1, len(train_valid_samples)),
                             chromosomes=CHROMOSOMES,
                             chromosome_probs=np.repeat(1, len(CHROMOSOMES)),
                             features=features,
                             labels=labels,
                             pred_res=pred_resolution,
                             load_avg=True)

#workout output bp's and buffer bp
buffer_bp, target_length, target_bp = create_buffer(window_size=window_size, 
                                                    pred_res=pred_resolution, 
                                                    pred_prop=pred_prop)
num_buffer = np.int(np.ceil(buffer_bp/target_length))


Path(TRAIN_DATA_PATH).mkdir(parents=True, exist_ok=True)
Path(PROJECT_PATH / "checkpoint").mkdir(parents=True, exist_ok=True)



#only run if checkpoint file of genomic coverage not already saved
if(not os.path.isfile(PROJECT_PATH / "checkpoint"/"reg_cov.csv.gz")):
    #loop through chromosomes saving appropriate positions based on 
    #DNA filters - i.e. edge buffer regions and blacklist regions
    for ind,chrom_i in enumerate(CHROMOSOMES):
        #get bins
        bins = np.arange(0,CHROM_LEN[ind]+target_bp,target_bp)
        #remove buffer - based on edge regions
        bins = bins[num_buffer:len(bins)-num_buffer]
        #remove blacklist regions
        blacklist_regions = load_bigwig(BLACKLIST_PATH)
        keep = []
        #if blacklist within dna window remove
        for index, pos in enumerate(bins):
            if(blacklist_regions.entries(chrom_i,
                                         #include all DNA input reg
                                         pos - buffer_bp,
                                         pos+target_bp+buffer_bp) is None):
                keep.append(index)
        bins = bins[keep]
        #store in pd df
        if(ind==0):
            dna_bins = pd.DataFrame({
                'chr': chrom_i,
                'pred_strt': bins,
                'pred_end': bins+target_bp,
                'dna_strt': bins-buffer_bp,
                'dna_end': bins+target_bp+buffer_bp
            })
        else:
            tmp = pd.DataFrame({
                'chr': chrom_i,
                'pred_strt': bins,
                'pred_end': bins+target_bp,
                'dna_strt': bins-buffer_bp,
                'dna_end': bins+target_bp+buffer_bp
            })
            dna_bins = pd.concat([dna_bins, tmp])
    dna_bins = dna_bins.reset_index()

    #pick random start positions for DNA sequences validation set regions
    #set seed
    np.random.seed(101)

    #Now filter based on cell type filters
    #Cell type filtes based on coverage peaks
    #get coverage for each mark in each cell type
    #using chromHMM definition >2 -log10 p-val
    cov = []
    peak_cutoff = 2
    for index, row in dna_bins.iterrows():
        if(index%1000==0):
            print(row['chr'] +": "+str(row['pred_strt']))
        dna_strt = row['pred_strt'] 
        the_chr = row['chr']
        all_y = load_y(data = data_conn,
                       target_length=target_length,
                       labels=labels,
                       cells=train_valid_samples,
                       selected_chromosome=the_chr,
                       selected_cell=train_valid_samples,
                       window_start = dna_strt,
                       buffer_bp = buffer_bp,
                       window_size = window_size,
                       pred_res = pred_resolution,
                       arcsin_trans=False,
                       debug=False)
        y_thres = (all_y>peak_cutoff)*1
        cov.append(np.mean(y_thres,axis=0))

    cov_comb = np.c_[cov]
    cell_mark_ord = [e1+', '+e2 for e1,e2 in itertools.product(train_valid_samples,labels)]
    #merge coverage with region
    cov_reg = pd.concat([dna_bins,
                         pd.DataFrame(cov_comb, columns=cell_mark_ord)], 
                        axis=1)
    #go from wide to long
    cov_reg = cov_reg.reset_index()
    cov_reg = pd.melt(cov_reg, id_vars=list(dna_bins.columns.values), value_vars=cell_mark_ord)
    #split out histone mark and cell columns
    cov_reg[['cell', 'mark']] = cov_reg['variable'].str.split(', ', 1, expand=True)
    #delete unnecessary columns
    cov_reg.drop(cov_reg.columns.difference(['chr','pred_strt','pred_end',
                                   'dna_strt','dna_end','cell',
                                   'mark','value']), 1, inplace=True)
    #save
    cov_reg.to_csv(PROJECT_PATH / "checkpoint"/"reg_cov.csv.gz",
                   index=False,compression='gzip')
else:
    cov_reg = pd.read_csv(PROJECT_PATH / "checkpoint"/"reg_cov.csv.gz")

#filter to wanted coverage
cov_reg = cov_reg[cov_reg['value']>coverage_threshold]
#match number of positions for each hist mark
count_hist = cov_reg['mark'].value_counts()
n = count_hist.min()
cov_reg = cov_reg.groupby('mark').sample(n=n,random_state=101)
#get unique pos based on all hist marks
cov_reg = cov_reg.drop_duplicates(['chr','dna_strt','cell'])
cov_reg=cov_reg.reset_index(drop=True)

#now save positions - 
#we want validation dataset to have different DNA sequence
#to training to avoid overfitting to the training sequence
#so randomly move the validation dna seq start position to avoid this

#first get validation cells
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

random.seed(101)
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
# Validation
valid_cells = train_valid_samples[np.ix_(s_valid_index)]


import datetime
#loop through all training cells saving training data of each pred reg
for index, row in cov_reg.iterrows():
    if(index>=start) & (index<=end):
        if(index%1000==0):
                print(datetime.datetime.now())
                print(row['chr'] +": "+str(row['dna_strt']))
        #shift validation dna_strt so not same DNA positions as training
        if row['cell'] in valid_cells:
            #set seed based on current strt pos so same res
            #if running in batch or altogether
            np.random.seed(row['dna_strt'])
            random.seed(row['dna_strt'])
            chrom_len = CHROM_LEN[CHROMOSOMES == row['chr']][0]
            #shift by up to a quarter of the predictive window
            #low of pred_resolution so y always moves to new position
            rand_int = int(
                np.random.randint(low=pred_resolution, high=target_bp//4, size=1)
            )
            shift = min(rand_int,chrom_len-window_size)
            strt = row['dna_strt']+shift
        else:
            strt = row['dna_strt'] 
        the_chr = row['chr']
        cell_i = row['cell']
        #als save rev comp and rand pos shift
        X,y = next(generate_data(cells=cell_i,chromosomes=CHROMOSOMES,
                                 cell_probs=np.repeat(1, len(train_valid_samples)),
                                 chromosome_probs=np.repeat(1, len(CHROMOSOMES)),
                                 features=features,labels=labels,data=data_conn,
                                 window_size=window_size,pred_res=pred_resolution,
                                 pred_prop=pred_prop,rand_pos=False,
                                 chro=the_chr,pos=strt,
                                 n_genomic_positions=window_size_CA,
                                 up_dwn_stream_bp = up_dwn_stream_bp,
                                 reverse_complement=True,
                                 rand_seq_shift=True,
                                 rtn_y_avg = True,
                                 chrom_access_delta = True,
                                 data_trans = enf#trans DNA now, quicker than in model
                                ))

        #get id for cell
        cell_id = list(SAMPLE_NAMES)[SAMPLES.index(cell_i)]
        #save
        #save local ATAC with DNA
        for i in range(X['dna'].shape[0]):
            np.savez(TRAIN_DATA_PATH/f'{cell_id}_{the_chr}_strt_{strt}_{i}.npz', 
                     X_dna=X['dna'][i:i+1,:,:],
                     X_chrom_access=X['chrom_access_lcl'][i:i+1,:],
                     X_chrom_access_gbl=X['chrom_access_gbl'][i:i+1,:],
                     y_avg=y['avg'][i:i+1,:,:],
                     y_act=y['act'][i:i+1,:,:]
                    )
