#PRE-SAVE DNA seq for sldp predicitons so it will speed
#up prediction time


#model specifics
from EnformerCelltyping.utils import(
    initiate_bigwigs,
    generate_sample,
    create_ref_alt_DNA_window,
    plot_snp_dna_window,
    predict_snp_effect_sldp_checkpoint
)

#import constants
from EnformerCelltyping.constants import (
    CHROM_LEN,
    CHROMOSOMES,
    SAMPLES,
    SAVE_PATH,
    HIST_MARKS,
    DATA_PATH)

import pathlib
import numpy as np
import math
import pandas as pd
#enformer imports
import tensorflow as tf
import datetime
import os 
import random
import glob

# test on all chromosomes
test_len = CHROM_LEN
test_chrom = CHROMOSOMES

effect_mode = 'both'

# Set random seeds.
np.random.seed(101)
tf.random.set_seed(101)
random.seed(101)

#inputs
cell = 'Neutrophil'
#load all hm3 filt snps and find unique ones
all_hm3_snps = glob.glob(str(DATA_PATH / 'qtl'/'*hm3_snps.sumstats.gz'))
all_snps = []
for snp_i in all_hm3_snps:
    all_snps.append(pd.read_csv(snp_i,sep='\t'))
sumstats = pd.concat(all_snps)
#remove dups
sumstats.drop_duplicates(subset=['SNP','CHR','BP','A1','A2'],inplace = True)
sumstats = sumstats.reset_index()
print(f"SNPS: {sumstats.shape[0]}")
pathlib.Path(SAVE_PATH/'snp_effects/').mkdir(parents=True, exist_ok=True)

model=None
#Transform input data by passing through Enformer
#less memory intensive than using map() on data generator
from EnformerCelltyping.utils import create_enf_chopped_model
enf = create_enf_chopped_model(str(DATA_PATH / "enformer_model"))

#create data generator class for samples
data_generator = generate_sample(
    cells = SAMPLES,
    chromosomes=test_chrom,
    arcsin_trans = False,
    reverse_complement = True,
    rand_seq_shift = True,
    return_y = False,
    rtn_rand_seq_shift_amt=True,
    data_trans = enf
    )

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

import datetime
#loop through SNPs saving training data
for index, row in sumstats.iterrows():
    if(index>=start) & (index<=end):
        strt_t = datetime.datetime.now()
        dna_strt, snp_pos = create_ref_alt_DNA_window(chro = 'chr'+str(row['CHR']), 
                                                  pos = row['BP'])
        
        #can plot if we want:
        #plot_snp_dna_window(dna_strt,snp_pos)
        #Predict the effective difference of the alternative allele
        agg_eff = predict_snp_effect_sldp_checkpoint(model=model,
                                                     alt=row['A2'],cell = cell, 
                                                     chro='chr'+str(row['CHR']),
                                                     dna_strt=dna_strt, snp_pos=snp_pos,
                                                     data_generator = data_generator,                    
                                                     checkpoint_pth=["./data/sldp/checkpoint/"],
                                                     effect_mode = effect_mode,
                                                     no_pred = True #key, this means it just saves data for SNP pos
                                                    )
        end_t = datetime.datetime.now()
        print(index,(end_t-strt_t).total_seconds())