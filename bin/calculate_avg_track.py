"""
This script is modified from Leopard to create an average dnase bigwig
The average is in training the model to infer the difference 
from the training cell type and the average and pass this as an input track.
"""

import os
import sys
import numpy as np
import pyBigWig
import argparse
import datetime

from EnformerCelltyping.constants import(
        CHROMOSOMES, 
        CHROM_LEN, 
        AVG_DATA_PATH,
        TO_AVG_DATA,
        SAMPLE_NAMES,
        HIST_MARKS
        )        

#input mark
# argv
def get_args():
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument('-m', '--MARK', default='atac', type=str, help='Epigenetic mark to be averaged')
    args = parser.parse_args()
    return args

args=get_args()

MARK=args.MARK.lower()
#assert it's a valid choice
assert MARK in HIST_MARKS, f"Must choose valid Epigenetic mark to average from {HIST_MARKS}"

print("starting calculation of avg.bigwig")
print(datetime.datetime.now())


chr_len_grch37={}
for i in np.arange(len(CHROMOSOMES)):
    chr_len_grch37[CHROMOSOMES[i]]=CHROM_LEN[i]

OUTPUT = AVG_DATA_PATH[MARK]
MARK_DATA = TO_AVG_DATA[MARK]

REF_GENOME = "grch37"
AVG_BP = 128

# Exclude datasets from cells to be used for validation (3 immune cells and microglia)
excl = ['MonocyteCD4CD16', 'MatureNeutrophil', 'CD4T', 'Micorglia']
SAMPLE_NAMES = np.delete(SAMPLE_NAMES, np.isin(SAMPLE_NAMES,excl))
#files to be averaged
#append bp averaging amount
train_dnase = np.array([x1 + x2 for x1,x2 in zip(SAMPLE_NAMES,[f'_{AVG_BP}']*len(SAMPLE_NAMES))])
#filter to training dnase files
sample_files = {k:v for (k,v) in MARK_DATA.items() if np.any(train_dnase == k)}

if REF_GENOME=='grch37':
    chr_len = chr_len_grch37
    num_bp = CHROM_LEN
else: # grch37 & grch38 only for now
    chr_len = chr_len_grch38
    num_bp = CHROM_LEN_38

#input_all=args.input
bw_output = pyBigWig.open(str(OUTPUT),'w')

bw_output.addHeader(list(zip(CHROMOSOMES , num_bp)), maxZooms=0) # zip two turples


for the_chr in CHROMOSOMES:
    print('calculating average for ' + the_chr)
    x = np.zeros(chr_len[the_chr])
    for the_input in sample_files:
        print('loading ' + the_input)
        bw=pyBigWig.open(str(sample_files[the_input]))
        tmp=np.array(bw.values(the_chr,0,chr_len[the_chr]))
        tmp[np.isnan(tmp)]=0 # set nan to 0
        x += tmp
        bw.close()
    x=x/len(sample_files)
    ## convert to bigwig format
    # pad two zeroes
    z=np.concatenate(([0],x,[0]))
    # find boundary
    starts=np.where(np.diff(z)!=0)[0]
    ends=starts[1:]
    starts=starts[:-1]
    vals=x[starts]
    if starts[0]!=0:
        ends=np.concatenate(([starts[0]],ends))
        starts=np.concatenate(([0],starts))
        vals=np.concatenate(([0],vals))
    if ends[-1]!=chr_len[the_chr]:
        starts=np.concatenate((starts,[ends[-1]]))
        ends=np.concatenate((ends,[chr_len[the_chr]]))
        vals=np.concatenate((vals,[0]))
    # write 
    chroms = np.array([the_chr] * len(vals))
    bw_output.addEntries(chroms, starts, ends=ends, values=vals)

bw_output.close()

print("calculation of avg.bigwig complete")
print(datetime.datetime.now())
