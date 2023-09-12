"""
This script is modified from [Leopard](https://github.com/GuanLab/Leopard) 
to create an average dnase bigwig
The average, quant and delta are used in training the model. The average is
also used as a benchmark for predictive performance.
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
        AVG_DATA,
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

print("starting calculation of avg.bigwig and quantiles")
print(datetime.datetime.now())


chr_len_grch37={}
for i in np.arange(len(CHROMOSOMES)):
    chr_len_grch37[CHROMOSOMES[i]]=CHROM_LEN[i]

OUTPUT = AVG_DATA_PATH[MARK]
MARK_DATA = AVG_DATA[MARK]

REF_GENOME = "grch37"
AVG_BP = 128

# Exclude datasets from cells to be used for validation (3 immune cells and microglia)
excl = ['MonocyteCD4CD16', 'MatureNeutrophil', 'CD4T', 'Microglia']
SAMPLE_NAMES = np.delete(SAMPLE_NAMES, np.isin(SAMPLE_NAMES,excl))
#files to be averaged
#append bp averaging amount
train_mark = np.array([x1 + x2 for x1,x2 in zip(SAMPLE_NAMES,[f'_{AVG_BP}']*len(SAMPLE_NAMES))])
#filter to training files
sample_files = {k:v for (k,v) in MARK_DATA.items() if np.any(train_mark == k)}

if REF_GENOME=='grch37':
    chr_len = chr_len_grch37
    num_bp = CHROM_LEN
else: # grch37 & grch38 only for now
    chr_len = chr_len_grch38
    num_bp = CHROM_LEN_38

#input_all=args.input
bw_output = pyBigWig.open(str(OUTPUT),'w')
bw_output.addHeader(list(zip(CHROMOSOMES , num_bp)), maxZooms=0) # zip two turples

#add quantile files
quant_files={
    quant:pyBigWig.open(str(OUTPUT.with_name(OUTPUT.stem + f'_quantile_{quant}' + OUTPUT.suffix)),'w')
    for quant in range(0,10)
}
#add headers
for i in range(0,10):
    quant_files[i].addHeader(list(zip(CHROMOSOMES , num_bp)), maxZooms=0) # zip two turples


#first open a connection to all bigwigs
print('loading files')
cell_dat={
    cell:pyBigWig.open(str(sample_files[cell]))
    for cell in sample_files
}

#info for quants
def round_nearest_half_get_bin_ind(arr,lim=4.5):
            #set limit for biggest values
            arr[arr>lim] = lim
            return np.int16(np.round(arr * 2))
n_quants = 10

for the_chr in CHROMOSOMES:
    print('calculating average for ' + the_chr)
    x = np.zeros(chr_len[the_chr])
    x_quant = np.zeros((n_quants,chr_len[the_chr]),dtype=np.int16)
    quant_dat = []
    for the_input in sample_files:
        tmp=np.array(cell_dat[the_input].values(the_chr,0,chr_len[the_chr]))
        tmp[np.isnan(tmp)]=0 # set nan to 0
        x += tmp
        #arc-sinh trans and count up which quantile each position falls into
        tmp = np.arcsinh(tmp)
        tmp = round_nearest_half_get_bin_ind(tmp)
        x_quant[tmp,np.arange(len(tmp))] += 1
        del tmp
    #now convert the counts in x_quant to percentages
    x_quant = x_quant/x_quant.sum(axis=0)
    x=x/len(sample_files)
    ## convert to bigwig format
    # pad two zeroes
    z=np.concatenate(([0],x,[0]))
    # find boundary
    starts=np.where(np.diff(z)!=0)[0]
    ends=starts[1:]
    starts=starts[:-1]
    vals=x[starts]
    #same for quants
    x_quant = x_quant[:,starts]
    #make sure first and last entry are the edges of the chrom
    if starts[0]!=0:
        ends=np.concatenate(([starts[0]],ends))
        starts=np.concatenate(([0],starts))
        vals=np.concatenate(([0],vals))
        x_quant=np.hstack([np.zeros((n_quants,1),dtype=np.int16), x_quant])
    if ends[-1]!=chr_len[the_chr]:
        starts=np.concatenate((starts,[ends[-1]]))
        ends=np.concatenate((ends,[chr_len[the_chr]]))
        vals=np.concatenate((vals,[0]))
        x_quant=np.hstack([x_quant, np.zeros((n_quants,1),dtype=np.int16)])
    # write 
    chroms = np.array([the_chr] * len(vals))
    bw_output.addEntries(chroms, starts, ends=ends, values=vals)
    #now write for each quantile
    for i in range(0,10):
        quant_files[i].addEntries(chroms, starts, ends=ends, values=x_quant[i,:])

bw_output.close()

print("calculation of avg.bigwig and quantile bigwigs complete")
print(datetime.datetime.now())