import argparse
from tqdm import tqdm
import pyBigWig
import numpy as np

#pass inputs
# argv
def get_args():
    parser = argparse.ArgumentParser(description="mkeData")
    parser.add_argument('-c', '--cell', default="", type=str, help='Name of Cell. to be predicted for')
    parser.add_argument('-p', '--path_chrom_access', default="", type=str, help='Path to chromatin accessability for the cell type')
    parser.add_argument('-o', '--output_dir', default="", type=str, help='Path to output')
    args = parser.parse_args()
    return args

args=get_args()

cell = args.cell.strip()
out_pth = args.output_dir.strip()
chrom_access_pth = args.path_chrom_access.strip()


#create dir if doesn't exist already
from pathlib import Path
Path(out_pth).mkdir(parents=True, exist_ok=True)

#first let's sort our data loader
#PROJECT_PATH - path to the EnformerCelltyping repo to get relative paths
from EnformerCelltyping.constants import PROJECT_PATH
#import the data generator which will take care of any preprocessing
from EnformerCelltyping.utils import generate_sample

#data generator class for samples
data_generator = generate_sample(
    cells = {cell:chrom_access_pth}, #should be a dict
    )

#load Enformer celltyping model
import tensorflow as tf
from EnformerCelltyping.enf_celltyping import Enformer_Celltyping

#load histone marks from constants so the ordering of predictions is known
from EnformerCelltyping.constants import PRED_HIST_MARKS

hist_marks = PRED_HIST_MARKS

model = Enformer_Celltyping(enf_path=str(PROJECT_PATH / "data/enformer_model"),#path to enformer model
                            use_prebuilt_model=True,
                            enf_celltyping_pth = str(PROJECT_PATH /'EnformerCelltyping'/'enformer_celltyping_weights')
                           )
#compile loaded model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
              loss=tf.keras.losses.mean_squared_error,
              #mse for each hist mark
              metrics=['mse'])


#Now let's predict genome-wide
#need chromosomes and their lengths
from EnformerCelltyping.constants import CHROM_LEN, CHROMOSOMES
#Model will predict on chromsomes 1-22 (not sex chromosomes)
#EC predicts in a funnel so it can't predict at start or very end or chrom
#need to account for this
from EnformerCelltyping.constants import WINDOW_SIZE_DNA, WINDOW_SIZE_LCL_CA
buff_ca_dna = (WINDOW_SIZE_LCL_CA-WINDOW_SIZE_DNA)//2
#also need to know prediction window size in base-pairs
from EnformerCelltyping.constants import TARGET_BP

#need to save res to conv to a bigWig later
signal_all = dict(zip(hist_marks,[[] for i in hist_marks]))
#open connections to the bigwig files to store results
pred_bigwigs = {i:pyBigWig.open(out_pth+f"{cell}_{i}.bigWig", "w") for i in hist_marks}
#add header to each, all chromosomes and their sizes - need to do before adding values
for hist_i,hist in enumerate(hist_marks):
    pred_bigwigs[hist].addHeader(list(zip(CHROMOSOMES,CHROM_LEN)))

#loop through chormosomes
for ind,chro in enumerate(tqdm(CHROMOSOMES)):
    print('Chromosome: ',chro)
    
    chro_all = []
    dna_strt_all = []
    strt = buff_ca_dna
    end = CHROM_LEN[ind]
    
    while (strt+(WINDOW_SIZE_DNA+buff_ca_dna)<=end):
        #load X data for input to model
        X = data_generator.load(pos=strt,chro=chro,cell=cell)
        #predict
        pred = model.predict(X)
        #store res
        chro_all.append(chro)
        dna_strt_all.append(strt)
        for hist_i,hist in enumerate(hist_marks):
            signal_all[hist].append(pred[0,:,hist_i])
        #move strt so looking in new pred window
        strt = strt + TARGET_BP
    #now let's save the values   
    #need to move starts to start of prediciton region
    dna_pred_buff = (WINDOW_SIZE_DNA-TARGET_BP)//2
    pred_strt_all = [i+dna_pred_buff for i in dna_strt_all]
    #now need a numpy array from first pred strt to (last pred strt+target length)
    pos_bp = np.arange(pred_strt_all[0],(pred_strt_all[-1]+TARGET_BP),step=128)
    #need end pos too
    pos_bp_end = pos_bp + 128
    #need to input chromosome too
    pos_chro = np.repeat(chro, pos_bp.shape[0])
    #need to expand the signals and convert from a list to a single np array for each
    for hist_i,hist in enumerate(hist_marks):
        signal_all[hist] = np.concatenate(signal_all[hist], axis=0)
        #now let's save
        pred_bigwigs[hist].addEntries(pos_chro, pos_bp, ends=pos_bp_end, 
                                      values=signal_all[hist])
        #reset for next loop
        signal_all[hist] = []
    del pos_bp, pos_bp_end, pos_chro, pred_strt_all
    
#finally once done predicting close connection
for hist_i,hist in enumerate(hist_marks): 
    pred_bigwigs[hist].close()