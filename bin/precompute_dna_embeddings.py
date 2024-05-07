import argparse
from tqdm import tqdm
import pyBigWig
import numpy as np

#pass inputs
# argv
def get_args():
    parser = argparse.ArgumentParser(description="mkeData")
    parser.add_argument('-d', '--dna_embed_dir', default="./data/dna_embed", type=str, help='Path to output for DNA embeddings') 
    args = parser.parse_args()
    return args

args=get_args()

out_pth = args.dna_embed_dir.strip()

#create dir if doesn't exist already
from pathlib import Path
Path(out_pth).mkdir(parents=True, exist_ok=True)

#first let's sort our data loader
#PROJECT_PATH - path to the EnformerCelltyping repo to get relative paths
from EnformerCelltyping.constants import PROJECT_PATH
#import the data generator which will take care of any preprocessing
from EnformerCelltyping.utils import generate_sample
#Transform DNA input data by passing through Enformer
#less memory intensive than using map() on data generator
from EnformerCelltyping.utils import create_enf_chopped_model
enf = create_enf_chopped_model(str(PROJECT_PATH / "data/enformer_model"))#path to enformer model

#need to use any cell so use the one downloaded as test cell
#path to the cell type-specific chromatin accessibility bigWig
chrom_access_pth = str(PROJECT_PATH /'data/demo/Nott19_Microglia_128.bigWig')
#give the name of the cell
cell = 'Nott19_Microglia'

#data generator class for samples
data_generator = generate_sample(
    cells = {cell:chrom_access_pth}, #should be a dict
    data_trans=enf
    )

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

#loop through chormosomes
for ind,chro in enumerate(tqdm(CHROMOSOMES)):
    print('Chromosome: ',chro)
    
    strt = buff_ca_dna
    end = CHROM_LEN[ind]

    while (strt+(WINDOW_SIZE_DNA+buff_ca_dna)<=end):    
        #load X data for input to model
        X = data_generator.load(pos=strt,chro=chro,cell=cell)
        #store DNA embeddings
        np.savez(out_pth+f"/{chro}_{strt}.npz",dna=X['dna'])
        #move strt so looking in new pred window
        strt = strt + TARGET_BP
    
    del X
    
print("Complete")