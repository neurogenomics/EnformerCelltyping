#model specifics
from EnformerCelltyping.utils import(
    generate_sample,
    predict_snp_effect_sldp_checkpoint,
    create_ref_alt_DNA_window
)
#import constants
from EnformerCelltyping.constants import (
    PROJECT_PATH,
    CHROM_LEN,
    CHROMOSOMES,
    SAMPLES,
    SAVE_PATH,
    HIST_MARKS,
    DATA_PATH)
#package imports
import numpy as np
import tensorflow as tf
import pandas as pd
import math
import glob
import datetime
import os 
import random
import gc
import pathlib

#snp effect mode
effect_mode = 'both'

# Set random seeds.
np.random.seed(101)
tf.random.set_seed(101)
random.seed(101)

#inputs
import argparse
#run in parallel
#pass inputs
# argv
def get_args():
    parser = argparse.ArgumentParser(description="predSnpEff")
    parser.add_argument('-c', '--cell', default='', type=str, help='Cell to predict in')
    parser.add_argument('-s', '--sumstats', default='', type=str, help='SNP sumstats to use')
    parser.add_argument('-r', '--reverse', default='forward', type=str, help='Run through all SNPs in what order')
    args = parser.parse_args()
    return args

args=get_args()

cell=args.cell.lstrip().replace("\t", "") #cell = 'Neutrophil'
sumstats_ext=args.sumstats.lstrip().replace("\t", "") #sumstats_ext='neut_K27AC'
reverse=args.reverse.lstrip().replace("\t", "") #reverse='forward'

#need to update 'T-Cell' and stomach
if cell=='TCell':
    cell='T-Cell'
if cell=='STOMACH1':    
    cell='STOMACH - 1'
sumstats_pth=str(DATA_PATH/f'qtl/{sumstats_ext}_hm3_snps.sumstats.gz')
sumstats_name = os.path.basename(sumstats_pth)
sumstats_name = sumstats_name.split('.')[0]
sumstats = pd.read_csv(sumstats_pth,sep='\t')
sumstats = sumstats.reset_index()

pathlib.Path(SAVE_PATH/'snp_effects/').mkdir(parents=True, exist_ok=True)


#search for other snp predictions in this cell type - any overlapping SNPs can just be copied
sum_ss = glob.glob(str(SAVE_PATH/f'snp_effects/{cell}*_sum.sumstats.gz'))
max_ss = glob.glob(str(SAVE_PATH/f'snp_effects/{cell}*_max.sumstats.gz'))

#clean up dir - move all to one pd if high number of files
if (len(sum_ss)+len(max_ss)>2_000):
    #split by cell and label being tested and the sumstats they were tested on
    #join together pd df's based on agg type, cell type and histone mark
    del_files = []
    #don't delete agg files
    keep_files= []
    #run for max first
    max_ss_sumstats = set([os.path.basename(i).split('.')[0].split('_',2)[2] for i in max_ss])
    for l_ind,l_i in enumerate(labels):
        all_dat_hist = []
        for ss_ind, ss_i in enumerate(max_ss_sumstats):
            filename = str(SAVE_PATH/f'snp_effects/{cell}_{l_i}_{ss_i}.sumstats.gz')
            #save file to be deleted once data successfully saved
            del_files.append(filename)
            all_dat_hist.append(pd.read_csv(filename, index_col=None, header=0,sep='\t'))
        all_dat_hist = pd.concat(all_dat_hist,axis=0, ignore_index=True)
        #remove any dups
        all_dat_hist.drop_duplicates(subset=['SNP','A1','A2'],inplace = True,keep='first')
        #now save as one df
        all_dat_hist.to_csv(SAVE_PATH/f'snp_effects/{cell}_{l_i}_ALL_CHECKPOINT_indexALL_ALL_ALL_max.sumstats.gz',
                                 sep='\t',index=False,compression='gzip')
        keep_files.append(str(SAVE_PATH/f'snp_effects/{cell}_{l_i}_ALL_CHECKPOINT_indexALL_ALL_ALL_max.sumstats.gz'))
    #run for sum
    sum_ss_sumstats = set([os.path.basename(i).split('.')[0].split('_',2)[2] for i in sum_ss])
    for l_ind,l_i in enumerate(labels):
        all_dat_hist = []
        for ss_ind, ss_i in enumerate(sum_ss_sumstats):
            filename = str(SAVE_PATH/f'snp_effects/{cell}_{l_i}_{ss_i}.sumstats.gz')
            #save file to be deleted once data successfully saved
            del_files.append(filename)
            all_dat_hist.append(pd.read_csv(filename, index_col=None, header=0,sep='\t'))
        all_dat_hist = pd.concat(all_dat_hist,axis=0, ignore_index=True)
        #remove any dups
        all_dat_hist.drop_duplicates(subset=['SNP','A1','A2'],inplace = True,keep='first')
        #now save as one df
        all_dat_hist.to_csv(SAVE_PATH/f'snp_effects/{cell}_{l_i}_ALL_CHECKPOINT_indexALL_ALL_ALL_sum.sumstats.gz',
                                 sep='\t',index=False,compression='gzip')
        keep_files.append(str(SAVE_PATH/f'snp_effects/{cell}_{l_i}_ALL_CHECKPOINT_indexALL_ALL_ALL_sum.sumstats.gz'))
    #now delete all files
    #remove keep files
    del_files = list(set(del_files) - set(keep_files))
    for file_i in del_files:
        if os.path.exists(file_i):
            os.remove(file_i)
    del all_dat_hist    
    #re-search
    sum_ss = glob.glob(str(SAVE_PATH/f'snp_effects/{cell}*_sum.sumstats.gz'))
    max_ss = glob.glob(str(SAVE_PATH/f'snp_effects/{cell}*_max.sumstats.gz')) 

if len(sum_ss)>0:
    #split by cell and label being tested and the sumstats they were tested on
    sum_ss_cell = set([os.path.basename(i).split('.')[0].split('_',2)[0] for i in sum_ss])
    sum_ss_label = set([os.path.basename(i).split('.')[0].split('_',2)[1] for i in sum_ss])
    sum_ss_sumstats = set([os.path.basename(i).split('.')[0].split('_',2)[2] for i in sum_ss])
    li_sum = []
    #checkpoints saved with hist marks separate, need to join these back
    for c_i in sum_ss_cell: #this will always just be one value and equal to cell
        for ss_i in sum_ss_sumstats:
            for l_ind,l_i in enumerate(labels):
                filename = str(SAVE_PATH/f'snp_effects/{c_i}_{l_i}_{ss_i}.sumstats.gz')
                df_li = pd.read_csv(filename, index_col=None, header=0,sep='\t')
                df_li.rename(columns={'Z':'Z_'+l_i}, inplace=True)
                if l_ind == 0:
                    df = df_li.copy()
                else:
                    #merge so all hist marks for one checkpoint back in one pd df
                    #merge so each label has it's own column
                    df = df.merge(df_li,how='inner',
                                  left_on=['SNP','A1','A2','N'],
                                  right_on=['SNP','A1','A2','N'])
            #combine all Z's into one column with array of values
            df['Z'] = df[['Z_'+l for l in labels]].values.tolist()
            df.drop(['Z_'+l for l in labels], axis=1, inplace=True)          
            li_sum.append(df)
    prev_sum = pd.concat(li_sum, axis=0, ignore_index=True)
    #remove any dups
    prev_sum.drop_duplicates(subset=['SNP','A1','A2'],inplace = True)
    #split by cell and label being tested and the sumstats they were tested on
    max_ss_cell = set([os.path.basename(i).split('.')[0].split('_',2)[0] for i in max_ss])
    max_ss_label = set([os.path.basename(i).split('.')[0].split('_',2)[1] for i in max_ss])
    max_ss_sumstats = set([os.path.basename(i).split('.')[0].split('_',2)[2] for i in max_ss])
    li_max = []
    for c_i in max_ss_cell:
        for ss_i in max_ss_sumstats:
            for l_ind,l_i in enumerate(labels):
                filename = str(SAVE_PATH/f'snp_effects/{c_i}_{l_i}_{ss_i}.sumstats.gz')
                df_li = pd.read_csv(filename, index_col=None, header=0,sep='\t')
                df_li.rename(columns={'Z':'Z_'+l_i}, inplace=True)
                if l_ind == 0:
                    df = df_li.copy()
                else:
                    #merge so each label has it's own column
                    df = df.merge(df_li,how='inner',
                                  left_on=['SNP','A1','A2','N'],
                                  right_on=['SNP','A1','A2','N'])
            #combine all Z's into one column with array of values
            df['Z'] = df[['Z_'+l for l in labels]].values.tolist()
            df.drop(['Z_'+l for l in labels], axis=1, inplace=True)  
            li_max.append(df)
    del df_li, df
    prev_max = pd.concat(li_max, axis=0, ignore_index=True)
    del li_max, li_sum
    #remove any dups
    prev_max.drop_duplicates(subset=['SNP','A1','A2'],inplace = True,keep='first')
    #filter to this sumstats
    prev_sum = prev_sum.merge(sumstats[['SNP','A1','A2','index']],how='inner',
                              left_on=['SNP','A1','A2'],right_on=['SNP',"A1","A2"])
    prev_max = prev_max.merge(sumstats[['SNP','A1','A2','index']],how='inner',
                              left_on=['SNP','A1','A2'],right_on=['SNP',"A1","A2"])
    #reorder prev_sum and max so same as sumstats
    prev_sum.sort_values(by=['index'],inplace=True)
    prev_max.sort_values(by=['index'],inplace=True)
    #assume same for both effect types so just use one to filter
    sumstats_new = sumstats.merge(prev_max[['SNP','A1','A2']],how='outer',
                                  left_on=['SNP','A1','A2'],right_on=['SNP',"A1","A2"],
                                  indicator=True).query('_merge=="left_only"')
    sumstats_new.drop('_merge', axis=1, inplace=True,errors='ignore')
    sumstats_new.drop('index', axis=1, inplace=True,errors='ignore')
    sumstats_new = sumstats_new.reset_index()
    #now get rows from sumstats that are in prev and recreate sumstats so the first rows are these
    sumstats_old = sumstats.merge(prev_max[['SNP','A1','A2']],how='inner',
                              left_on=['SNP','A1','A2'],right_on=['SNP',"A1","A2"])
    sumstats = pd.concat([sumstats_old,sumstats_new], axis=0, ignore_index=True)
    sumstats.drop('index', axis=1, inplace=True,errors='ignore')
    sumstats = sumstats.reset_index()
    #Now add the effect values for the SNPs already encountered to the results
    all_snp_agg_eff_sum = prev_sum['Z'].tolist()
    all_snp_agg_eff_max = prev_max['Z'].tolist()
    del prev_max, prev_sum, sumstats_old
else:
    sumstats_new = sumstats.copy()
    all_snp_agg_eff_sum = []
    all_snp_agg_eff_max = []

if reverse == 'reverse':
    sumstats_new = sumstats_new.iloc[::-1]
    sumstats_new.drop('_merge', axis=1, inplace=True,errors='ignore')
    sumstats_new.drop('index', axis=1, inplace=True,errors='ignore')
    sumstats_new = sumstats_new.reset_index()


#load Enformer celltyping model
from EnformerCelltyping.enf_celltyping import Enformer_Celltyping

model = Enformer_Celltyping(use_prebuilt_model=True,
                            enf_celltyping_pth = str(PROJECT_PATH /'EnformerCelltyping'/'enformer_celltyping_weights')
                           )
#compile loaded model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
              loss=tf.keras.losses.mean_squared_error,
              #mse for each hist mark
              metrics=['mse'])


#Transform DNA input data by passing through Enformer
#less memory intensive than using map() on data generator
from EnformerCelltyping.utils import create_enf_chopped_model
enf = create_enf_chopped_model(str(PROJECT_PATH / "data/enformer_model"))#path to enformer model

#data generator class for samples
data_generator = generate_sample(
    cells = SAMPLES,
    data_trans=enf,
    arcsin_trans = False,
    reverse_complement = True,
    rand_seq_shift = True,
    return_y = False,
    rtn_rand_seq_shift_amt=True,
    )

print(f"SNPS: {sumstats_new.shape[0]}")
#loop through each SNP prediciting it's effective change
for index, row in sumstats_new.iterrows():
    print(row['SNP'])
    strt_time = datetime.datetime.now()
    dna_strt, snp_pos = create_ref_alt_DNA_window(chro = 'chr'+str(row['CHR']), 
                                              pos = row['BP'])
    #if SNP too close to edge just skip
    if len(dna_strt)>0:
        #can plot if we want:
        #plot_snp_dna_window(dna_strt,snp_pos)
        #Predict the effective difference of the alternative allele
        agg_eff_max,agg_eff_sum = predict_snp_effect_sldp_checkpoint(model=model,
                                                                     alt=row['A2'],cell = cell, 
                                                                     chro='chr'+str(row['CHR']),
                                                                     dna_strt=dna_strt, snp_pos=snp_pos,
                                                                     data_generator = data_generator,
                                                                     checkpoint_pth = str(DATA_PATH/'sldp'/'checkpoint'),
                                                                     effect_mode = effect_mode)
        all_snp_agg_eff_max.append(agg_eff_max)
        all_snp_agg_eff_sum.append(agg_eff_sum)
    else:
        #can't pred on snp as it's at edge so delete it
        sumstats_new.drop(index, inplace=True)  
        #drop from full too
        sumstats.drop('index', axis=1, inplace=True)
        sumstats = sumstats[~((sumstats.SNP==row['SNP']) &
                              (sumstats.CHR==row['CHR']) &
                              (sumstats.BP==row['BP']) &
                              (sumstats.A2==row['A2']))].reset_index()
    #help with memory usage, clear mem every 50
    if (index%50==0 and not index==0):
        tf.keras.backend.clear_session()
        gc.collect()
    #save progress in case run crashes
    if (index%20_000==0 and not index==0):
        print(index)
        print("saving checkpoint")
        keep_cols = ['SNP','A1','A2','Z','N']
        for h_i,hist_i in enumerate(labels):
            print(hist_i)
            res_i = sumstats[0:len(all_snp_agg_eff_max)].copy()
            for eff_i in ['max','sum']:
                if eff_i == 'max':
                    z = [x[h_i] for x in all_snp_agg_eff_max]
                else:#sum
                    z = [x[h_i] for x in all_snp_agg_eff_sum]
                res_i['Z'] = z
                #save sumstats
                                #save sumstats
                res_i = res_i[keep_cols]
                res_i.to_csv(SAVE_PATH/f'snp_effects/{cell}_{hist_i}_{sumstats_name}_CHECKPOINT_index{index}_{reverse}_{eff_i}.sumstats.gz',
                             sep='\t',index=False,compression='gzip')
        print((datetime.datetime.now()-strt_time).total_seconds())  
#now just update sumstats dataset with correct columns and save
#we want columns SNP A1 A2 Z N, separated for each 
#output channel (binding protein)
#Now save 
cell = cell.replace(" ", "")
keep_cols = ['SNP','A1','A2','Z','N']
for ind,hist_i in enumerate(labels):
    res_i = sumstats.copy()
    for eff_i in ['max','sum']:
        if eff_i == 'max':
            z = [x[ind] for x in all_snp_agg_eff_max]
        else:#sum
            z = [x[ind] for x in all_snp_agg_eff_sum]
        res_i['Z'] = z
        #save split by chr as .sannot.gz files
        pathlib.Path(SAVE_PATH/f'snp_effects/{cell}_{hist_i}_{sumstats_name}_{eff_i}_annots').mkdir(parents=True, exist_ok=True)
        for chro in set(sumstats['CHR']):
            f_name_annot = SAVE_PATH/f'snp_effects/{cell}_{hist_i}_{sumstats_name}_{eff_i}_annots/{str(chro)}.sannot.gz'
            res_chr_i = res_i.copy()
            res_chr_i = res_chr_i[res_chr_i['CHR']==chro]
            res_chr_i = res_chr_i[['CHR','BP','SNP','A1','A2','Z']]
            #make sure no na's
            res_chr_i = res_chr_i[~res_chr_i['Z'].isna()]
            res_chr_i.rename(columns={'Z': f'EC_{cell}_{hist_i}_{eff_i}'},inplace=True)
            res_chr_i.to_csv(str(f_name_annot),sep='\t',index=False,compression='gzip')
        #save sumstats
        res_i_ = res_i.copy()[keep_cols]
        #make sure no na's
        res_i_ = res_i_[~res_i_['Z'].isna()]
        res_i_.to_csv(SAVE_PATH/f'snp_effects/{cell}_{hist_i}_{sumstats_name}_{eff_i}.sumstats.gz',
                     sep='\t',index=False,compression='gzip')
