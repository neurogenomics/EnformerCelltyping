"""
Run SLDP of all the annotations in the annot_path.

python run_sldp.py [annot_path]

annot_path - sumstats with predicted directional effect value ('SNP','A1','A2','Z','N')

Here we use sldp to ask whether alleles that are predicted to change the binding of a 
transcription factor/epigenetic mark have a genome-wide tendency to increase disease risk.

Relevant SLDP parameters:
**GWAS sumstats/QTL parameters** - 
--pss-chr PSS_CHR  
  Path to .pss.gz file, without chromosome number or .pss.gz extension. This is the phenotype 
  that SLDP will analyze.
--sumstats-stem SUMSTATS_STEM
  Path to a .sumstats.gz file, not including ".sumstats.gz" extension. SLDP will process this 
  into a set of .pss.gz files before running.
**Model predictions parameters** - 
--sannot-chr SANNOT_CHR [SANNOT_CHR ...]
  One or more (space-delimited) paths to gzipped annot files, without chromosome number or 
  .sannot.gz/.RV.gz extension. These are the annotations that SLDP will analyze against the 
  phenotype.
**Other** - 
--T T                 
  number of times to sign flip for empirical p-values. Default is 10^6.
"""
import pandas as pd
import numpy as np
import os
import glob
import argparse
import pathlib

#pass inputs
# argv
def get_args():
    parser = argparse.ArgumentParser(description="mkeData")
    parser.add_argument('-s', '--sumstats', default="", type=str, help='Path to GWAS Sumstats')
    parser.add_argument('-p', '--pred', default="", type=str, help='Path to model prediction Sumstats')
    args = parser.parse_args()
    return args

args=get_args()

gwas_sumstats_pth = args.sumstats
pred_sumstats_pth = args.pred
#make sure cell names don't have spaces or tabs in them - would be in the pred
pred_sumstats_pth = pred_sumstats_pth.replace(" ", "")
pred_sumstats_pth = pred_sumstats_pth.replace("\t", "")
gwas_sumstats_pth = gwas_sumstats_pth.replace(" ", "")
gwas_sumstats_pth = gwas_sumstats_pth.replace("\t", "")

#ref files
conf = './metadata/sldp_config.json'

pred_sumstats = os.path.splitext(os.path.splitext(pred_sumstats_pth)[0])[0]
gwas_sumstats = os.path.splitext(os.path.splitext(gwas_sumstats_pth)[0])[0]
gwas_name = os.path.basename(gwas_sumstats)
name = os.path.splitext(os.path.splitext(os.path.basename(pred_sumstats))[0])[0]
sannot_path = './model_results/snp_effects/{}_annots/'.format(name)
outfile_name = './model_results/predictions_sldp/{}_{}'.format(gwas_name,name)


if not pathlib.Path('./model_results/predictions_sldp').is_dir():
    pathlib.Path('./model_results/predictions_sldp').mkdir(parents=True)

pred_ss = pd.read_csv(pred_sumstats_pth,sep='\t')
qtl_ss = pd.read_csv(gwas_sumstats_pth,sep='\t')
# create a column marking df2 values
pred_ss['marker'] = 1
joined = pd.merge(qtl_ss, pred_ss[['SNP','A1','A2','marker']], on=['SNP','A1','A2'], how='left')
filt_gwas_pth = gwas_sumstats+'_for_'+os.path.basename(pred_sumstats)
joined = joined[joined['marker']==1][qtl_ss.columns]
joined.to_csv(filt_gwas_pth+'.sumstats.gz',sep='\t',index=False,compression='gzip')    

sldp_command = 'sldp --sumstats-stem "{}" --sannot-chr "{}" --outfile-stem "{}" --config "{}"'.format(filt_gwas_pth,sannot_path,outfile_name,conf)

print(sldp_command)
os.system(sldp_command)