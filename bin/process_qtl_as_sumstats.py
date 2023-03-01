from EnformerCelltyping.constants import DATA_PATH,CHROMOSOME_DATA

import pandas as pd
import os.path

import argparse
#run in parallel
#pass inputs
# argv
def get_args():
    parser = argparse.ArgumentParser(description="mkeData")
    parser.add_argument('-p', '--qtl_pth', default='./data/qtl/mono_K27AC_log2rpm_31052017_bedmap_peer_10_all_summary.Beta_changed.SE.Eigen.pval.txt.gz',
                        type=str, help='path to QTL dataset')
    args = parser.parse_args()
    return args

args=get_args()

qtl_pth=args.qtl_pth

#filter to hapmap3 SNPs
#using hm3 rsid's - taken from SLDP paper reference - same as SLDP uses
hm3= pd.read_csv('./data/sldp/1000G_hm3_noMHC.rsid',header=None)
hm3.rename(columns={0: 'SNP'}, inplace=True)

qtl = pd.read_csv(qtl_pth,sep='\t',
                  usecols=['SNP','CHR','BP','A1','A2',
                           effect_reg_col_name,adj_p_val_col_name,eff_col])  
#remove any QTLs where distance is greater than model's limit
#can't predict if effected position is greater than models predicitive window
#get end of effected region - i.e. end of peak to be safe
qtl['end_reg'] = qtl[effect_reg_col_name].str.split(":").str[2]
qtl['end_reg'] = qtl['end_reg'].astype('int')
window_size = 1562*128
qtl = qtl[abs(qtl['BP']-qtl['end_reg'])<=window_size//2]
#only going to run prediction for each unique SNP (based on chr, pos, A1, A2)
qtl.drop(qtl.columns.difference(['SNP','CHR','BP','A1','A2',eff_col]), 1, inplace=True)
#split out to multiple orws where non-bi-allelic SNP => A2 = AT...
qtl['A2']=qtl.A2.apply(lambda x: list(x))
qtl = qtl.explode('A2').reset_index(drop=True)
#remove where alt = ref
qtl = qtl[qtl['A1']!=qtl['A2']]
qtl.drop_duplicates(inplace=True)
qtl.reset_index()
qtl.drop('index', axis=1, inplace=True)
#filter to hm3
qtl_hm3 = qtl.merge(hm3,how='inner',left_on=['SNP'], right_on=['SNP'])
qtl_hm3.drop_duplicates(subset=['SNP','CHR','BP','A1','A2'],inplace = True)
qtl_hm3 = qtl_hm3.reset_index()
qtl_hm3.drop('index', axis=1, inplace=True)
#Need to add in N since missing for blood cells
cell=os.path.basename(qtl_pth).split('_')[0]
if not 'N' in qtl_hm3.columns:
    if(cell=="neut"):
        N = 165
    elif(cell=="tcel"):
        N = 125
    elif(cell=="mono"):
        N = 158
    qtl_hm3['N'] = N
qtl_hm3.to_csv(qtl_pth.split('log2rpm')[0]+'hm3_snps.sumstats.gz',compression='gzip',index=False,sep='\t')