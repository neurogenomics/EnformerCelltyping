#for each chromosome, find peaks in norm pred and find top 10% of regions 
#that peaks reduce the most without global
#Also find 10% of peaks with no to little change
import pyBigWig
import pathlib
import numpy as np
import os

from dna_hist_mark_pred.constants import (
    CHROM_LEN, 
    CHROMOSOMES,
    HIST_MARKS)

labels = HIST_MARKS 
#remove atac from hist 
labels.remove('atac')

#input mark
import argparse
# argv
def get_args():
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument('-cell', '--cell', default='Monocyte', type=str, help='Cell type')
    args = parser.parse_args()
    return args

args=get_args()

cell=args.cell

SAVE_PATH = pathlib.Path("./model_results")
pred_res = 128
top_x = .1
bot_x = .01

#load values and get diff
for hist_i in labels:
    print(hist_i)
    mono_hist = pyBigWig.open(str(SAVE_PATH /"predictions"/f"{cell}_{hist_i}.bigWig"))
    mono_hist_ng = pyBigWig.open(str(SAVE_PATH /"predictions"/f"{cell}_no_gbl_{hist_i}.bigWig"))
    #save to bigwig, convert to bed after
    bw_top = pyBigWig.open(str(SAVE_PATH /"predictions"/f"{cell}_{hist_i}_top.bigWig"), "w")
    bw_bot = pyBigWig.open(str(SAVE_PATH /"predictions"/f"{cell}_{hist_i}_bot.bigWig"), "w")
    bw_top.addHeader(list(zip(CHROMOSOMES , CHROM_LEN)), maxZooms=0) # zip two turples
    bw_bot.addHeader(list(zip(CHROMOSOMES , CHROM_LEN)), maxZooms=0) # zip two turples
    for ind,the_chr in enumerate(CHROMOSOMES):
        print("***"*5)
        print(the_chr)
        mono_hist_vals = np.nan_to_num(mono_hist.values(the_chr, 0, CHROM_LEN[ind],numpy=True))
        mono_hist_ng_vals = np.nan_to_num(mono_hist_ng.values(the_chr, 0, CHROM_LEN[ind],numpy=True))
        #restrict positions to where there was a peak with gbl then wasn't with none
        peak_act_mask = mono_hist_vals>2
        no_peak_ng_mask = mono_hist_ng_vals<2
        peak_combn_mask = np.logical_and(peak_act_mask, no_peak_ng_mask)
        diff = mono_hist_vals[peak_combn_mask]-mono_hist_ng_vals[peak_combn_mask]
        order = diff.argsort()
        ranks = order.argsort()
        #get indexes from org for diff so can work back to genomic locations
        #get genomic position indicies for diff
        diff_ind = np.where((mono_hist_vals>2)&(mono_hist_ng_vals<2))[0]
        #10% with most drop in peak size
        print(f"median drop top 10%: {np.median(diff[ranks>ranks.shape[0]*.9])}, {diff[ranks>ranks.shape[0]*.9].shape[0]//128} 128bp positions")
        #now get genomic positions for top X%
        top10_genomic_pos = diff_ind[ranks>ranks.shape[0]*(1-top_x)]
        #validate we got them
        print(f"median drop top 10%: {np.median(mono_hist_vals[top10_genomic_pos]-mono_hist_ng_vals[top10_genomic_pos])}, {(mono_hist_vals[top10_genomic_pos]-mono_hist_ng_vals[top10_genomic_pos]).shape[0]//128} 128bp positions")
        #now get positions closest to zero
        diff_abs = np.abs(mono_hist_vals[peak_act_mask]-mono_hist_ng_vals[peak_act_mask])
        order_abs = diff_abs.argsort()
        ranks_abs = order_abs.argsort()
        #get indexes from org for diff so can work back to genomic locations
        #get genomic position indicies for diff
        diff_abs_ind = np.where((mono_hist_vals>2))[0]
        #1% with most drop in peak size
        print(f"median drop bottom 1%: {np.median(diff_abs[ranks_abs<ranks_abs.shape[0]*.01])}, {diff_abs[ranks_abs<ranks_abs.shape[0]*.01].shape[0]//128} 128bp positions")
        #now get genomic positions for bottom X%
        bot10_genomic_pos = diff_abs_ind[ranks_abs<ranks_abs.shape[0]*bot_x]
        #validate we got them
        print(f"median drop bottom 1%: {np.median(mono_hist_vals[bot10_genomic_pos]-mono_hist_ng_vals[bot10_genomic_pos])}, {(mono_hist_vals[bot10_genomic_pos]-mono_hist_ng_vals[bot10_genomic_pos]).shape[0]//128} 128bp positions")
        #remove large files
        del diff_abs_ind, ranks_abs, order_abs, diff_abs, diff_ind, ranks, order, diff 
        del peak_act_mask, no_peak_ng_mask, mono_hist_ng_vals, mono_hist_vals, peak_combn_mask
        #save to bigwig, convert to bed after
        #get starts and ends, chroms
        starts = np.arange(0,CHROM_LEN[ind])
        ends = np.arange(1,CHROM_LEN[ind]+1)
        chroms = np.array([the_chr] * len(starts))
        #create array of zeros and add 1 for chosen positions
        vals = np.zeros(CHROM_LEN[ind])
        vals[top10_genomic_pos]=1.0
        vals_bot = np.zeros(CHROM_LEN[ind])
        vals_bot[bot10_genomic_pos]=1.0
        bw_top.addEntries(chroms, starts,
                ends=ends, values=vals)
        bw_bot.addEntries(chroms, starts,
                ends=ends, values=vals_bot)
    bw_top.close()
    bw_bot.close()

os.makedirs(str(SAVE_PATH /'motif_analysis'), exist_ok=True) 
#now just aggregate
hists = ['h3k27ac', 'h3k4me1', 'h3k4me3', 'h3k9me3', 'h3k27me3', 'h3k36me3']
dat = []
for hist_i in hists:
    bot_motifs = pd.read_csv(str(SAVE_PATH /'motif_analysis' / f'{cell_i}_{hist_i}_bot' /'knownResults.txt'),sep='\t')
    #keep just common cols
    bot_motifs = bot_motifs.iloc[:,0:5]
    bot_motifs['motif']='bot'
    bot_motifs['hist']=hist_i
    top_motifs = pd.read_csv(str(SAVE_PATH /'motif_analysis' / f'{cell_i}_{hist_i}_top' /'knownResults.txt'),sep='\t')
    #keep just common cols
    top_motifs = top_motifs.iloc[:,0:5]
    top_motifs['motif']='top'
    top_motifs['hist']=hist_i
    dat.append(pd.concat([top_motifs, bot_motifs]))
all_dat = pd.concat(dat)    
all_dat.drop_duplicates(inplace=True)
all_dat['cell'] = cell
#get sig res
sig_dat = []
sig_w_bot = []
print("Number of significant, unique TF's:")
for hist_i in hists:
    top_hi = all_dat[(all_dat['hist']==hist_i)&(all_dat['motif']=='top')&((all_dat['q-value (Benjamini)'] < 0.05))]
    bot_hi = all_dat[(all_dat['hist']==hist_i)&(all_dat['motif']=='bot')&((all_dat['q-value (Benjamini)'] < 0.05))]
    print(f'{hist_i}: {top_hi[~top_hi["Motif Name"].isin(bot_hi["Motif Name"].tolist())].shape[0]}')
    sig_dat.append(top_hi[~top_hi['Motif Name'].isin(bot_hi['Motif Name'].tolist())])
    sig_w_bot.append(top_hi)
sig_dat = pd.concat(sig_dat)
sig_dat['cell'] = cell

#now save for cell type enrichment analysis
#save data of sig TF's only due to Global Motifs and run EWCE
cells_sig_all = sig_dat.copy()
cells_sig_all[['TF','Experiment','Tool']] = cells_sig_all['Motif Name'].str.split('/',expand=True)
cells_sig_all.drop(['Motif Name', 'motif','P-value','Tool'], axis=1,inplace=True)
cells_sig_all[['TF','TF2']] = cells_sig_all['TF'].str.split('(',expand=True)
cells_sig_all = cells_sig_all[['cell','TF','TF2','Experiment','Consensus','Log P-value','q-value (Benjamini)']]
cells_sig_all.to_csv(str(SAVE_PATH /'motif_analysis' / 'all_cells_sig_gbl_motifs.csv'), sep=',',index=False)
#save all Tf's tested too to use as a background list for EWCE
cells_all_dat_all = all_dat.copy()
cells_all_dat_all[['TF','Experiment','Tool']] = cells_all_dat_all['Motif Name'].str.split('/',expand=True)
cells_all_dat_all.drop(['motif','P-value','Log P-value','Tool','q-value (Benjamini)','hist','cell'], axis=1,inplace=True)
cells_all_dat_all[['TF','TF2']] = cells_all_dat_all['TF'].str.split('(',expand=True)
cells_all_dat_all.drop_duplicates(inplace=True)
cells_all_dat_all.to_csv(str(SAVE_PATH /'motif_analysis' / 'homer_background_tfs.csv'), sep=',',index=False)
