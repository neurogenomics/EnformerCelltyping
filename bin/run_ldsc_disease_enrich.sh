#!/bin/bash

## Running ldsc - disease enrichment  analysis 
export sumstats=('processed_ebi-a-GCST90027158' 'processed_ieu-b-5099' 'AD_Jansen2019_ldsc' 'processed_ieu-a-1185' 'processed_mdd_excl_23andme' 'processed_ebi-a-GCST90002351' 'processed_intelligence' 'processed_cognitive_function' 'processed_GCST90029017' 'processed_ibd')

#note for --ref-ld-chr put in all annotation files here with the baseline model (note using v2.2 of baseline)
#for sumstats line, should have .gz \ at end, NOTE I've moded the files to match a MSS update to avoid rerunning munge_sumstats.py
for disease_type in "${sumstats[@]}"
do
  echo ${disease_type}
  python ./ldsc/ldsc.py \
  --h2 ./${disease_type}.sumstats \
  --ref-ld-chr ./pth/to/ldsc/ref/file/1000G_Phase3_baselineLD_v2.2_ldscores/baselineLD.,./annots/Nott19_Microglia_atac.,./annots/Nott19_Neuron_atac.,./annots/Nott19_Microglia_h3k27ac.,./annots/Nott19_Neuron_h3k27ac.,./annots/pred_Nott19_Microglia_h3k27ac.,./annots/pred_Nott19_Neuron_h3k27ac.,./annots/Nott19_Oligodendrocyte_atac.,./annots/Nott19_Oligodendrocyte_h3k27ac.,./annots/pred_Nott19_Oligodendrocyte_h3k27ac. \
  --out ./heritability/${disease_type} \
  --overlap-annot  \
  --frqfile-chr ./pth/to/ldsc/ref/file/1000G_Phase3_frq/1000G.EUR.QC. \
  --w-ld-chr ./pth/to/ldsc/ref/file/1000G_Phase3_weights_hm3_no_MHC/weights.hm3_noMHC. \
  --print-coefficients
done
