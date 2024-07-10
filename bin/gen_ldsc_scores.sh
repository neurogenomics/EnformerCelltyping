#!/bin/bash

## Generating ldsc scores 
export annot_beds=("Nott19_Microglia_atac" "Nott19_Neuron_atac" "Nott19_Microglia_h3k27ac" "Nott19_Neuron_h3k27ac" "pred_Nott19_Microglia_h3k27ac" "pred_Nott19_Neuron_h3k27ac" "Nott19_Oligodendrocyte_atac" "Nott19_Oligodendrocyte_h3k27ac" "pred_Nott19_Oligodendrocyte_h3k27ac")

for bed_i in "${annot_beds[@]}"
do
    for chrom in {1..22}
    do
        python ./ldsc/ldsc.py \
        --l2 \
        --bfile ./pth/to/ldsc/ref/file/1000G_EUR_Phase3_plink/1000G.EUR.QC.${chrom} \
        --ld-wind-cm 1 \
        --annot ./annots/${bed_i}.${chrom}.annot.gz  \
        --thin-annot \
        --out ./annots/${bed_i}.${chrom} \
        --print-snps ./pth/to/list/of/rs/ids/of/snps/to/filter/results/to/we/used/hapmap3/list.txt #hapmap3 snps
done
done
