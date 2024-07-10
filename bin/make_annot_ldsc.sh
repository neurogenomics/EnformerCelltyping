#!/bin/bash

## Creating an annot file 
export annot_beds=("Nott19_Microglia_atac" "Nott19_Neuron_atac" "Nott19_Microglia_h3k27ac" "Nott19_Neuron_h3k27ac" "pred_Nott19_Microglia_h3k27ac" "pred_Nott19_Neuron_h3k27ac" "Nott19_Oligodendrocyte_atac" "Nott19_Oligodendrocyte_h3k27ac" "pred_Nott19_Oligodendrocyte_h3k27ac" )

for bed_i in "${annot_beds[@]}"
do
    for chrom in {1..22}
    do
        ## Step 1: Creating an annot file
        python ./ldsc/make_annot.py \
        --bed-file ./annots/${bed_i}.${chrom}.bed \
        --bimfile ./pth/to/ldsc/ref/file/1000G_EUR_Phase3_plink/1000G.EUR.QC.${chrom}.bim \
        --annot-file ./annots/${bed_i}.${chrom}.annot.gz
done
done
