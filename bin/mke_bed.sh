#!/bin/bash

#conda activate bigwigtobed


export files=("./annots/Nott19_Microglia_atac" "./annots/Nott19_Neuron_atac" "./annots/Nott19_Microglia_h3k27ac" "./annots/Nott19_Neuron_h3k27ac" "./annots/pred_Nott19_Microglia_h3k27ac" "./annots/pred_Nott19_Neuron_h3k27ac" "./annots/Nott19_Oligodendrocyte_atac" "./annots/Nott19_Oligodendrocyte_h3k27ac" "./annots/pred_Nott19_Oligodendrocyte_h3k27ac")

for file_i in "${files[@]}"
do	
    for chrom in {1..22}
    do
        bigWigToBedGraph -chrom="chr${chrom}" "${file_i}.bigWig" "${file_i}.${chrom}.bedGraph"
done
done

