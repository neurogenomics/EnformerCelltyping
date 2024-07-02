#!/bin/bash

#input parameter
cell=$1

if [[ -z "$cell" ]]; then
	cell="Monocyte"
fi
echo $cell

source ~/anaconda3/etc/profile.d/conda.sh
#GRADIENT ANALYSIS - NEEDS TO BE RUN FIRST
conda activate bigwigtobed 
#Now run rest in parallel for each hist mark
# run in parallel for speed
hists=("h3k27ac","h3k4me1", "h3k9me3")
bed_motifs () {
	local hist_i=$1
	local cell=$2
	#print
	echo "$cell"    
	echo "$hist_i"
	#convert bigwig files to bed
	bigWigToBedGraph "model_results/predictions_bigwigs/${cell}_${hist_i}_grad_top.bigWig" "model_results/predictions_bigwigs/${cell}_${hist_i}_grad_top.bedGraph"
	#now run motif analysis with Homer
	mkdir "model_results/motif_analysis/${cell}_${hist_i}_grad_top"
	findMotifsGenome.pl "model_results/predictions_bigwigs/${cell}_${hist_i}_grad_top.bedGraph" hg19 "model_results/motif_analysis/${cell}_${hist_i}_grad_top/" -size 200
}
export -f bed_motifs

parallel --tmpdir ~/.tmp --jobs 2 bed_motifs {} "$cell" ::: "${hists[@]}"
