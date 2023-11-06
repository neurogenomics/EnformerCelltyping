#!/bin/bash

#input parameter
cell=$1

if [[ -z "$cell" ]]; then
	cell="Monocyte"
fi
#echo $cell

source ~/anaconda3/etc/profile.d/conda.sh
conda activate enformer
#run python to create bigwig files of changing regions for each hist mark first
python motif_analysis.py -c ${cell}
conda deactivate

conda activate bigwigtobed 
#Now run rest in parallel for each hist mark
# run in parallel for speed
hists=("h3k27ac" "h3k4me3" "h3k36me3" "h3k9me3" "h3k4me1" "h3k27me3")
bed_motifs () {
	local hist_i=$1
	local cell=$2
	#print
	echo "$cell"    
	echo "$hist_i"
	#convert bigwig files to bed
	bigWigToBedGraph "model_results/predictions/${cell}_${hist_i}_bot.bigWig" "model_results/predictions/${cell}_${hist_i}_bot.bedGraph"
	bigWigToBedGraph "model_results/predictions/${cell}_${hist_i}_top.bigWig" "model_results/predictions/${cell}_${hist_i}_top.bedGraph"
	#now run motif analysis with Homer
	mkdir "model_results/motif_analysis/${cell}_${hist_i}_bot"
	mkdir "model_results/motif_analysis/${cell}_${hist_i}_top"
	findMotifsGenome.pl "model_results/predictions/${cell}_${hist_i}_bot.bedGraph" hg19 "model_results/motif_analysis/${cell}_${hist_i}_bot/" -size 128
	findMotifsGenome.pl "model_results/predictions/${cell}_${hist_i}_top.bedGraph" hg19 "model_results/motif_analysis/${cell}_${hist_i}_top/" -size 128
}
export -f bed_motifs

parallel --tmpdir ~/.tmp --jobs 6 bed_motifs {} "$cell" ::: "${hists[@]}"
