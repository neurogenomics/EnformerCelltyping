#!/bin/bash

echo "Average bigwig tracks starting"
now="$(date)"
printf "Current date and time %s\n" "$now"

# run in parallel for speed
avg_tracks () {
    local bigwig_file=$1
    #Average all tracks to larger bin
    bin_size=128
    echo "$bigwig_file"
    #R script to average track at specified size
    Rscript ./bin/avg_bigwig_tracks.R $bigwig_file $bin_size
}
export -f avg_tracks

parallel --jobs 16 avg_tracks ::: `find ./data/h3k27ac ./data/h3k4me3/ ./data/h3k36me3/ ./data/h3k9me3/ ./data/h3k4me1/ ./data/h3k27me3/ ./data/atac/ -name '*.bigWig' -not -name *_128.bigWig`
