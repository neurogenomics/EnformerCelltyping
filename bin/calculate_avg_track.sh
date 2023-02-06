#create avg track for all epi marks
doit() {
    #for each epigenetic mark
    track=${1%.*}
    echo "$track"
    #first sample each chromosome
    python ./bin/calculate_avg_track.py -m "$track"
}
export -f doit
#run all at once
marks=(h3k27ac h3k4me1 h3k4me3 h3k9me3 h3k27me3 h3k36me3 atac)
parallel --jobs 7 doit ::: "${marks[@]}"
