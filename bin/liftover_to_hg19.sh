#!/bin/bash

# Converts certain files from hg38 file to hg19
mkdir ./data/liftover

wget --timestamping 'http://hgdownload.soe.ucsc.edu/goldenPath/hg38/liftOver/hg38ToHg19.over.chain.gz' -O ./data/liftover/hg38ToHg19.over.chain.gz

# list files to be lifted over, any file ending in _hg38

echo "Liftover bigwig of files to hg19 starting"
now="$(date)"
printf "Current date and time %s\n" "$now"

# run in parallel for speed
liftover () {
    local run=$1
    echo "$run"
    CrossMap.py bigwig ./data/liftover/hg38ToHg19.over.chain.gz $run "${run%_hg38.bigWig}"
    #remove intermediate files
    rm "${run%_hg38.bigWig}".bgr
    rm "${run%_hg38.bigWig}".sorted.bgr
    #rename lifted over file .bigWig rather than bw
    mv -- "${run%_hg38.bigWig}".bw "${run%_hg38.bigWig}.bigWig"
}
export -f liftover

parallel --jobs 2 liftover ::: `find ./data/ -name '*_hg38.bigWig'`

echo "liftover of bigwig files complete"
now="$(date)"
printf "Current date and time %s\n" "$now"

