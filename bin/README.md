# bin

Supplementary scripts for running preprocessing steps or other analysis
for Enformer Celltyping.

* download_data_parallel.py - To download the data to replicate the training
  in the paper.
* download_Enformer_Celltyping_dependencies.py - Download just the files necessary
  to use the trained, Enformer Celltyping model.
* get_sldp_dependencies.py - List dependencies for running SLDP. Used when 
  downloading files.
* precomute_train_data.py - To Pre-load the X (DNA and local and global chromatin
  accessibility data) and Y (histone marks) from the bigwig files at the training
  locations and saving them as npz files to be used for training.
* liftover_to_hg19.sh - script to liftover any downloaded hg38 files used to train 
  the model to hg19.
* avg_bigwig_tracks.sh - script to change the resolution of all bigwig files from 25
  base-pairs (the default) to 128 base-pairs.
* avg_bigwig_tracks.R - R script with a function to change the resolution of a bigwig 
  file from one base-pair resolution to another.
* calculate_avg_track.sh - script to calculate the average signal of the training cells
  for each histone mark and chromatin accessibilty (ATAC-seq)
* calculate_avg_track.py - calculate the average signal for the inputted epigenetic mark
  based on training cells
* run_sldp.py - script to run [SLDP](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6202062/)
  on genetic variants relating to a given phenotype to look for directional effect. Need a
  GWAS sumstats file and a model's predictions of effect of the SNP (like Enformer Celltyping).
