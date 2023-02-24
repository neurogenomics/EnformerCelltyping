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
* precompute_dna_embeddings.py - script to precompute the DNA embeddings genome-wide (chro 1-22)
  for hg19. This will save time if predicitng genome-wide histone mark signals across mutliple 
  cell types of interest.
* predict_genome.py - script to predict the genome-wide histone mark signals for a cell type of 
  interest. See [using_enformer_celltyping](https://github.com/neurogenomics/EnformerCelltyping/blob/master/using_enformer_celltyping.ipynb)
  for more information on requirements of this.
* predict_genome_precomp.py - script to predict the genome-wide histone mark signals for a cell 
  type of interest with precomputed DNA embeddings to speed up the run. See 
[using_enformer_celltyping](https://github.com/neurogenomics/EnformerCelltyping/blob/master/using_enformer_celltyping.ipynb)
  for more information on requirements of this.
* process_qtl_as_sumstats.py - script to process the BLUEPRINT hQTL datasets in a sumstats format
  to be used with Enformer Celltyping predictions and SLDP. See 
  [Kundu et al](https://www.biorxiv.org/content/10.1101/2020.01.15.907436v1.full) for more details 
  about the datasets.
* munge.R - R script to standardise and quality control the BLUEPRINT hQTL datasets. This step was
  done before running the process_qtl_as_sumstats.py script.
* mke_sldp_train_data_arr.py - Precomputes the DNA embedding and chromatin accessibility data for all
  ~860k unique SNPs from the BLUEPRINT hQTL datsets. This is set up to be run in parallel using the 
  `./metadata/create_sldp_train_dat.txt` file. See [reproducing_hQTL_SNP_effect_prediction](https://github.com/neurogenomics/EnformerCelltyping/blob/master/reproducing_hQTL_SNP_effect_prediction.ipynb) for more details.
* pred_snp_effect_checkpoint_arr.py - Predict the effect of genetic variants for a chosen cell type and 
hQTL dataset, making use of the precomputed DNA embeddings and saving the predicted effects so they can
be used in other hQTL predictions for the same cell type.
* pred_snp_effect_all_qtl_arr.py - Predict the effect of genetic variants for a chosen cell type and
the unique genetic variants from all hQTL datasets, making use of the precomputed DNA embeddings and 
saving the predicted effects. This can be used in combination with red_snp_effect_checkpoint_arr.py. 
