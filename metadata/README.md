# Enformer Celltyping Metadata

This folder contains metadata needed for various functionality of
Enformer Celltyping. 

## Data loader datasets

Files needed to load data to predict or train Enformer Celltyping:

* dna.csv - metadata giving links to DNA bigwig files. These will be used
  for model training and when using the model for predictions.
* tss_hg19.csv.gz - The transcriptional start site of known genes in hg19
* PanglaoDB_markers_27_Mar_2020.tsv.gz - Known marker genes from PangloaDB
  Note - These are used to derive the global chromatin accessibility signature
  of a cell type. See load_chrom_access_prom() in EnformerCelltyping/utils.py 
  for more details.

## Model training datasets

Files used in the training of Enformer Celltyping:

* cell_info.csv - Information on the [EpiMap](http://compbio.mit.edu/epimap/)
  cell types and assays used for training.
* h3k27ac/h3k27me3/h3k36me3/h3k4me1/h3k4me3/h3k9me3.csv - metadata giving 
  links to histone mark bigwig files that are to be predicted for model
  training.
* atac.csv - metadata giving links to ATAC-seq bigwig files that are used to
  derive the chromatin accessibility data used by the model.
* model_ref.csv - other files needed by the model when training, contains a link
  to ENCODE blacklist regions which are excluded from the training regions.

## Downstream uses of Enformer Celltyping

Files needed for doing downstream analysis with Enformer Celltyping:

* sldp_config.json - Parameters used for running [SLDP](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6202062/)
