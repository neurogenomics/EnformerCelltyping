if(!require(MungeSumstats)) 
  devtools::install_github("neurogenomics/MungeSumstats")
  BiocManager::install("BSgenome.Hsapiens.1000genomes.hs37d5")
  BiocManager::install("SNPlocs.Hsapiens.dbSNP155.GRCh37")
#ensure MSS v1.5.16 at least
print(packageVersion("MungeSumstats"))

#get your path
if(!require(here))
  devtools::install_github("r-lib/here")
print(here::here())

library(here)
library(data.table)
library(MungeSumstats)

#set number of threads for speed
n_threads = 20
setDTthreads(n_threads)
#get QTL files to be munged
qtls <- 
  list.files(path = here("projects","hQTL_blueprint_phase2"), 
             include.dirs = FALSE,pattern ="\\.txt.gz$")

#loop through each
for (qtl_i in qtls){
  #short handle
  nme <- substr(qtl_i,0,10)
  print(nme)
  #load QTL file
  dt <- 
    data.table::fread(here("projects","hQTL_blueprint_phase2",
                          qtl_i))
  #split out chr, pos, ref, alt
  dt[, c("chr", "pos_ref_alt") := tstrsplit(chr.pos_ref_alt, ":", fixed=TRUE)]
  dt[,chr.pos_ref_alt:=NULL]
  dt[, c("pos", "ref","alt") := tstrsplit(pos_ref_alt, "_", fixed=TRUE)]
  dt[,pos_ref_alt:=NULL]
  
  rtrn <- MungeSumstats::format_sumstats(
    dt,
    ref_genome = 'GRCH37',
    dbSNP = 155,
    nThread = n_threads,
    save_path =here("projects","hQTL_blueprint_phase2","Munged",
                    qtl_i),
    #calc Z-score
    compute_z = 'BETA',
    #non-biallelic and same SNP to multi pos allowed
    bi_allelic_filter = FALSE,
    check_dups = FALSE,
    #only want SNPS
    indel = FALSE,
    #### Record logs
    log_folder_ind = TRUE,
    log_mungesumstats_msgs = TRUE,
    log_folder =here("projects","hQTL_blueprint_phase2","Munged",nme,"logs")
  ) 
}

