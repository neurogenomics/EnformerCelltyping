#EpiMap data is averaged at 25 bp intervals
#We want to average over larger regions for predictions

library(rtracklayer)
library(BSgenome.Hsapiens.UCSC.hg19)
library(GenomicRanges)

#read in inputted values for plotting
args <- commandArgs(trailingOnly = TRUE)
bigwig_file <- args[1]
#run in subdir so need to ref parent dir
#bigwig_file <- paste0("../",bigwig_file)
pred_bin_avg <- as.integer(args[2]) #128


#Don't run for DNA, liftover, SLDP or model_ref folders
direct_parent_folder <- basename(sub("/[^/]+$", "", sub(".[^.]+$", "", bigwig_file)))
#Also don't run if the file has already been averaged to the desired level
avg_lvl <- (sub(".*_", "", sub(".[^.]+$", "",basename(bigwig_file))))

if(!(direct_parent_folder %in% c("dna","liftover","model_ref","sldp"))&&
       (is.na(avg_lvl) || (avg_lvl!=as.character(pred_bin_avg) && avg_lvl!="hg38"))){
#load assay
true_epi_mark <- 
    rtracklayer::import(bigwig_file)
#average to Xbp level
gr_windows <- 
    GenomicRanges::tileGenome(GenomicRanges::seqinfo(BSgenome.Hsapiens.UCSC.hg19::Hsapiens),
                                tilewidth=pred_bin_avg, 
                                cut.last.tile.in.chrom=TRUE)
data_cov <- GenomicRanges::coverage(true_epi_mark, weight="score")
rm(true_epi_mark)
seqlevels(gr_windows, pruning.mode="coarse") <- names(data_cov)
epi_mark_avg <- GenomicRanges::binnedAverage(gr_windows, data_cov, 
                                                  "score")
#keep chr 1:22, X,Y
epi_mark_avg <- 
    epi_mark_avg[seqnames(epi_mark_avg) %in% 
                          paste0("chr",c(seq_len(22),"X","Y"))]
#update seq lengths
seqlevels(epi_mark_avg, pruning.mode="coarse") <- 
    paste0("chr",c(seq_len(22),"X","Y"))
#trim parts not found in hg19
seqinfo(epi_mark_avg) <- Seqinfo(genome="hg19")
epi_mark_avg <- trim(epi_mark_avg)
rtracklayer::export.bw(object=epi_mark_avg, 
                       con = paste0(sub(".[^.]+$", "", bigwig_file),"_",
                                        pred_bin_avg,".bigWig"))
}
