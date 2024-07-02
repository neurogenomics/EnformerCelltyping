## custom functions ------------------------------------
#custom EWCE plot to add colour to cell type text based on tissue
ewce_plot_extra <- function(total_res,
                            celltype_map,
                            mtc_method = "bonferroni",
                            q_threshold = 0.05,
                            col_sig = TRUE,
                            ctd = NULL,
                            annotLevel = 1, 
                            heights = c(.3, 1), 
                            widths = c(1,0.05),
                            tiss_leg_mark_size=5,
                            make_dendro = FALSE,
                            verbose = TRUE) {
  
  requireNamespace("ggplot2")
  requireNamespace("patchwork")    
  
  EWCE:::check_mtc_method(mtc_method = mtc_method)
  multiList <- TRUE
  if (is.null(total_res$list)) multiList <- FALSE
  #### If using dendrogram ####
  if(isTRUE(make_dendro)){
    #### Check if ctd is provided ####
    if(is.null(ctd)){
      messager(
        "Warning: Can only add the dendrogram when ctd is provided.",
        "Setting make_dendro=FALSE.",
        v=verbose)
      make_dendro <- FALSE
    } else {
      # Find the relevant level of the CTD annotation 
      if (length(ctd[[annotLevel]]$plotting) > 0) {
        annotLevel <-
          which(unlist(lapply(ctd,
                              FUN = cells_in_ctd,
                              cells = as.character(
                                total_res$CellType
                              )
          )) == 1)
        err_msg2 <- paste0(
          "All of the cells within total_res should come",
          " from a single annotation layer of the CTD"
        )
        if (length(annotLevel) == 0) {
          stop(err_msg2)
        }
        cell_ordr <- ctd[[annotLevel]]$plotting$cell_ordering
      }else{
        #generate dendrogram - gives ordering
        ctdIN <- EWCE:::prep_dendro(ctdIN = ctd[[annotLevel]], 
                             expand = c(0, 1.3))  
        cell_ordr <- ctdIN$plotting$cell_ordering
      }
      #### Set order of cells ####
      total_res$CellType <-
        factor(x = total_res$CellType,
               levels = cell_ordr, 
               ordered = TRUE
        )
    } 
  }
  #### Multiple testing correction across all rows ####
  if (!"q" %in% colnames(total_res)) {
    total_res$q <- stats::p.adjust(total_res$p,
                                   method = mtc_method
    )
  }
  #### Mark significant rows with asterixes ####
  ast_q <- rep("", dim(total_res)[1])
  ast_q[total_res$q < q_threshold] <- "*"
  total_res$ast_q <- ast_q
  #add on tissue info
  total_res[celltype_map,Tissue:=i.Tissue,on="CellType"]
  #only add tissue colour if significant result
  col_tiss_uniq <- unique(total_res[q<q_threshold]$Tissue)
  total_res[,col_tiss:="Non-Signif\nTissue"]
  total_res[Tissue %in% col_tiss_uniq,col_tiss:=Tissue]
  #get cols for these
  #gg_color_hue <- function(n) {
  #  hues = seq(15, 375, length = n + 1)
  #  hcl(h = hues, l = 65, c = 100)[1:n]
  #}
  #pal <- gg_color_hue(length(unique(total_res$col_tiss)))
  print(paste0(length(col_tiss_uniq)," Tissues have a significant association"))
  if (col_sig){
    if(length(col_tiss_uniq)<=12){
      pal <- RColorBrewer::brewer.pal(12, "Paired")
    }else{
      pal <- c(RColorBrewer::brewer.pal(12, "Paired"),"#F5CDB4","grey")
      #extended_palette = ["#9A8822",,"#F8AFA8",
      #                    "#FDDDA0","#74A089","#85D4E3",
      #                    #added extra to make 7
      #                    '#78A2CC']
    }  
    #remove yellow, too bright
    pal <- c(pal[pal!="#FFFF99"],"#FFD700")
    uni_tiss_grps <- unique(total_res$col_tiss)
    pal_dt <- 
      data.table(
        col_tiss=sort(uni_tiss_grps[uni_tiss_grps != "Non-Signif\nTissue"]),
        col=pal[1:length(col_tiss_uniq)])
    pal_dt <- data.table::rbindlist(list(pal_dt,
                                         data.table(col_tiss=c("Non-Signif\nTissue"),
                                                    col=c("black"))))
    #add on tissue colour info
    total_res[pal_dt,col:=i.col,on="col_tiss"] 
    #### Plot ####
    total_res$sd_from_mean[total_res$sd_from_mean < 0] <- 0
    #create factor so tissue colours in order
    col_tiss_ordr <- unique(total_res$col_tiss)
    col_tiss_ordr <-c(sort(col_tiss_ordr[col_tiss_ordr!="Non-Signif\nTissue"]),
                      "Non-Signif\nTissue")
    total_res$col_tiss <-
      factor(x = total_res$col_tiss,
             levels = col_tiss_ordr, 
             ordered = TRUE
      )
    data.table::setorderv(total_res,c("list","CellType"))
    
    total_res_uniq <- unique(total_res, by = "CellType")
  }else{ #colour all tissue
    #colour everything by tissue
    total_res[,col_tiss:=Tissue]
    pal <- c(RColorBrewer::brewer.pal(12, "Paired"),"#F5CDB4","grey","brown")
    #remove yellow, too bright
    pal <- c(pal[pal!="#FFFF99"],"#FFD700")
    pal_dt <- data.table(col_tiss=unique(celltype_map$Tissue),col=pal)
    #add on tissue colour info
    total_res[pal_dt,col:=i.col,on="col_tiss"]
    #### Plot ####
    total_res$sd_from_mean[total_res$sd_from_mean < 0] <- 0
    #create factor so tissue colours in order
    col_tiss_ordr <- unique(total_res$col_tiss)
    col_tiss_ordr <-sort(col_tiss_ordr)
    total_res$col_tiss <-
      factor(x = total_res$col_tiss,
             levels = col_tiss_ordr, 
             ordered = TRUE
      )
    data.table::setorderv(total_res,c("list","CellType"))
    
    total_res_uniq <- unique(total_res, by = "CellType")
  }  
  graph_theme <- ggplot2::theme_bw(base_size = 11, 
                                   base_family = "Helvetica") +
    ggplot2::theme(
      text = ggplot2::element_text(size = 11),
      axis.title.y = ggplot2::element_text(vjust = 0.6,size=11),
      strip.background = ggplot2::element_rect(fill = "white"),
      strip.text = ggplot2::element_text(color = "black"),
      #add col to cell type text 
      axis.text.x = ggplot2::element_text(colour = total_res_uniq$col)
    )
  #get col legend to add in later
  plt <- ggplot2::ggplot(total_res,
                         ggplot2::aes(x=CellType,y=p,col=col_tiss))+
    ggplot2::geom_point(size=tiss_leg_mark_size)+
    ggplot2::scale_colour_manual(values = pal_dt$col)+
    graph_theme+
    ggplot2::theme(
      legend.background = ggplot2::element_rect(color = "transparent",
                                                fill="transparent"),
      legend.text=ggplot2::element_text(size = 9,
                                        margin = ggplot2::margin(#r = -2,b=-2,
                                                                 l=-5, 
                                                                 unit = "pt")),
      legend.key = ggplot2::element_rect(colour = "transparent", 
                                         fill = "transparent"),
      axis.text.x = ggplot2::element_text(angle = 90, hjust = 1,vjust=0.2),
      legend.title=ggplot2::element_text(size=10)
      )+ ggplot2::labs(color=' Tissue') 
  #get legend
  leg <- cowplot::get_legend(plt)
  
  upperLim <- max(abs(total_res$sd_from_mean), na.rm = TRUE)
  total_res$y_ast <- total_res$sd_from_mean * 1.05
  total_res$abs_sd <- abs(total_res$sd_from_mean)
  
  if ("Direction" %in% colnames(total_res)) {
    the_plot <- ggplot2::ggplot(total_res) +
      ggplot2::geom_bar(
        ggplot2::aes_string(x = "CellType", y = "abs_sd",
                            fill = "Direction"
        ),
        position = "dodge", stat = "identity"
      ) +
      graph_theme
  } else {
    the_plot <- ggplot2::ggplot(total_res) +
      ggplot2::geom_bar(
        ggplot2::aes_string(x = "CellType", y = "abs_sd", 
                            fill = "abs_sd"),
        stat = "identity"
      ) +
      ggplot2::scale_fill_gradient(low = "blue", high = "red") +
      graph_theme +
      ggplot2::theme(legend.position = "none")
  }
  
  # Setup the main plot
  the_plot <- the_plot +
    ggplot2::theme(
      plot.margin = ggplot2::unit(c(.5, 0, 0, 0), "mm"),
      axis.text.x = ggplot2::element_text(angle = 90, hjust = 1,vjust=0.2)
    ) +
    ggplot2::theme(panel.border = ggplot2::element_rect(
      colour = "black",
      fill = NA, linewidth = 1
    )) +
    ggplot2::xlab("Cell type") +
    ggplot2::theme(strip.text.y = ggplot2::element_text(angle = 0)) +
    ggplot2::ylab("Std.Devs. from the mean") 
  
  the_plot <- the_plot +
    ggplot2::scale_y_continuous(breaks = c(0, ceiling(upperLim * 0.66)),
                                expand = c(0, 1.1)) +
    ggplot2::geom_text(
      ggplot2::aes_string(label = "ast_q", x = "CellType", y = "y_ast"),
      size = 10
    )
  if (isTRUE(multiList)) {
    the_plot <- the_plot +
      ggplot2::facet_grid("list ~ .", 
                          scales = "free", 
                          space = "free_x")
  }
  #### Prepare output list ####
  output <- list()
  output$plain <- the_plot 
  if (isTRUE(make_dendro)) {
    #ctdIN wion't exist if plotting found earlier
    if(length(ctd[[annotLevel]]$plotting) > 0){
      ctdIN <- prep_dendro(ctdIN = ctd[[annotLevel]], 
                           expand = c(0, .66))  
    }  
    #update plot ordering by dendrogram
    dendro <- ctdIN$plotting$ggdendro_horizontal+ggplot2::scale_y_continuous(
      limits = c(0,max(ctdIN$plotting$ggdendro_horizontal$data$y)),
      expand = c(0,0))
    output$withDendro <- (
      ((dendro/the_plot)+
         patchwork::plot_layout(heights = heights)) |
        leg)+patchwork::plot_layout(heights = heights,widths = widths)
  }
  
  return(output)
}
## ------------------------------------

library(data.table)
library(patchwork)

sig_motifs <- fread("~/all_cells_sig_gbl_motifs.csv")
back_motifs <- fread("~/homer_background_tfs.csv")

#get TF for sig motifs
sig_motifs[, c("TF.1", "TF.2","TF.3") := tstrsplit(`Motif Name`, "/", 
                                                   fixed=TRUE)]
sig_motifs[, c("TF", "TF2") := tstrsplit(TF.1, "(", fixed=TRUE)]

#REMOVE MOTIFS IN ALL CELL TYPES TESTED
unique_cell_motif <- unique(sig_motifs[,c("Motif Name","cell")])
num_cts <- 7#6
rmv_motifs <- unique_cell_motif[, .N,by="Motif Name"][N>=num_cts,]$`Motif Name`
#now remove
sig_motifs <- sig_motifs[!`Motif Name` %in% rmv_motifs,]

#get cell types to test
CT <- unique(sig_motifs$cell)
#get ctd -
#CTD file with data from adult mouse whole-body atlas. 
#Uses the Smart-seq2 (FACS) subset of the data and covers a greater number of 
#cell-types than the Droplet version.
#Reference: doi:10.1038/s41586-018-0590-4
#TM_f <- MAGMA.Celltyping::get_ctd("ctd_TabulaMuris_facs")
#Descartes
#CTD file with scRNA-seq data from human embryo across multiple organ systems.
#Reference: doi:10.1126/science.aba7721
# https://descartes.brotmanbaty.org/bbi/human-gene-expression-during-development/
Des <- MAGMA.Celltyping::get_ctd("ctd_DescartesHuman")

#rename cell types
#taken from https://github.com/neurogenomics/rare_disease_celltyping/blob/master/data/DescartesHuman_celltype_mapping.csv
#rename descartes
descartes_mappings <- 
  read.csv("~/Downloads/DescartesHuman_celltype_mapping.csv") 
descartes_mappings$level1_nice = gsub("_"," ",descartes_mappings$level1) 
descartes_mappings$level1_nice = gsub("\\."," - ",
                                      descartes_mappings$level1_nice) 
descartes_mappings_dt <- setDT(descartes_mappings)
#Rename ctd
des_ctd_lvl_cell <- data.table(level2 = colnames(Des$level_2$mean_exp))
setkey(des_ctd_lvl_cell,level2)
setkey(descartes_mappings_dt,level1)
des_ctd_lvl_cell[descartes_mappings_dt,CellType:=i.level1_nice,]
des_ctd_lvl_cell[descartes_mappings_dt,Tissue:=i.tissue,]
#rename
colnames(Des$level_2$mean_exp) <- colnames(Des$level_2$specificity) <-
  colnames(Des$level_2$specificity_quantiles) <- des_ctd_lvl_cell$CellType

#get ctd
#CTD file derived from adult human cortex scRNA-seq data collected by the Allen 
#Institute for Brain Science (AIBS) Note that this CTD used an early release of 
#the AIBS data that only included samples from human 
#Medial Temporal Gyrus (MTG).
#Reference: doi:10.1038/s41586-019-1506-7
#AIB <- MAGMA.Celltyping::get_ctd("ctd_AIBS")
#allAIB <- MAGMA.Celltyping::get_ctd("AllenBrainInstituteHuman_smartseqv4")


res_cells <- vector(mode="list",length=length(CT))
names(res_cells) <- CT
res_cells_brain <- res_cells
for(cell_i in CT){
  set.seed(101)
  print(cell_i)
  hits_i <- sig_motifs[cell==cell_i]$TF
  backs <- back_motifs$TF
  #split where multiple genes mentioned - for both hits and bg
  hits_i <- unlist(strsplit(hits_i, ":",fixed = TRUE))
  backs <- unlist(strsplit(backs, ":",fixed = TRUE))
  hits_i <- unlist(strsplit(hits_i, "+",fixed = TRUE))
  backs <- unlist(strsplit(backs, "+",fixed = TRUE))
  #some manual changes
  #"NF-E2"-"NFE2"
  hits_i[hits_i=='NF-E2']='NFE2'
  backs[backs=='NF-E2']='NFE2'
  #"PU.1"-"SPI1"
  hits_i[hits_i=='PU.1']='SPI1'
  backs[backs=='PU.1']='SPI1'
  #"AP-1" - "JUN"
  hits_i[hits_i=='AP-1']='JUN'
  backs[backs=='AP-1']='JUN'
  #"AP-2gamma" - "TFAP2C"
  hits_i[hits_i=='AP-2gamma']='TFAP2C'
  backs[backs=='AP-2gamma']='TFAP2C'
  #"AP-2alpha" - "TFAP2A"
  hits_i[hits_i=='AP-2alpha']='TFAP2A'
  backs[backs=='AP-2alpha']='TFAP2A'
  #"Tlx?" - "NR2E1"
  hits_i[hits_i=='Tlx?']='NR2E1'
  backs[backs=='Tlx?']='NR2E1'
  #another split
  hits_i <- unlist(strsplit(hits_i, "-"))
  backs <- unlist(strsplit(backs, "-"))
  #final manual change
  #"Nkx6.1" - "Nkx6-1"
  hits_i[hits_i=='Nkx6.1']='Nkx6-1'
  backs[backs=='Nkx6.1']='Nkx6-1'
  
  #unique
  backs <- unique(backs)
  hits_i <- unique(hits_i) 
  print(paste0("Hits: ",length(hits_i)," Background: ",length(backs)))
  
  #homer uses multiple species
  for(spec in list('mouse','fly','rat','zebrafish','Saccharomyces cerevisiae')){
    spec_hits <- orthogene::map_genes(
      genes = hits_i,
      species = spec,
      drop_na = TRUE,
      verbose = FALSE)$name
    spec_backs <- orthogene::map_genes(
      genes = backs,
      species = spec,
      drop_na = TRUE,
      verbose = FALSE)$name
    #get orthologs
    if(!is.null(spec_backs)){
      spec_hits_hsp <-orthogene::map_orthologs(
        genes = spec_hits,
        input_species = spec,
        output_species = 'human',
        method = 'gprofiler',
        verbose = FALSE
      )$ortholog_gene
      #add to genes - take unique
      hits_i <- unique(append(hits_i,spec_hits_hsp))
    }
    if(!is.null(spec_backs)){
      spec_backs_hsp <-orthogene::map_orthologs(
        genes = spec_backs,
        input_species = spec,
        output_species = 'human',
        method = 'gprofiler',
        verbose = FALSE
      )$ortholog_gene
      #add to genes - take unique
      backs <- unique(append(backs,spec_backs_hsp))
    }
  }
  
  
  #check how many lost
  checkedLists <- EWCE::check_ewce_genelist_inputs(
    sct_data = Des,#TM_f,
    hits = hits_i,
    bg = backs,
    genelistSpecies = 'human',
    sctSpecies = "human",#'mouse',
    sctSpecies_origin = "human",#'mouse',
    geneSizeControl = FALSE,
    output_species = 'human',
    min_genes = 3,
    verbose = TRUE
  )
  
  print(paste0("Filt Hits: ",length(checkedLists$hits),
               " Filt Background: ",length(checkedLists$bg),
               " Filt Bg+H: ",length(checkedLists$bg)+length(checkedLists$hits))
        )

  #run analysis
  res <- EWCE::bootstrap_enrichment_test(sct_data = Des,#TM_f,
                                         sctSpecies = "human",#"mouse",
                                         genelistSpecies = "human",
                                         hits = hits_i,
                                         bg = backs,
                                         reps = 10000,
                                         annotLevel = 2,
                                         geneSizeControl = TRUE
                                         )
  #save res
  res_cells[[cell_i]] <- res$results
}

res_cells_dt <- rbindlist(res_cells,idcol = 'list')


#map naming to 'nicer' ontology names - from MultiEWCE
res_cells_dt[,ctd:="DescartesHuman"]

#func from multiEWCE with slight change
map_celltype_ME <- function(results,
                         input_col="CellType",
                         map = KGExplorer::get_data_package(
                           package = "MultiEWCE",
                           name="celltype_maps"),
                         rm_prefixes=c("Adult","Fetus","HESC"),
                         by=c("ctd","author_celltype")
){
  author_celltype <- NULL;
  new_cols <- c("cell_type_ontology_term_id","cell_type")
  if(all(new_cols %in% names(results))) {
    return(results)
  }
  print("Mapping cell types to cell ontology terms.")
  results[,author_celltype:=gsub(paste(paste0("^",rm_prefixes,"_"),
                                       collapse = "|"),"",
                                 get(input_col),ignore.case = TRUE)]
  if(!all(by %in% names(results))) {
    stopper("All 'by' columns must be in 'results'.")
  }
  #remove _ from map
  map[,author_celltype:=gsub("_", " ", author_celltype)]
  results_cl <- data.table::merge.data.table(results,
                                             map,
                                             by=by,
                                             all.x = TRUE)
  if(sum(is.na(results_cl$cell_type_ontology_term_id))>0){
    stop("Missing 'cell_type_ontology_term_id' for",
            sum(is.na(results_cl$cell_type_ontology_term_id))," rows.")
  }
  ## Rename cols to make more concise and conform to hpo_id/hpo_name format
  data.table::setnames(results_cl,
                       c("cell_type_ontology_term_id","cell_type"),
                       c("cl_id","cl_name"))
  return(results_cl)
}

res_cells_dt_ontol <- map_celltype_ME(res_cells_dt)
#grouped some cells so rename to make unique
res_cells_dt_ontol[, counter := seq_len(.N), by = .(list,cl_name)]
res_cells_dt_ontol[, counter := as.character(counter) ]
res_cells_dt_ontol[counter!="1", counter := paste0(" ",counter)]
res_cells_dt_ontol[counter=="1", counter := ""]
res_cells_dt_ontol[,cl_name_uniq:=paste0(cl_name,counter)]
#use cl_name for CellType
res_cells_dt_ontol[,CellType:=cl_name_uniq]
#add in new cell type naming everywhere
map <- KGExplorer::get_data_package(package = "MultiEWCE",
                                    name="celltype_maps")
map_des <- map[ctd=='DescartesHuman',]
setkey(map_des,"author_celltype")
setkey(des_ctd_lvl_cell,"level2")
des_ctd_lvl_cell[map_des,CellType:=i.cell_type]
#now add numbers
des_ctd_lvl_cell[, counter := seq_len(.N), by = .(CellType)]
des_ctd_lvl_cell[, counter := as.character(counter) ]
des_ctd_lvl_cell[counter!="1", counter := paste0(" ",counter)]
des_ctd_lvl_cell[counter=="1", counter := ""]
des_ctd_lvl_cell[,Celltype_uniq:=paste0(CellType,counter)]
des_ctd_lvl_cell[,CellType:=Celltype_uniq]
#also need to update Des
#rename
colnames(Des$level_2$mean_exp) <- colnames(Des$level_2$specificity) <-
  colnames(Des$level_2$specificity_quantiles) <- des_ctd_lvl_cell$Celltype_uniq
#now plot
plot_body <- ewce_plot_extra(total_res = res_cells_dt_ontol,
                             celltype_map = des_ctd_lvl_cell,
                             mtc_method = "BH",
                             col_sig = FALSE,
                             make_dendro = TRUE,
                             annotLevel = 2,
                             ctd = Des,
                             heights = c(0.05,1), 
                             widths = c(1,0.04),
                             tiss_leg_mark_size=2.4
                             )
ggplot2::ggsave(plot_body$withDendro,
                filename="~/Downloads/tf_cell_specificity.pdf",
                dpi = 1200,width = 16,height = 10, units ="in")