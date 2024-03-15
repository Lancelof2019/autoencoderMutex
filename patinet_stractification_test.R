##### Patient stratification
options(stringsAsFactors = F)
library(NbClust)
library(ggplot2)
library(grid)
library(ComplexHeatmap)
library(circlize)
library(tidyverse)
library(maftools)
#/home/lance/Downloads/testWorkSpace/test12/data/tcga_data_processed/samples.RData


#/home/lance/Downloads/testWorkSpace/test12/data/python_related/result/community/communityScores_compare19.csv


#/home/lance/Downloads/testWorkSpace/test12/data/tcga_data_processed/exp_intgr.RData
#/home/lance/Downloads/testWorkSpace/test12/data/tcga_data_processed/mty_intgr.RData
#/home/lance/Downloads/testWorkSpace/test12/data/tcga_data_processed/snv_intgr.RData
#/home/lance/Downloads/testWorkSpace/test12/data/tcga_data_processed/cnv_intgr.RData.RData
#/home/lance/Downloads/testWorkSpace/test12/data/tcga_data_processed/clinical_info.RData
#/home/lance/Downloads/testWorkSpace/test12/data/tcga_data/therapy.RData
#/home/lance/Downloads/testWorkSpace/test12/data/tcga_data/radiation.RData
#/home/lance/Downloads/testWorkSpace/test12/data/spinglass/melanet_cmt.RData
#/home/lance/Downloads/testWorkSpace/test12/data/community_scores.RData
#/home/lance/Downloads/testWorkSpace/test12/figure/best_number_of_clusters.pdf
#/home/lance/Downloads/testWorkSpace/test12/data/therapy_radiation_df.RData

#/home/lance/Downloads/testWorkSpace/test12/figure/heatmap_cmtScores.pdf

#/home/lance/Downloads/testWorkSpace/test12/data/sample_partition.RData

ht_opt$message = FALSE
samples <- readRDS('/home/lance/Downloads/testWorkSpace/test12/data/tcga_data_processed/samples.RData')
# cmtScores <- read.csv("./data/python_related/result/communityScores.csv", check.names = F, header = F)
cmtScores <- read.csv(('/home/lance/Downloads/testWorkSpace/test12/data/python_related/result/community/correct_combined45.csv'), check.names = F, header = F)
exp_intgr <- readRDS('/home/lance/Downloads/testWorkSpace/test12/data/tcga_data_processed/exp_intgr.RData')
mty_intgr <- readRDS('/home/lance/Downloads/testWorkSpace/test12/data/tcga_data_processed/mty_intgr.RData')
snv_intgr <- readRDS('/home/lance/Downloads/testWorkSpace/test12/data/tcga_data_processed/snv_intgr.RData')
cnv_intgr <- readRDS('/home/lance/Downloads/testWorkSpace/test12/data/tcga_data_processed/cnv_intgr.RData')
clinicalInfo <- readRDS('/home/lance/Downloads/testWorkSpace/test12/data/tcga_data_processed/clinical_info.RData')
therapy <- readRDS('/home/lance/Downloads/testWorkSpace/test12/data/tcga_data/therapy.RData')
radiation <- readRDS('/home/lance/Downloads/testWorkSpace/test12/data/tcga_data/radiation.RData')
melanet_cmt <- readRDS('/home/lance/Downloads/testWorkSpace/test12/data/spinglass/melanet_cmt.RData')

row.names(cmtScores) <- samples
colnames(cmtScores) <- paste0("cmt", 1:ncol(cmtScores))
saveRDS(cmtScores, ('/home/lance/Downloads/testWorkSpace/test12/data/community_scores45_1.RData'))

### Determine the best number of clusters
nc <- NbClust(scale(cmtScores), distance = "euclidean", min.nc = 2, max.nc = 10, method = "complete", index = "all")

pdf("/home/lance/Downloads/testWorkSpace/test12/figure/best_number_of_clusters45_1.pdf", width = 7, height = 7)
ggplot(data.frame(cluster = factor(nc$Best.nc[1,])), aes(x = cluster)) + geom_bar(stat = "count", fill = "#C1BFBF") + labs(x = "Number of clusters", y = "Number of criteria", title = "Number of clusters chosen by 26 criteria") + theme(text = element_text(size = 18), plot.title = element_text(hjust = 0.5, face = "bold"), panel.grid = element_blank(), panel.background = element_rect(fill = 'transparent', color = 'black')) + 
  scale_y_continuous(breaks = seq(0,14,2), limits = c(0,14))
dev.off()

### Community scores of clustered patients
tumorType <- clinicalInfo[,"shortLetterCode"]
tumorStage <- clinicalInfo[,"tumor_stage"]
tumorStage[-grep("^stage", tumorStage)] <- NA
tumorStage <- gsub("^stage ", "", tumorStage)
tumorStage <- gsub("[a-c]$", "", tumorStage)

tumorStage <- clinicalInfo[,"tumor_stage"]
tumorStage[-grep("^stage", tumorStage)] <- NA
tumorStage <- gsub("^stage ", "", tumorStage)
tumorStage <- gsub("[a-c]$", "", tumorStage)

therapy <- therapy[3:nrow(therapy),]
ifTherapy <- substr(samples, 1, 12) %in% therapy$bcr_patient_barcode
ifTherapy <- ifelse(ifTherapy, "Yes", "No")

radiation <- radiation[3:nrow(radiation),]
ifRadiation <- substr(samples, 1, 12) %in% radiation$bcr_patient_barcode
ifRadiation <- ifelse(ifRadiation, "Yes", "No")

therapy_type <- sapply(substr(samples, 1, 12), function(x){paste(sort(unique(therapy$pharmaceutical_therapy_type[therapy$bcr_patient_barcode == x])),collapse = ";")})

tr_df <- data.frame(sample = samples, theray = ifTherapy, radiation = ifRadiation, therapy_type = therapy_type)
saveRDS(tr_df, "/home/lance/Downloads/testWorkSpace/test12/data/therapy_radiation_df45_1.RData")

tumorType_col_fun <- c("TM" = "#CC79A7", "TP" = "#0072B2")
tumorStage_col_fun <- c("0" = "#FAFCC2", "i" = "#FFEFA0", "ii" = "#FFD57E", "iii" = "#FCA652", "iv" = "#AC4B1C")
ifTherapy_col_fun <- c("Yes" = "red", "No" = "gray")
ifRadiation_col_fun <- c("Yes" = "red", "No" = "gray")

topAnno <- HeatmapAnnotation(Therapy = ifTherapy, Radiation = ifRadiation, `Tumor type` = tumorType, `Tumor stage` = tumorStage, col = list(Therapy = ifTherapy_col_fun, Radiation = ifRadiation_col_fun, `Tumor type` = tumorType_col_fun, `Tumor stage` = tumorStage_col_fun), border = T, show_annotation_name = T)
ht = Heatmap(t(scale(cmtScores)), 
             name = "Community score", 
             show_column_names = F,
             # top_annotation = topAnno,
             clustering_distance_columns = "euclidean",
             clustering_method_columns = "complete",
             column_split = 3,
             column_title = "%s",
)

pdf("/home/lance/Downloads/testWorkSpace/test12/figure/heatmap_cmtScores45_1.pdf", width = 13, height = 13)
draw(ht, merge_legends = TRUE)
dev.off()


ht = draw(ht)
rowOrder <- row_order(ht)
colOrder <- column_order(ht)
samplePartition <- data.frame(cluster = rep(1:length(colOrder), lengths(colOrder)), sampleID = unlist(colOrder))
samplePartition <- samplePartition[order(samplePartition$sampleID),]
saveRDS(samplePartition, "/home/lance/Downloads/testWorkSpace/test12/data/sample_partition45_1.RData")

tcgaData <- list(exp = t(exp_intgr),
                 mty = t(mty_intgr),
                 snv = t(snv_intgr),
                 cnv = t(cnv_intgr))
tumorType1 <- tumorType[unlist(colOrder)]
tumorStage1 <- tumorStage[unlist(colOrder)]
ifTherapy1 <- ifTherapy[unlist(colOrder)]
ifRadiation1 <- ifRadiation[unlist(colOrder)]
topAnno1 <- HeatmapAnnotation(Therapy = ifTherapy1, Radiation = ifRadiation1, `Tumor type` = tumorType1, `Tumor stage` = tumorStage1, col = list(Therapy = ifTherapy_col_fun, Radiation = ifRadiation_col_fun, `Tumor type` = tumorType_col_fun, `Tumor stage` = tumorStage_col_fun), border = T, show_annotation_name = T)
for(i in 1:length(tcgaData)){
  dmat <- tcgaData[[i]]
  dmat <- dmat[,unlist(colOrder)]
  if(i == 1){
    col_fun <- colorRamp2(c(min(dmat), max(dmat)),c("#FFFFFF","#FFC7C7"))
  }
  else if(i == 2){
    col_fun <- colorRamp2(c(min(dmat), max(dmat)),c("#D9EBEA","#68B0AB"))
  }
  else if(i == 3){
    col_fun <- colorRamp2(c(min(dmat), mean(dmat), max(dmat)),c("#DBDBDB","#c2a5cf","#7b3294"))         
  }
  else{
    col_fun <- colorRamp2(c(min(dmat), mean(dmat), max(dmat)),c("blue","white","red"))
  }
  rht <- Heatmap(dmat,
                 name = names(tcgaData)[i],
                 col = col_fun,
                 top_annotation = topAnno1,
                 show_column_names = F,
                 show_row_names = F,
                 cluster_columns = F,
                 cluster_rows = T,
                 column_split = rep(1:length(colOrder), lengths(colOrder)))
  pdf(paste("/home/lance/Downloads/testWorkSpace/test12/figure/heatmap45_1_", names(tcgaData)[i], ".pdf", sep = ""));
  draw(rht, merge_legends = TRUE);
  dev.off()
}

plot_genes_exp <- c() 
plot_genes_mty <- c()
ht_list <- list()
col_fun <- list()

