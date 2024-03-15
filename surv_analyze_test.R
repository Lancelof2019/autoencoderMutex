##### Survival analysis
options(stringsAsFactors = F)
library(TCGAbiolinks)
clinicalInfo <- readRDS("/home/lance/Downloads/testWorkSpace/test12/data/tcga_data_processed/clinical_info.RData")
samplePartition <- readRDS("/home/lance/Downloads/testWorkSpace/test12/data/sample_partition45_1.RData")
survivalInfo <- clinicalInfo[,c("shortLetterCode", "tumor_stage","vital_status","days_to_death","days_to_last_follow_up")]
survivalInfo$hc <- samplePartition$cluster # ???˲??ξ???????
survivalInfo$tumor_stage[which(survivalInfo$tumor_stage == "stage 0")] <- 0
survivalInfo$tumor_stage[which(survivalInfo$tumor_stage == "stage i")] <- 1
survivalInfo$tumor_stage[which(survivalInfo$tumor_stage == "stage ia")] <- 1
survivalInfo$tumor_stage[which(survivalInfo$tumor_stage == "stage ib")] <- 1
survivalInfo$tumor_stage[which(survivalInfo$tumor_stage == "stage ic")] <- 1
survivalInfo$tumor_stage[which(survivalInfo$tumor_stage == "stage ii")] <- 2
survivalInfo$tumor_stage[which(survivalInfo$tumor_stage == "stage iia")] <- 2
survivalInfo$tumor_stage[which(survivalInfo$tumor_stage == "stage iib")] <- 2
survivalInfo$tumor_stage[which(survivalInfo$tumor_stage == "stage iic")] <- 2
survivalInfo$tumor_stage[which(survivalInfo$tumor_stage == "stage iii")] <- 3
survivalInfo$tumor_stage[which(survivalInfo$tumor_stage == "stage iiia")] <- 3
survivalInfo$tumor_stage[which(survivalInfo$tumor_stage == "stage iiib")] <- 3
survivalInfo$tumor_stage[which(survivalInfo$tumor_stage == "stage iiic")] <- 3
survivalInfo$tumor_stage[which(survivalInfo$tumor_stage == "stage iv")] <- 4
survivalInfo$tumor_stage[which(survivalInfo$tumor_stage == "not reported")] <- "NA"
survivalInfo$tumor_stage[which(survivalInfo$tumor_stage == "i/ii nos")] <- "NA"
survivalInfo$tumor_stage[which(is.na(survivalInfo$tumor_stage))] <- "NA"
#TCGAanalyze_survival(survivalInfo, clusterCol = "hc", color = c("#33A02C","#1F78B4","#E31A1C"), filename = "/home/lance/Downloads/testWorkSpace/test12/figure/surv_analysis/survival_analysis42_1.pdf", conf.int = F, width = 7, height = 7)
TCGAanalyze_survival(survivalInfo, clusterCol = "hc", color = c("#33A02C", "#1F78B4", "#E31A1C", "#6A3D9A", "#FF7F00"),filename = "/home/lance/Downloads/testWorkSpace/test12/figure/surv_analysis/survival_analysis45_1.pdf", conf.int = F, width = 7, height = 7)
#color = c("#33A02C", "#1F78B4", "#E31A1C", "#6A3D9A", "#FF7F00"),
#result <- TCGAanalyze_survival(survivalInfo, clusterCol = "hc", color = c("#33A02C","#1F78B4","#E31A1C"), filename=NULL,pvalue = TRUE,conf.int = F, width = 7, height = 7)
result <- TCGAanalyze_survival(survivalInfo, clusterCol = "hc", color = c("#33A02C", "#1F78B4", "#E31A1C", "#6A3D9A", "#FF7F00"), filename=NULL,pvalue = TRUE,conf.int = F, width = 7, height = 7)
attributes(result)
#print(result$pvalue)
# 从结果中提取Log-rank测试的p值
#p_value <- result$logrank$pval
print(p_value)
print(result)
TCGAanalyze_survival(survivalInfo, clusterCol = "shortLetterCode", filename = "/home/lance/Downloads/testWorkSpace/test12/figure/surv_analysis/survival_analysis_tumorType45_1.pdf", conf.int = F, width = 7, height = 7)
TCGAanalyze_survival(survivalInfo, clusterCol = "tumor_stage", filename = "/home/lance/Downloads/testWorkSpace/test12/figure/surv_analysis/survival_analysis_tumorStage45_1.pdf", conf.int = F, width = 7, height = 7)
survivalInfo$vital_status_numeric <- ifelse(survivalInfo$vital_status == "Dead", 1, 0)
library(pdftools)
text <- pdf_text("/home/lance/Downloads/testWorkSpace/test12/figure/surv_analysis/survival_analysis45_1.pdf")
#matches <- grep("p < 0\\.[0-9]+", text, value = TRUE)
#print(matches)
#matches1<- grep("\\b[p]\\s*<\\s*0\\.0001\\b",text, value = TRUE)
#print(matches1)
#p<0\\.[0-9]+
matches_locations <- gregexpr("p < 0\\.[0-9]+", text)
matches <- regmatches(text, matches_locations)
#print(matches)
numeric_value <- sub("p\\s*<\\s*", "", matches)
#print(numeric_value)
numeric_value <- as.numeric(numeric_value)
print(numeric_value)
