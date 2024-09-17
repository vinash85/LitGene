# Load required libraries
library(Seurat)
library(dplyr)
library(ggplot2)
library(readr)
library(WebGestaltR)
# File path
emb_filename <- '/home/macaulay/macaulay/test/df_solubility_cl2.csv'  

# Create necessary directories
base_dir <- paste0("enrichment_analysis_cl/")
enrichment_output_dir <- paste0(base_dir, "/GeneLLM_all")
dir.create(base_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(enrichment_output_dir, recursive = TRUE, showWarnings = FALSE)
data <- read.csv(emb_filename, row.names=1)
data <- na.omit(data)
expression_data <- data %>% select( -Label) #%>% as.matrix()
expression_data <- t(expression_data)

metadata <- data %>% select(Label)
# metadata <- t(metadata)
seurat_obj <- CreateSeuratObject(counts = expression_data)

VariableFeatures(seurat_obj) <- rownames(seurat_obj)
new_seu <- SetAssayData(
    object = seurat_obj,
    layer = "scale.data",
    new.data = as.matrix(expression_data)
)
seurat_obj <- AddMetaData(seurat_obj, metadata = metadata)

seurat_obj <- RunPCA(new_seu, verbose = FALSE)
seurat_obj <- RunUMAP(seurat_obj, reduction = "pca", dims = 1:50)
seurat_obj <- RunTSNE(seurat_obj, reduction = "pca", dims = 1:50)
seurat_obj <- FindNeighbors(seurat_obj, reduction = "pca", dims = 1:50)
seurat_obj <- FindClusters(seurat_obj, resolution = 0.4)
seurat_obj$Label = data[,1]

seurat_obj_file_path <- paste0(base_dir, "/seurat_obj.rds")
saveRDS(seurat_obj, file = seurat_obj_file_path)
print(paste("Seurat object saved at:", seurat_obj_file_path))


# # Load the Seurat object
# seurat_obj <- readRDS(file = seurat_obj_file_path)



# Save the initial TSNE plot colored by cluster indices
initial_tsne_file_path <- paste0(base_dir, "/GeneLLM_cluster_tsne.pdf")
pdf(initial_tsne_file_path, width = 13, height = 9)
print(DimPlot(seurat_obj, reduction = "tsne", label = TRUE, label.size = 5, group.by = 'Label'))
dev.off()

# Save the initial TSNE plot colored by cluster indices
initial_tsne_file_path <- paste0(base_dir, "/GeneLLM_cluster_umap.pdf")
pdf(initial_tsne_file_path, width = 13, height = 9)
print(DimPlot(seurat_obj, reduction = "umap", label = TRUE, label.size = 5, group.by = 'Label'))
dev.off()

print('1')
# Save the initial TSNE plot colored by cluster indices
initial_tsne_file_path <- paste0(base_dir, "/GeneLLM_cluster_pca.pdf")
pdf(initial_tsne_file_path, width = 13, height = 9)
print(DimPlot(seurat_obj, reduction = "pca", label = TRUE, label.size = 5, group.by = 'Label'))
dev.off()

initial_tsne_file_path <- paste0(base_dir, "/GeneLLM_cluster_test_numbered_tsne.pdf")
pdf(initial_tsne_file_path, width = 13, height = 9)
print(DimPlot(seurat_obj, reduction = "tsne", label = TRUE, label.size = 5))
dev.off()


getLowestPValueAndFDRDescription <- function(enrichResult, originalNames) {
    lowestDescriptions <- originalNames  # Start with original cluster names
    
    for (cluster in names(enrichResult)) {
        result <- enrichResult[[cluster]]
        if (nrow(result) > 0) {
            orderedResult <- result[order(result$pValue, result$FDR), ]
            lowestDescriptions[cluster] <- orderedResult$description[1]
        } else {
            # Retain the original numeric index as the name
            lowestDescriptions[cluster] <- cluster
        }
    }
    
    return(lowestDescriptions)
}

# Perform Enrichment Analysis
all_list <- list()                                                                                   
enrichResult <- list()
originalNames <- levels(seurat_obj)
for(i in levels(seurat_obj)) {
    all_list[[i]] <- WhichCells(seurat_obj, idents = i)
    cluster_file_path <- paste0(base_dir, "/GeneLLM_all_cluster", i, ".txt")
    write.table(all_list[[i]], cluster_file_path, quote = F, sep = "\t", row.names = F, col.names = F)
    enrichResult[[i]] <- WebGestaltR(minNum = 3, enrichMethod="ORA", organism="hsapiens", projectName=paste0(base_dir, "Reactome_cluster", i), enrichDatabase="pathway_Reactome", interestGeneFile=paste0(base_dir, "GeneLLM_all_cluster", i, ".txt"), referenceGene = colnames(seurat_obj), referenceGeneType = "genesymbol", reportNum=50, outputDirectory = paste0(base_dir, "GeneLLM_all"), interestGeneType="genesymbol")
}

new_names <- getLowestPValueAndFDRDescription(enrichResult, originalNames)


seurat_obj <- RenameIdents(object = seurat_obj, new_names)
# Save the named TSNE plot
named_tsne_file_path <- paste0(base_dir, "/GeneLLM_clusterTSNE_all_named.pdf")
pdf(named_tsne_file_path, width = 13, height = 9)
print(DimPlot(seurat_obj, reduction = "tsne", label = TRUE, label.size = 5) + NoLegend(), group.by = task_name)
dev.off()

print(new_names)

