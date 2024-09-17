# Load required libraries
library(Seurat)
library(dplyr)
library(ggplot2)
library(readr)
library(WebGestaltR)
# File path
emb_filename <- '/data/macaulay/GeneLLM2/data/obesity_disease_gene_embeddings.csv'  

# Create necessary directories
base_dir <- paste0("enrichment_analysis_all_disease_res/")
enrichment_output_dir <- paste0(base_dir, "/GeneLLM_all")
dir.create(base_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(enrichment_output_dir, recursive = TRUE, showWarnings = FALSE)
data <- read.csv(emb_filename, row.names=1)
data <- na.omit(data)
expression_data <- data %>% select( -Disease) #%>% as.matrix()
expression_data <- t(expression_data)

metadata <- data %>% select(Disease)
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
seurat_obj <- FindClusters(seurat_obj, resolution = 3.0)
seurat_obj$Disease = data[,1]

seurat_obj_file_path <- paste0(base_dir, "seurat_obj.rds")
saveRDS(seurat_obj, file = seurat_obj_file_path)
print(paste("Seurat object saved at:", seurat_obj_file_path))


# # Load the Seurat object
# seurat_obj_file_path <- paste0(base_dir, "seurat_obj.rds")

# seurat_obj <- readRDS(file = seurat_obj_file_path)



# Save the initial TSNE plot colored by cluster indices
initial_tsne_file_path <- paste0(base_dir, "/GeneLLM_cluster_tsne.pdf")
pdf(initial_tsne_file_path, width = 13, height = 9)
print(DimPlot(seurat_obj, reduction = "tsne", label = TRUE, label.size = 5, group.by = 'Disease'))
dev.off()

# # Save the initial TSNE plot colored by cluster indices
# initial_tsne_file_path <- paste0(base_dir, "/GeneLLM_cluster_umap.pdf")
# pdf(initial_tsne_file_path, width = 13, height = 9)
# print(DimPlot(seurat_obj, reduction = "umap", label = TRUE, label.size = 5, group.by = 'Label'))
# dev.off()

# print('1')
# # Save the initial TSNE plot colored by cluster indices
# initial_tsne_file_path <- paste0(base_dir, "/GeneLLM_cluster_pca.pdf")
# pdf(initial_tsne_file_path, width = 13, height = 9)
# print(DimPlot(seurat_obj, reduction = "pca", label = TRUE, label.size = 5, group.by = 'Label'))
# dev.off()

initial_tsne_file_path <- paste0(base_dir, "/GeneLLM_cluster_test_numbered_tsne.pdf")
pdf(initial_tsne_file_path, width = 13, height = 9)
print(DimPlot(seurat_obj, reduction = "tsne", label = TRUE, label.size = 5, group.by = 'Disease'))
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





































# Load required libraries
library(Seurat)
library(dplyr)
library(ggplot2)
library(readr)
library(viridis) # For color scale in ggplot

# File paths
emb_filename <- '/data/macaulay/GeneLLM2/data/obesity_disease_gene_embeddings.csv'
disease_emb_filename <- '/data/macaulay/GeneLLM2/contrastive/disease_embeddings.csv'

# Create necessary directories
base_dir <- "enrichment_analysis_all_disease_res2/"
enrichment_output_dir <- paste0(base_dir, "/GeneLLM_all")
dir.create(base_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(enrichment_output_dir, recursive = TRUE, showWarnings = FALSE)

# Load the gene embeddings
data <- read.csv(emb_filename, row.names = 1)
data <- na.omit(data)
expression_data <- data %>% select(-Disease)

# Load the disease embeddings
disease_data <- read.csv(disease_emb_filename, row.names = 1)

# Prepare the disease metadata
disease_metadata <- data.frame(Disease = rownames(disease_data))
rownames(disease_metadata) <- rownames(disease_data)

# Combine gene and disease embeddings
combined_embeddings <- rbind(expression_data, disease_data)
expression_data <- t(combined_embeddings)

# Combine metadata
gene_metadata <- data %>% select(Disease)
combined_metadata <- rbind(gene_metadata, disease_metadata)

# Create a Seurat object with the combined data
seurat_obj <- CreateSeuratObject(counts = expression_data)
VariableFeatures(seurat_obj) <- rownames(seurat_obj)
seurat_obj <- SetAssayData(
    object = seurat_obj,
    layer = "scale.data",
    new.data = as.matrix(expression_data)
)
seurat_obj <- AddMetaData(seurat_obj, metadata = combined_metadata)

# Run dimensionality reduction
seurat_obj <- RunPCA(seurat_obj, verbose = FALSE)
seurat_obj <- RunUMAP(seurat_obj, reduction = "pca", dims = 1:50)
seurat_obj <- RunTSNE(seurat_obj, reduction = "pca", dims = 1:50)



# Plot with ggplot2, adding labels for disease points
tsne_plot <- ggplot() +
  geom_point(data = gene_data, aes(x = tSNE_1, y = tSNE_2, color = Disease), alpha = 0.5) +
  geom_point(data = disease_data_plot, aes(x = tSNE_1, y = tSNE_2), shape = 17, size = 3, color = 'black') +
  geom_text(data = disease_data_plot, aes(x = tSNE_1, y = tSNE_2, label = Disease), vjust = 1.5, color = 'red', size = 3.5) +
  scale_color_viridis_d() +
  theme_minimal() +
  labs(title = "t-SNE Plot of Gene and Disease Embeddings", x = "t-SNE 1", y = "t-SNE 2")

# Display the plot
print(tsne_plot)

# Optionally, save the plot to a file
ggsave(paste0(base_dir, "/Gene_and_Disease_TSNE_labeled.pdf"), tsne_plot, width = 10, height = 8)
