from shapleyAnalysis import *
import sys
import os
import pandas as pd
from torch.nn import DataParallel
from GeneLLM_class import *

## IMPORT MODEL AND TOKENIZER ##
from transformers import BertTokenizerFast

## DATA ##
input_data_path = "clean_genes.csv"
gene_loaded_data = pd.read_csv(input_data_path)
sentences = gene_loaded_data["Summary"].tolist()
geneNames = gene_loaded_data["Gene name"].tolist()
gene_to_idx = {gene:idx for idx, gene in enumerate(geneNames)}
tokenizer_max_length= 512
device = 'cuda:1'

model = pickleLoad('solubilityGO/best_model.pkl')
tokenizer = BertTokenizerFast.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
save_dir = '/home/dandreas/GeneLLM2/solubilityGO/'
dataset_name = 'clean_genes_solubilityGO'
shap_values = getSHAPValues(sentences, model, tokenizer, tokenizer_max_length, device, dataset_name, save_dir)