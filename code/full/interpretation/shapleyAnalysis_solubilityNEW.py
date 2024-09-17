#!/usr/bin/env python
# coding: utf-8

# # **Define Model**
# 
# ### Instructions
# 
# **1.** Put model ```.py``` file in this directory
# 
# 
# **2.** Change ```state_dict_path``` and ```input_data_path``` as needed
# 
# 
# **3.** Check model hyperparameters
# 
# 
# **4.** Run the next block

# In[ ]:


from shapleyAnalysis import *
import sys
import os

## IMPORT MODEL AND TOKENIZER ##
from solubilityModelClassNEW import *
from transformers import BertTokenizerFast
model_path = '/data/ajararweh/solubility/best_model.pkl'

## DATA ##
input_data_path = "clean_genes.csv"
gene_loaded_data = pd.read_csv(input_data_path)
sentences = gene_loaded_data["Summary"].tolist()
geneNames = gene_loaded_data["Gene name"].tolist()
gene_to_idx = {gene:idx for idx, gene in enumerate(geneNames)}


device="cuda:1"
tokenizer_max_length= 512
model = pickleLoad(model_path)
model = model.module
tokenizer = BertTokenizerFast.from_pretrained(model.model_name)

save_dir = '/home/dandreas/GeneLLM2/solubilityNEW/'
dataset_name = 'clean_genes_solubilityNEW'
shap_values = getSHAPValues(sentences, model, tokenizer, tokenizer_max_length, device, dataset_name, save_dir)
