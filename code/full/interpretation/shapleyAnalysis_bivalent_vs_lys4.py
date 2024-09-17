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
from GeneLLM import * 
from transformers import BertTokenizerFast
state_dict_name = 'BivalentVsLys4_2'
state_dict_path = 'best_models/best_model_'+state_dict_name+'.pth'

## DATA ##
input_data_path = "clean_genes.csv"
gene_loaded_data = pd.read_csv(input_data_path)
sentences = gene_loaded_data["Summary"].tolist()
geneNames = gene_loaded_data["Gene name"].tolist()
gene_to_idx = {gene:idx for idx, gene in enumerate(geneNames)}

## HYPERPARAMETERS ## 
pool= "cls"
drop_rate= 0.1
model_name= 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
gene2vec_flag= False
gene2vec_hidden= 200
device="cuda:0"
tokenizer_max_length= 512
n_labels= 2
task_type='classification'

## INITIALIZE MODEL AND TOKENIZER ##

model = FineTunedBERT(pool= pool, 
                    task_type = task_type, 
                    n_labels = n_labels,
                    drop_rate =  drop_rate, 
                    model_name = model_name,
                    gene2vec_flag= gene2vec_flag,
                    gene2vec_hidden = gene2vec_hidden).to(device)

model.load_state_dict(torch.load(state_dict_path))

tokenizer = BertTokenizerFast.from_pretrained(model.model_name)


# # **Shapley Analysis**
# 
# ### Instructions
# Check the following parameters and run the next block.
# 
# 
# ```save_dir``` is the directory where the SHAP values will be saved.
# 
# 
# ```dataset_name``` will be the prefix of the shap values file saved by the analysis.

# In[ ]:


save_dir = '/home/dandreas/GeneLLM2/data/SHAPValues/'
dataset_name = 'clean_genes_'+state_dict_name
shap_values = getSHAPValues(sentences, model, tokenizer, tokenizer_max_length, device, dataset_name, save_dir)

