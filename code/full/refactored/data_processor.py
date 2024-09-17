###########################################################
# Project: GeneLLM
# File: data_processor.py
# License: MIT
# Code Authors: Jararaweh A., Macualay O.S, Arredondo D., & 
#               Virupakshappa K.
###########################################################
import pandas as pd
import re
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizerFast
from transformers import XLNetTokenizer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


def process_sent(sent):
    """ @ala and @macaulay add description"""
    pattern1 = r'(?:"(.*?)"|\'(.*?)\')'
    pattern2 = r"\[provided by .*?\]"
    pattern3 = r"\(PubMed:\d+(?:\s+\d+)*\)"
    pattern4 = r"\(\s+[\w\s]+\s+[\w]+\s+\)"
    pattern5 = r"\s*\(Microbial infection\)"
    pattern6 = r"\[(Isoform [^\]]+)\]:\s*"
    pattern7 = r"\(By similarity\)"

    matches = re.findall(pattern1, sent)
    captured_content = [match[0] if match[0] else match[1] for match in matches]
    text = " ".join(captured_content)
    text = re.sub(pattern2, "", text)
    text = re.sub(pattern3, "####", text)
    text = re.sub(pattern4, "", text)
    text = re.sub(pattern5, "", text)
    text = re.sub(pattern6, r"\1 ", text)
    text = re.sub(pattern7, "", text)
    text = re.sub(r'\s+', ' ', text)
    
    return text

def process_Go_data(path):
    """ @ala and @macaulay add description"""
    
    df = pd.read_csv(path)
    
    df["Gene Name"] = df.index
    df = df.melt(id_vars=["Gene Name"], var_name="GO Term", value_name="Gene Names")


    df = df.dropna(subset=["Gene Names"])

    df = df.reset_index(drop=True)

    gene_ontology_dict = {}

    for _, row in df.iterrows():
        gene_name = row["Gene Names"]
        go_term = row["GO Term"]

        if gene_name not in gene_ontology_dict:
            gene_ontology_dict[gene_name] = [go_term]
        else:
            gene_ontology_dict[gene_name].append(go_term)

            
    
    for gene_name, go_terms in gene_ontology_dict.items():
        if len(go_terms) == 1:
            gene_ontology_dict[gene_name] = go_terms

            
    
    knownGenes = []
    for i in range (14):
        file_path = f'/data/macaulay/GeneLLM2/data/knownGenes/GeneLLM_all_cluster{i}.txt'
        with open(file_path, 'r') as file:
            for line in file:
                knownGenes.append(line.strip()) 
                

    genes = pd.read_csv("data/genes.csv")
    genes["Summary"] = genes["Summary"].apply(lambda sent: process_sent(sent))
    genes = genes.drop_duplicates(subset='Summary')
    genes = genes[genes['Gene name'].isin(knownGenes)]

    genes["StrLabel"] = genes["Gene name"].apply(lambda name: gene_ontology_dict.get(name,None))
    
    genes_go = genes.dropna(subset=["StrLabel"]).reset_index(drop=True)  

    
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(genes_go['StrLabel'])
    
    genes_go["Label"] = labels.tolist() 
    
    return genes_go, len(mlb.classes_)

def loading_data(input_data_path = 'data/subcellular_location.csv', task_type = "classification"):
    """ @ala and @macaulay add description"""
    if task_type == 'multilabel':
        genes_datas, n_labels = process_Go_data(input_data_path)

        return genes_datas, n_labels

    else:
        knownGenes = []
        for i in range (14):
            file_path = f'/home/tailab/GeneLLM/data/knownGenes/GeneLLM_all_cluster{i}.txt'
            with open(file_path, 'r') as file:
                for line in file:
                    knownGenes.append(line.strip()) 


        input_data = pd.read_csv(input_data_path)
        gene_name = input_data.columns[0]
        # global task_name
        column_name = input_data.columns[1]
        labels_dict = input_data.set_index(gene_name)[column_name].to_dict()


        genes = pd.read_csv("/mnt/data/GeneLLM/data/clean_genes.csv")
        # genes = genes.dropna(subset=['Summary'])
        # genes = genes.drop_duplicates(subset='Summary')
        # genes = genes[genes['Summary'].apply(lambda summary: len(summary.split())) >= 15]
        # genes["Summary"] = genes["Summary"].apply(lambda sent: process_sent(sent))
        genes = genes.drop_duplicates(subset='Summary')
        genes = genes[genes['Gene name'].isin(knownGenes)]

        if task_type == "classification":

            genes[column_name] = genes["Gene name"].apply(lambda name: labels_dict.get(name,None))

            genes_loc = genes.dropna(subset=[column_name]).reset_index(drop=True)

            labels, uniques = pd.factorize(genes_loc[column_name])
            label_dict = dict(zip(uniques, range(len(uniques))))
            n_labels = len(uniques)
            

            genes_loc["Label"] = genes_loc[column_name].map(label_dict)

            #print(genes_loc.Label.value_counts())
        
            return genes_loc, n_labels

        elif task_type == "regression":
            genes["Label"] = genes["Gene name"].apply(lambda name: labels_dict.get(name,None))
            genes_datas = genes.dropna(subset=["Label"]).reset_index(drop=True)
            n_labels = 1

            return genes_datas, n_labels
         
def process_data(genes, max_length, batch_size, val_genes = None , test_genes = None, 
                 task_type = "classification", gene2vec_flag = True,
                 model_name = "microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract", task_name = 'Subcellular_location', model_type= 'Pubmed_large'):
    
    """ @ala and @macaulay add description"""
    if "xlnet" in model_name:
        tokenizer = XLNetTokenizer.from_pretrained(model_name)

    else:    
        tokenizer = BertTokenizerFast.from_pretrained(model_name)

    
    
    sentences, labels = genes["Summary"].tolist() , genes["Label"].tolist()
    g_index, g_name = genes.index.tolist() , genes["Gene name"].tolist()
    
            

    tokens = tokenizer.batch_encode_plus(sentences, max_length = max_length,
                                         padding="max_length", truncation=True)

    data = {'input_ids': tokens["input_ids"],
            'token_type_ids': tokens["token_type_ids"],
            'attention_mask': tokens["attention_mask"],
            "labels": labels,
            "g_index": g_index,
            "g_name": g_name
           }
    
    tokens_df = pd.DataFrame(data)
    print(f"Shape of tokens_df before gene2vec:{tokens_df.shape}")
    
    #############################################
    if gene2vec_flag:
        # print("Adding Gene2Vec data ...")
        
        # unirep_df = pd.read_csv('/mnt/data/macaulay/datas/training_gene_embeddings.csv')
        # Gene2Vec = unirep_df.set_index('Gene').T.to_dict('list')
                    
        # tokens_df = tokens_df[tokens_df['g_name'].isin(set(Gene2Vec.keys()) & set(tokens_df["g_name"]))]
        
        # tokens_df["gene2vec"] = tokens_df["g_name"].apply(lambda name: 
        #                                                   Gene2Vec[name])# if name in Gene2Vec.keys() else None )

        print("Adding Gene2Vec data ...")
        
        Gene2Vec = dict()

        file_path = f'/data/macaulay/GeneLLM2/data/gene2vec_embeddings.txt'
        with open(file_path, 'r') as file:
            for line in file:

                name, embed = line.strip().split("	")
                embed = [float(value) for value in embed.split()] 

                Gene2Vec[name.strip()] = embed
                    
            
        tokens_df = tokens_df[tokens_df['g_name'].isin(set(Gene2Vec.keys()) & set(tokens_df["g_name"]))]
        
        tokens_df["gene2vec"] = tokens_df["g_name"].apply(lambda name: 
                                                          Gene2Vec[name])# if name in Gene2Vec.keys() else None )
    
    #############################################
    print(f"Shape of tokens_df after gene2vec:{tokens_df.shape}")
    
    #val_genes, test_genes,
    

    if val_genes is not None:
        val_tokens = tokens_df[tokens_df['g_name'].isin(val_genes)]
        test_tokens = tokens_df[tokens_df['g_name'].isin(test_genes)]
        train_tokens = tokens_df[~tokens_df['g_name'].isin(val_genes + test_genes)]


            
    else:

        os.makedirs(f'{data_path}/{task_name}/{model_type}/folds', exist_ok=True)

        # kf = KFold(n_splits=5, shuffle=True, random_state=42) if task_type == "regression" else StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # for fold, (train_index, val_index) in enumerate(kf.split(tokens_df, None if task_type == "regression" else tokens_df['labels'])):
        

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (train_index, val_index) in enumerate(kf.split(tokens_df)):
            train_tokens = tokens_df.iloc[train_index]
            val_tokens = tokens_df.iloc[val_index]
            train_tokens, test_tokens = train_test_split(train_tokens,test_size=0.20, random_state=42)
            
            # Save test gene names to a CSV file
            test_gene_names = test_tokens['g_name']
            test_gene_names.to_csv(f'{data_path}/{task_name}/{model_type}/folds/test_gene_names_fold_{fold}.csv', index=False)

            # Save val gene names to a CSV file
            val_gene_names = val_tokens['g_name']
            val_gene_names.to_csv(f'{data_path}/{task_name}/{model_type}/folds/val_gene_names_fold_{fold}.csv', index=False)

    

    train_tokens = train_tokens.reset_index(drop=True)
    val_tokens = val_tokens.reset_index(drop=True)
    test_tokens = test_tokens.reset_index(drop=True)
    
    if gene2vec_flag:
    
        train_dataset = TensorDataset(torch.tensor(train_tokens["input_ids"].tolist()),
                                      torch.tensor(train_tokens["attention_mask"].tolist()),
                                      torch.tensor(train_tokens["gene2vec"]),
                                      torch.tensor(train_tokens["labels"]),
                                      torch.tensor(train_tokens["g_index"]))
        
        val_dataset = TensorDataset(torch.tensor(val_tokens["input_ids"].tolist()) ,
                            torch.tensor(val_tokens["attention_mask"].tolist()),
                            torch.tensor(val_tokens["gene2vec"]),
                            torch.tensor(val_tokens["labels"]),
                            torch.tensor(val_tokens["g_index"]))
        
        test_dataset = TensorDataset(torch.tensor(test_tokens["input_ids"].tolist()),
                             torch.tensor(test_tokens["attention_mask"].tolist()),
                             torch.tensor(test_tokens["gene2vec"]),
                             torch.tensor(test_tokens["labels"]),
                             torch.tensor(test_tokens["g_index"]))
    else:
        train_dataset = TensorDataset(torch.tensor(train_tokens["input_ids"].tolist()),
                                      torch.tensor(train_tokens["attention_mask"].tolist()),
                                      torch.tensor(train_tokens["labels"]),
                                      torch.tensor(train_tokens["g_index"]))
        
        val_dataset = TensorDataset(torch.tensor(val_tokens["input_ids"].tolist()) ,
                            torch.tensor(val_tokens["attention_mask"].tolist()),
                            torch.tensor(val_tokens["labels"]),
                            torch.tensor(val_tokens["g_index"]))
        
        test_dataset = TensorDataset(torch.tensor(test_tokens["input_ids"].tolist()),
                             torch.tensor(test_tokens["attention_mask"].tolist()),
                             torch.tensor(test_tokens["labels"]),
                             torch.tensor(test_tokens["g_index"]))
        
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    
    return train_loader, val_loader, test_loader

