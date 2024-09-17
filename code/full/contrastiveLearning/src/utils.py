import transformers
from transformers import XLNetTokenizer, XLNetModel
from transformers import AutoModel, BertTokenizerFast
from transformers import AdamW

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
import pandas as pd
import numpy as np
import ast


from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def process_interaction_data(genes, max_length=512, batch_size=100,
                             val_genes = None , test_genes = None, 
                             test_split_size=0.7, frac =1,
                             model_name= "bert-base-cased"):
    
    
    
    if frac < 1:
        genes = genes.sample(frac=frac) #.reset_index(drop=True)
        
    
    if "xlnet" in model_name:
        tokenizer = XLNetTokenizer.from_pretrained(model_name)

    else:    
        tokenizer = BertTokenizerFast.from_pretrained(model_name)

    

    anchor = genes["Anchor"].tolist()
    positive = genes["Positive"].tolist()
    negative = genes["Negative"].tolist() 
    g_index = genes.index.tolist()
#     g_name = genes["Gene name"].tolist()
    

    tokens_a = tokenizer.batch_encode_plus(anchor, max_length = max_length,
                                           padding="max_length", truncation=True)

    tokens_p = tokenizer.batch_encode_plus(positive, max_length = max_length,
                                           padding="max_length", truncation=True)

    tokens_n = tokenizer.batch_encode_plus(negative, max_length = max_length,
                                           padding="max_length", truncation=True)

    
    #'token_type_ids': tokens["token_type_ids"],
    data = {
        'input_ids_a': tokens_a["input_ids"],
        'attention_mask_a': tokens_a["attention_mask"],

        'input_ids_p': tokens_p["input_ids"],
        'attention_mask_p': tokens_p["attention_mask"],

        'input_ids_n': tokens_n["input_ids"],
        'attention_mask_n': tokens_n["attention_mask"],
        
        "g_index": g_index,
#         "g_name": g_name,
#         "labels": labels,
    }
    
    tokens_df = pd.DataFrame(data)

    print(tokens_df.shape)

    if val_genes is not None:
        val_tokens = tokens_df.loc[val_genes]
        test_tokens =  tokens_df.loc[test_genes]
        train_tokens = tokens_df.drop(val_genes+test_genes)
        
    else:

        train_tokens, test_tokens = train_test_split(tokens_df, test_size=test_split_size,
                                                     random_state=42)

        test_tokens, val_tokens = train_test_split(test_tokens,test_size=0.5,
                                                   random_state=42)


    train_tokens = train_tokens.reset_index(drop=True)
    val_tokens = val_tokens.reset_index(drop=True)
    test_tokens = test_tokens.reset_index(drop=True)

    print(train_tokens.shape, val_tokens.shape, test_tokens.shape)
    
    train_dataset = TensorDataset(
        torch.tensor(train_tokens["input_ids_a"].tolist()),
        torch.tensor(train_tokens["attention_mask_a"].tolist()),
        torch.tensor(train_tokens["input_ids_p"].tolist()),
        torch.tensor(train_tokens["attention_mask_p"].tolist()),
        torch.tensor(train_tokens["input_ids_n"].tolist()),
        torch.tensor(train_tokens["attention_mask_n"].tolist()),
#         torch.tensor(train_tokens["labels"]),
        torch.tensor(train_tokens["g_index"])
    )

    val_dataset = TensorDataset(
        torch.tensor(val_tokens["input_ids_a"].tolist()) ,
        torch.tensor(val_tokens["attention_mask_a"].tolist()),        
        torch.tensor(val_tokens["input_ids_p"].tolist()) ,
        torch.tensor(val_tokens["attention_mask_p"].tolist()),
        torch.tensor(val_tokens["input_ids_n"].tolist()) ,
        torch.tensor(val_tokens["attention_mask_n"].tolist()),
#         torch.tensor(val_tokens["labels"]),
        torch.tensor(val_tokens["g_index"])
    )

    test_dataset = TensorDataset(
        torch.tensor(test_tokens["input_ids_a"].tolist()),
        torch.tensor(test_tokens["attention_mask_a"].tolist()),
        torch.tensor(test_tokens["input_ids_p"].tolist()),
        torch.tensor(test_tokens["attention_mask_p"].tolist()),
        torch.tensor(test_tokens["input_ids_n"].tolist()),
        torch.tensor(test_tokens["attention_mask_n"].tolist()),
#         torch.tensor(test_tokens["labels"]),
        torch.tensor(test_tokens["g_index"])
    )

        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    
    return train_loader, val_loader, test_loader

def process_triplet_data(genes, max_length=512, batch_size=100,
                         val_genes = None , test_genes = None, 
                         test_split_size=0.7,
                         model_name= "bert-base-cased"):
    
    
    seed = np.random.randint(0,1000)
    if "xlnet" in model_name:
        tokenizer = XLNetTokenizer.from_pretrained(model_name)

    else:    
        tokenizer = BertTokenizerFast.from_pretrained(model_name)

    
    labels = genes["Label"].tolist()
    anchor = genes["Anchor"].tolist()
    positive = genes["Positive"].tolist()
    negative = genes["Negative"].tolist() 
    g_index = genes.index.tolist()
    g_name = genes["Gene name"].tolist()
    

    tokens_a = tokenizer.batch_encode_plus(anchor, max_length = max_length,
                                           padding="max_length", truncation=True)

    tokens_p = tokenizer.batch_encode_plus(positive, max_length = max_length,
                                           padding="max_length", truncation=True)

    tokens_n = tokenizer.batch_encode_plus(negative, max_length = max_length,
                                           padding="max_length", truncation=True)

    
    #'token_type_ids': tokens["token_type_ids"],
    data = {
        'input_ids_a': tokens_a["input_ids"],
        'attention_mask_a': tokens_a["attention_mask"],

        'input_ids_p': tokens_p["input_ids"],
        'attention_mask_p': tokens_p["attention_mask"],

        'input_ids_n': tokens_n["input_ids"],
        'attention_mask_n': tokens_n["attention_mask"],
        
        "g_index": g_index,
        "g_name": g_name,
        "labels": labels,
    }
    
    tokens_df = pd.DataFrame(data)


    if val_genes is not None:
        val_tokens = tokens_df.loc[val_genes]
        test_tokens =  tokens_df.loc[test_genes]
        train_tokens = tokens_df.drop(val_genes+test_genes)
        
    else:

        train_tokens, test_tokens = train_test_split(tokens_df, test_size=test_split_size,
                                                     random_state=seed)

        test_tokens, val_tokens = train_test_split(test_tokens,test_size=0.5,
                                                   random_state=seed)


    train_tokens = train_tokens.reset_index(drop=True)
    val_tokens = val_tokens.reset_index(drop=True)
    test_tokens = test_tokens.reset_index(drop=True)

    
    train_dataset = TensorDataset(
        torch.tensor(train_tokens["input_ids_a"].tolist()),
        torch.tensor(train_tokens["attention_mask_a"].tolist()),
        torch.tensor(train_tokens["input_ids_p"].tolist()),
        torch.tensor(train_tokens["attention_mask_p"].tolist()),
        torch.tensor(train_tokens["input_ids_n"].tolist()),
        torch.tensor(train_tokens["attention_mask_n"].tolist()),
        torch.tensor(train_tokens["labels"]),
        torch.tensor(train_tokens["g_index"]))

    val_dataset = TensorDataset(
        torch.tensor(val_tokens["input_ids_a"].tolist()) ,
        torch.tensor(val_tokens["attention_mask_a"].tolist()),        
        torch.tensor(val_tokens["input_ids_p"].tolist()) ,
        torch.tensor(val_tokens["attention_mask_p"].tolist()),
        torch.tensor(val_tokens["input_ids_n"].tolist()) ,
        torch.tensor(val_tokens["attention_mask_n"].tolist()),
        torch.tensor(val_tokens["labels"]),
        torch.tensor(val_tokens["g_index"]))

    test_dataset = TensorDataset(
        torch.tensor(test_tokens["input_ids_a"].tolist()),
        torch.tensor(test_tokens["attention_mask_a"].tolist()),
        torch.tensor(test_tokens["input_ids_p"].tolist()),
        torch.tensor(test_tokens["attention_mask_p"].tolist()),
        torch.tensor(test_tokens["input_ids_n"].tolist()),
        torch.tensor(test_tokens["attention_mask_n"].tolist()),
        torch.tensor(test_tokens["labels"]),
        torch.tensor(test_tokens["g_index"]))

        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    
    return train_loader, val_loader, test_loader

def plot_latent(latents, labels, epoch, class_map = None, validation_type="train"):
    
    tsne = TSNE(n_components=2)
    scaler = StandardScaler()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        latents_tsne = tsne.fit_transform(latents)
    

    if class_map is not None:
        
        for i, class_label in enumerate(np.unique(labels)):
            class_indices = labels == class_label
            cl = class_map[class_label]
            plt.scatter(latents_tsne[class_indices, 0],
                        latents_tsne[class_indices, 1],
                        s=30, alpha=0.5,
                        label=f'{cl}')
#         plt.legend()
        
    else:
        plt.scatter(latents_tsne[:, 0], latents_tsne[:, 1])

        
    plt.title(f'Epoch {epoch}')
    plt.savefig(f"saved-figures/{validation_type}/latent_{epoch}.png")
    plt.close()
        
def process_data(genes, max_length, batch_size, val_genes = None , test_genes = None, 
                 task_type = "classification", gene2vec_flag = False, gene2vec_file_path = 'data/gene2vec_embeddings.csv', 
                 test_split_size=0.15, val_split_size=0.15,
                 model_name= "bert-base-cased"):
    
    train_loader, val_loader, test_loader = None,None,None
    
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
    
    
    #############################################
    if gene2vec_flag:
        print(f"Shape of tokens_df before gene2vec:{tokens_df.shape}")
        
        # Load the gene2vec embeddings from the CSV file
        gene2vec_df = pd.read_csv(gene2vec_file_path)

        # Convert the 'gene2vec' string to a list of floats using ast.literal_eval
        gene2vec_df['gene2vec'] = gene2vec_df['gene2vec'].apply(ast.literal_eval)

        # Create a dictionary from the DataFrame
        Gene2Vec = dict(zip(gene2vec_df['Gene name'].str.strip(), gene2vec_df['gene2vec']))

        # Filter the tokens_df to only include genes that are in the Gene2Vec dictionary
        tokens_df = tokens_df[tokens_df['g_name'].isin(Gene2Vec.keys())]

        # Add the gene2vec embeddings to the tokens_df
        tokens_df['gene2vec'] = tokens_df['g_name'].apply(lambda name: Gene2Vec[name])

        print(f"Shape of tokens_df after gene2vec:{tokens_df.shape}")
    #############################################
    
    
    if test_genes is not None:
        tokens_df = tokens_df.set_index("g_name")
        
        if gene2vec_flag:
                test_genes = list(set(test_genes) & set(Gene2Vec.keys()))

        if val_split_size > 0:
            val_tokens = tokens_df.loc[val_genes].reset_index(drop=True)
            test_tokens =  tokens_df.loc[test_genes].reset_index(drop=True)
            train_tokens = tokens_df.drop(val_genes+test_genes).reset_index(drop=True)
        else:
            
            test_tokens =  tokens_df.loc[test_genes].reset_index(drop=True)
            train_tokens = tokens_df.drop(test_genes).reset_index(drop=True)
 
                   
    else:
        train_tokens, test_tokens = train_test_split(tokens_df,
                                                        test_size=test_split_size,
                                                        stratify = tokens_df.labels if task_type == "classification" else None,
                                                        random_state=1)
            
        if val_split_size > 0:
            train_size = 1 - (val_split_size+test_split_size)
            val_relative_size = val_split_size / (train_size + val_split_size)

            train_tokens, val_tokens = train_test_split(train_tokens,
                                                    test_size=val_relative_size,
                                                    stratify = train_tokens.labels if task_type == "classification" else None,
                                                    random_state=1)
            val_tokens = val_tokens.reset_index(drop=True)

        test_tokens = test_tokens.reset_index(drop=True)
        train_tokens = train_tokens.reset_index(drop=True)
            

            
            
            


        
    
    if gene2vec_flag:
    
        train_dataset = TensorDataset(torch.tensor(train_tokens["input_ids"].tolist()),
                                      torch.tensor(train_tokens["attention_mask"].tolist()),
                                      torch.tensor(train_tokens["gene2vec"]),
                                      torch.tensor(train_tokens["labels"]),
                                      torch.tensor(train_tokens["g_index"]))
        if val_split_size > 0:      
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
        if val_split_size > 0: 
            val_dataset = TensorDataset(torch.tensor(val_tokens["input_ids"].tolist()) ,
                                torch.tensor(val_tokens["attention_mask"].tolist()),
                                torch.tensor(val_tokens["labels"]),
                                torch.tensor(val_tokens["g_index"]))
        
        test_dataset = TensorDataset(torch.tensor(test_tokens["input_ids"].tolist()),
                             torch.tensor(test_tokens["attention_mask"].tolist()),
                             torch.tensor(test_tokens["labels"]),
                             torch.tensor(test_tokens["g_index"]))
        
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    if val_split_size > 0: 
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    return train_loader, val_loader, test_loader


def get_metrics(y_true , y_pred, history,  val_type = "Train",
                task_type = "classification"):
    
    
    if task_type in ["multilabel", "classification", "interaction"]:
    
        average = "samples" if task_type == "multilabel" else "weighted"
    
        acc= accuracy_score(y_true , y_pred)
        f1 = f1_score(y_true , y_pred, average=average, zero_division=np.nan)
        prec = precision_score(y_true , y_pred, average=average, zero_division=np.nan)
        rec = recall_score(y_true , y_pred, average=average, zero_division=np.nan)

        history[val_type]["Accuracy"].append(acc)
        history[val_type]["F1"].append(f1)
        history[val_type]["Precision"].append(prec)
        history[val_type]["Recall"].append(rec)
        
        return acc, f1 , prec , rec
    
    
    elif task_type == "unsupervised":
        return None
    
    elif task_type == "regression":
        
        corrcoef = spearmanr(y_true, y_pred)[0]
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)        
        
        history[val_type]["Correlation"].append(corrcoef)
        history[val_type]["MAE"].append(mae)
        history[val_type]["MSE"].append(mse)
        history[val_type]["R2"].append(r2)
        
        return corrcoef, mae, mse, r2
    
    else:
        raise ValueError(f"Key Error task_type : {task_type} ")
        
def process_Go_data(path):
    
    
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

            
    with open('data/knownGenes.json', 'r') as file:
        knownGenes = file.read(file)
                

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

def process_sent_new(text):
    """
        For new summary.
    """

    #pattern1 = r'(?:"(.*?)"|\'(.*?)\')'
    patterns = [ r'\(PubMed:\d+(?:, PubMed:\d+)*\)', r"\[PubMed 10453732\]", #r"\(PubMed:\d+(?:\s+\d+)*\)",
                r"\[provided by .*?\]", r"\[supplied by .*?\]", r"\(\s+[\w\s]+\s+[\w]+\s+\)",
                r"\[(Isoform [^\]]+)\]:\s*", r"\(By similarity\)", r'\([^)]+ et al\., \d{4} [^)]+\)',
                r'\([^)]*(?:\[PubMed \d+\]|PubMed: \d+)[^)]*\)', r'-+'
               ]
    
    for p in patterns:
        text = re.sub(p, "", text)
        
    
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\. \.', '.', text)
    text = text.split("Copyrighted by the UniProt Consortium")[0]
    
    
    return text

def process_sent(sent):

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

def plot_GO_accuracy_histogram(labels, preds, level=2):
    
    labels = np.array(labels)
    preds = np.array(preds)
    
    n_classes = labels.shape[1]
    accuracies = [accuracy_score(labels[:, i], preds[:, i]) for i in range(n_classes)]
    bars = plt.bar(range(n_classes), accuracies)
    plt.xlabel('GO Term')
    plt.ylabel('Accuracy')
    plt.title(f'Level {level}: GO Terms')
    plt.xticks(ticks=range(n_classes), labels=[f'{i+1}' for i in range(n_classes)])
    
#     for bar, acc in zip(bars, accuracies):
#         yval = bar.get_height() -0.5
#         plt.text(bar.get_x() + bar.get_width()/2, yval,
#                  f'{acc:.2f}', ha='center', va='center', rotation='vertical')

    plt.show()

def process_Go_data(path):
    
    
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

            
    with open('data/knownGenes.json', 'r') as file:
        knownGenes = file.read(file)
                

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

def map_strLabels(string_labels):

    class_map = dict()
    integer_labels = []

    
    for labels in string_labels:
        integer_label_list = []
        for label in labels:
            if label not in class_map:
                class_map[label] = len(class_map) + 1
            integer_label_list.append(class_map[label])
        integer_labels.append(integer_label_list)
    return integer_labels, class_map

def get_go_level(gene_list, required_level=None, obo_file = "data/GO/go-basic.obo"):
    
    with open(obo_file, 'r') as f:
        content = f.read().split("[Term]")
        
    terms = {}
    namespaces = {}
    for section in content:
        term_id = None
        parents = set()
        namespace = None
        
        for line in section.split("\n"):
            if line.startswith("id: "):
                term_id = line.split("id: ")[1]
            elif line.startswith("is_a: "):
                parents.add(line.split("is_a: ")[1].split(" ! ")[0])
            elif line.startswith("namespace: "):
                namespace = line.split("namespace: ")[1]
                
        if term_id:
            terms[term_id] = parents
            if namespace:
                namespaces[term_id] = namespace
    
    
    def get_level(term, terms, cache={}):
        """Recursively determine the level of a term."""
        if term not in terms:
            return 0
        if term in cache:
            return cache[term]

        levels = [get_level(parent, terms, cache) for parent in terms[term]]
        cache[term] = 1 + max(levels, default=0)
        return cache[term]
        

    mart_export_data = pd.read_csv('data/GO/mart_export.txt', delimiter=",")
    mart_export_data = mart_export_data[mart_export_data['Gene name'].isin(gene_list)].dropna()

    

    go_terms= mart_export_data["GO term accession"].dropna().unique()

    go_to_level = {term: get_level(term, terms) for term in go_terms}

    level_to_go = dict()
    
    for key, value in go_to_level.items():
        
        if value in level_to_go:
            level_to_go[value].append(key)
        else:
            level_to_go[value] = [key]
    
    
    if required_level:    
        return level_to_go.get(required_level, None)
    
    return level_to_go
    
def from_obo_to_triplet(genes, 
                        obo_file= "data/GO/go-basic.obo",
                        mart_file = "data/GO/mart_export.txt",
                        triplet="gene-wise", #term-wise
                        sample=5,
                        level = 2
                       ):
    go_terms_in_level = get_go_level(genes['Gene name'].tolist(), required_level= level)
    
    GO_gene = pd.read_csv(mart_file, delimiter=",").dropna(subset=["GO term accession", "Gene name"])
    GO_gene = GO_gene[GO_gene['GO term accession'].isin(go_terms_in_level)]
    
    GO_graph = obonet.read_obo(obo_file)
    
    data = {
        "Gene name":[],
        "TermId":[],
        "TermName":[],
        "TermSummary":[],
        "TermNamespace":[],
        "is_a":[]}
        
    for idx, row in GO_gene.iterrows():
        
        termId = row["GO term accession"].strip()
        
        #Make sure the returned Terms from the specified level
        if  GO_graph.nodes.get(termId, False):
            
            geneName = row["Gene name"].strip()

            data["TermId"].append(termId)
            data["TermName"].append(GO_graph.nodes[termId]["name"])
            data["TermSummary"].append(re.findall(r'"(.*?)"', GO_graph.nodes[termId]["def"])[0])
            data["TermNamespace"].append(GO_graph.nodes[termId].get("namespace", None))
            data["is_a"].append(GO_graph.nodes[termId].get("is_a", None))
            data["Gene name"].append(geneName)
    
    GO_df = pd.DataFrame(data)
    
    print(GO_df)
    
    GO_df = GO_df[GO_df['Gene name'].isin(genes['Gene name'].tolist())]
    
    
    termid_to_gene = GO_df.groupby('TermId')['Gene name'].apply(list).to_dict()    
    gene_to_termid = GO_df.groupby('Gene name')['TermId'].apply(list).to_dict()
    termid_to_summary = GO_df.drop_duplicates('TermId').set_index('TermId')['TermSummary'].to_dict()
    
    gene_to_summary =  genes[genes['Gene name'].isin(
        gene_to_termid.keys())].set_index('Gene name')['Summary'].to_dict()
    

    gene_to_positive = dict()
    gene_to_negative = dict()
    
    # Formulate the triplet data
    for g_name in gene_to_termid.keys(): 
        
        gene_to_positive[g_name]=[]
        gene_to_negative[g_name]=[]
        
        # positives and negatives are genes based on their GO Terms
        if triplet == "gene-wise": 

            # positive gene from same GO Terms
            possible_positives = []
            for term in gene_to_termid[g_name]:
                possible_positives += termid_to_gene[term]

                
            # negative gene from other GO Terms
            possible_negatives = []
            for term in set(termid_to_gene.keys()) - set(gene_to_termid[g_name]):
                possible_negatives += termid_to_gene[term]

                
            s = len(possible_negatives) if len(possible_negatives) < len(possible_positives) else len(possible_positives)
            s = sample if sample < s else s
            
            
            rand_pos_gene= random.sample(possible_positives, k=s)
            for poss_positive in rand_pos_gene:
                gene_to_positive[g_name].append({"Term":gene_to_termid[poss_positive],
                                                 "Summary": gene_to_summary[poss_positive],
                                                 "PositiveGene":poss_positive})


            rand_neg_gene= random.sample(possible_negatives, k=s)
            for poss_negative in rand_neg_gene:
                gene_to_negative[g_name].append({"Term":gene_to_termid[poss_negative],
                                                  "Summary": gene_to_summary[poss_negative],
                                                  "NegativeGene":poss_negative})

        # positives and negatives are the GO Terms
        elif triplet == "term-wise":

            possible_positives = gene_to_termid[g_name]
            
            #rand_pos_term= random.sample(possible_positives, k=1)[0]
            for poss_positive in possible_positives:
                gene_to_positive[g_name].append({"Term":poss_positive,
                                                 "Summary": termid_to_summary[poss_positive]})


            possible_negatives = list(set(termid_to_gene.keys()) - set(gene_to_termid[g_name]))
            
            #rand_neg_term= random.sample(possible_negatives, k=1)[0]
            for poss_negative in possible_negatives: 
                gene_to_negative[g_name].append({"Term":poss_negative,
                                                 "Summary": termid_to_summary[poss_negative]})

        else:
            raise ValueError(f"Unknown triplet Value: {triplet}")
    

    
    triplet_data ={
        "Gene name":[],
        "Anchor": [],
        "StrLabel":[],
        "Positive":[],
        "PositiveTerm":[],
        "Negative":[],
        "NegativeTerm":[]
    }
    
    
    
    if triplet == "gene-wise":
        triplet_data["PositiveGene"]=[]
        triplet_data["NegativeGene"]=[]
        
        
    #Build the dataframe that consist of triplets, and go term labels
    for g_name in tqdm(gene_to_positive.keys()):

        for _ in gene_to_positive[g_name]:
            triplet_data["Gene name"].append(g_name)
            triplet_data["Anchor"].append(gene_to_summary[g_name])
            triplet_data["StrLabel"].append(gene_to_termid[g_name])


        for example in gene_to_positive[g_name]:
            triplet_data["Positive"].append(example["Summary"])
            triplet_data["PositiveTerm"].append(example["Term"])

            if triplet == "gene-wise":
                triplet_data["PositiveGene"].append(example["PositiveGene"])
        
        for example in gene_to_negative[g_name]:
            triplet_data["Negative"].append(example["Summary"])
            triplet_data["NegativeTerm"].append(example["Term"])

            if triplet == "gene-wise":
                triplet_data["NegativeGene"].append(example["NegativeGene"])
        

        
#     print(len(triplet_data["Gene name"]))
#     print(len(triplet_data["Anchor"]))
#     print(len(triplet_data["StrLabel"]))
#     print(len(triplet_data["Positive"]))
#     print(len(triplet_data["PositiveTerm"]))
#     print(len(triplet_data["Negative"]))
#     print(len(triplet_data["NegativeTerm"]))
#     print(len(triplet_data["PositiveGene"]))
#     print(len(triplet_data["NegativeGene"]))
    triplet_df = pd.DataFrame(triplet_data)
    triplet_df = triplet_df.dropna().sample(frac=1).reset_index(drop=True)
    
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(triplet_df.StrLabel.to_list())
    int_to_class_map = dict(enumerate(mlb.classes_))
    class_to_int_map = {v: k for k, v in int_to_class_map.items()}

    
    triplet_df["Label"] = labels.tolist()
    
#     labels, class_map = map_strLabels(triplet_df.StrLabel.to_list())
    
    
    go_terms_df = df = pd.DataFrame(list(termid_to_summary.items()), columns=['TermId', 'Summary'])
    go_terms_df["Label"] = go_terms_df.TermId.apply(lambda t: class_to_int_map[t])
    
    return triplet_df, go_terms_df, int_to_class_map

def process_triplet_data(genes, max_length=512, batch_size=100,
                         val_genes = None , test_genes = None, 
                         test_split_size=0.7, gene2vec_flag=False,
                         model_name= "bert-base-cased"):
    
    
    if "xlnet" in model_name:
        tokenizer = XLNetTokenizer.from_pretrained(model_name)

    else:    
        tokenizer = BertTokenizerFast.from_pretrained(model_name)

    
#     labels = genes["Label"].tolist()
    anchor = genes["Anchor"].tolist()
    positive = genes["Positive"].tolist()
    negative = genes["Negative"].tolist() 
    g_index = genes.index.tolist()
    g_name = genes["Gene name"].tolist()
    

    tokens_a = tokenizer.batch_encode_plus(anchor, max_length = max_length,
                                           padding="max_length", truncation=True)

    tokens_p = tokenizer.batch_encode_plus(positive, max_length = max_length,
                                           padding="max_length", truncation=True)

    tokens_n = tokenizer.batch_encode_plus(negative, max_length = max_length,
                                           padding="max_length", truncation=True)

    
    #'token_type_ids': tokens["token_type_ids"],
    data = {
        'input_ids_a': tokens_a["input_ids"],
        'attention_mask_a': tokens_a["attention_mask"],

        'input_ids_p': tokens_p["input_ids"],
        'attention_mask_p': tokens_p["attention_mask"],

        'input_ids_n': tokens_n["input_ids"],
        'attention_mask_n': tokens_n["attention_mask"],
        
        "g_index": g_index,
        "g_name": g_name,
#         "labels": labels,
    }
    
    if gene2vec_flag:
        data["PositiveGene"]= genes["PositiveGene"].tolist()
        data["NegativeGene"]= genes["NegativeGene"].tolist()
    
    
    tokens_df = pd.DataFrame(data)

    print(f"Shape of tokens_df before gene2vec:{tokens_df.shape}")
    
    #############################################
    if gene2vec_flag:
        print("Adding Gene2Vec data ...")
        
        Gene2Vec = dict()

        file_path = f'data/gene2vec_embeddings.txt'
        with open(file_path, 'r') as file:
            for line in file:

                name, embed = line.strip().split("	")
                embed = [float(value) for value in embed.split()] 

                Gene2Vec[name.strip()] = embed
                    
            
#         common_genes = set(Gene2Vec.keys()) & (set(tokens_df["g_name"]) | set(tokens_df["PositiveGene"]) | set(tokens_df["NegativeGene"]))
#         tokens_df = tokens_df[tokens_df['g_name'].isin(common_genes)]
        
        tokens_df["gene2vec_a"] = tokens_df["g_name"].apply(lambda name: Gene2Vec.get(name, None))
        tokens_df["gene2vec_p"] = tokens_df["PositiveGene"].apply(lambda name: Gene2Vec.get(name, None))
        tokens_df["gene2vec_n"] = tokens_df["NegativeGene"].apply(lambda name: Gene2Vec.get(name, None))
        tokens_df = tokens_df.dropna()
    
    #############################################
    print(f"Shape of tokens_df after gene2vec:{tokens_df.shape}")

    

    if val_genes is not None:
        val_tokens = tokens_df.loc[val_genes]
        test_tokens =  tokens_df.loc[test_genes]
        train_tokens = tokens_df.drop(val_genes+test_genes)
        
    else:

        train_tokens, test_tokens = train_test_split(tokens_df, test_size=test_split_size,
                                                     random_state=1)

        test_tokens, val_tokens = train_test_split(test_tokens,test_size=0.5,
                                                   random_state=1)


    train_tokens = train_tokens.reset_index(drop=True)
    val_tokens = val_tokens.reset_index(drop=True)
    test_tokens = test_tokens.reset_index(drop=True)

    if gene2vec_flag:
        train_dataset = TensorDataset(
            torch.tensor(train_tokens["input_ids_a"].tolist()),
            torch.tensor(train_tokens["attention_mask_a"].tolist()),
            torch.tensor(train_tokens["gene2vec_a"]),
            
            torch.tensor(train_tokens["input_ids_p"].tolist()),
            torch.tensor(train_tokens["attention_mask_p"].tolist()),
            torch.tensor(train_tokens["gene2vec_p"]),
            
            torch.tensor(train_tokens["input_ids_n"].tolist()),
            torch.tensor(train_tokens["attention_mask_n"].tolist()),
            torch.tensor(train_tokens["gene2vec_n"]),
            
#             torch.tensor(train_tokens["labels"]),
            torch.tensor(train_tokens["g_index"]))

        val_dataset = TensorDataset(
            torch.tensor(val_tokens["input_ids_a"].tolist()) ,
            torch.tensor(val_tokens["attention_mask_a"].tolist()),   
            torch.tensor(val_tokens["gene2vec_a"]),
            
            torch.tensor(val_tokens["input_ids_p"].tolist()) ,
            torch.tensor(val_tokens["attention_mask_p"].tolist()),
            torch.tensor(val_tokens["gene2vec_p"]),
            
            torch.tensor(val_tokens["input_ids_n"].tolist()) ,
            torch.tensor(val_tokens["attention_mask_n"].tolist()),
            torch.tensor(val_tokens["gene2vec_n"]),
            
#             torch.tensor(val_tokens["labels"]),
            torch.tensor(val_tokens["g_index"]))

        test_dataset = TensorDataset(
            torch.tensor(test_tokens["input_ids_a"].tolist()),
            torch.tensor(test_tokens["attention_mask_a"].tolist()),
            torch.tensor(test_tokens["gene2vec_a"]),
            
            torch.tensor(test_tokens["input_ids_p"].tolist()),
            torch.tensor(test_tokens["attention_mask_p"].tolist()),
            torch.tensor(test_tokens["gene2vec_p"]),
            
            torch.tensor(test_tokens["input_ids_n"].tolist()),
            torch.tensor(test_tokens["attention_mask_n"].tolist()),
            torch.tensor(test_tokens["gene2vec_n"]),
            
#             torch.tensor(test_tokens["labels"]),
            torch.tensor(test_tokens["g_index"]))
    
    else:
        
        train_dataset = TensorDataset(
            torch.tensor(train_tokens["input_ids_a"].tolist()),
            torch.tensor(train_tokens["attention_mask_a"].tolist()),
            torch.tensor(train_tokens["input_ids_p"].tolist()),
            torch.tensor(train_tokens["attention_mask_p"].tolist()),
            torch.tensor(train_tokens["input_ids_n"].tolist()),
            torch.tensor(train_tokens["attention_mask_n"].tolist()),
#             torch.tensor(train_tokens["labels"]),
            torch.tensor(train_tokens["g_index"]))

        val_dataset = TensorDataset(
            torch.tensor(val_tokens["input_ids_a"].tolist()) ,
            torch.tensor(val_tokens["attention_mask_a"].tolist()),        
            torch.tensor(val_tokens["input_ids_p"].tolist()) ,
            torch.tensor(val_tokens["attention_mask_p"].tolist()),
            torch.tensor(val_tokens["input_ids_n"].tolist()) ,
            torch.tensor(val_tokens["attention_mask_n"].tolist()),
#             torch.tensor(val_tokens["labels"]),
            torch.tensor(val_tokens["g_index"]))

        test_dataset = TensorDataset(
            torch.tensor(test_tokens["input_ids_a"].tolist()),
            torch.tensor(test_tokens["attention_mask_a"].tolist()),
            torch.tensor(test_tokens["input_ids_p"].tolist()),
            torch.tensor(test_tokens["attention_mask_p"].tolist()),
            torch.tensor(test_tokens["input_ids_n"].tolist()),
            torch.tensor(test_tokens["attention_mask_n"].tolist()),
#             torch.tensor(test_tokens["labels"]),
            torch.tensor(test_tokens["g_index"]))

        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    
    return train_loader, val_loader, test_loader
