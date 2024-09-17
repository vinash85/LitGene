###########################################################
# Project: GeneLLM
# File: model.py
# License: MIT
# Code Authors: Jararaweh A., Macualay O.S, Arredondo D., & 
#               Virupakshappa K.
###########################################################
import os
import torch
from model import FineTunedBERT
from transformers import BertTokenizerFast
from transformers import XLNetTokenizer
from torch.utils.data import DataLoader, TensorDataset
import warnings
import pandas as pd
import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from data_processor import loading_data, process_data
import shutil
import gc


def train(loader, model, loss_fn, optimizer, task_type= None,
          gene2vec_flag = True, device = "cuda",
          threshold=0.5 ):
    """
        task : conservation , sublocation, solubility
        task_type : regression or classification
    
    
    """
    
    train_loss = 0
    latents  = []

    total_preds = []
    total_labels = []
    probabilities = []
    
    # conservation , sublocation, solubility
   

    
    model.train()
    for batch in loader: 

        
        if gene2vec_flag:
            batch_inputs, batch_masks, gene2vec_embeddings, labels =  batch[0].to(device) , batch[1].to(device), batch[2].to(device), batch[3].to(device)
            embeddings, _ , preds = model(batch_inputs, batch_masks, gene2vec = gene2vec_embeddings)
            
            
        else:
            batch_inputs, batch_masks , labels =  batch[0].to(device) , batch[1].to(device), batch[2].to(device)
            embeddings, _ , preds = model(batch_inputs, batch_masks)

        
        
        if task_type == "regression":
            
            preds = preds.squeeze().float()
            labels = labels.squeeze().float()
            
            loss = loss_fn(preds, labels) 
            
            
            total_preds.extend(preds.cpu().detach())
            total_labels.extend(labels.cpu().detach())        

        
        elif task_type == "classification":
            # print(preds.shape, labels.shape)
            # print(preds, labels)
            
            loss = loss_fn(preds, labels)
            
            total_preds.extend(preds.argmax(1).type(torch.int).to('cpu').numpy())
            total_labels.extend(labels.type(torch.int).to('cpu').numpy())
            probas = F.softmax(preds, dim=1)
            probabilities.extend(probas.detach().cpu().numpy())

        
        
        elif task_type == "multilabel":

            preds = preds.to(torch.float32)
            labels = labels.to(torch.float32)

            loss = loss_fn(preds, labels)
            
            total_labels.extend(labels.cpu().type(torch.int).numpy().tolist())
            total_preds.extend((preds > 0.5).type(torch.int).cpu().numpy().tolist())
            probas = torch.sigmoid(preds)
            probabilities.extend(probas.detach().cpu().numpy())
            
        
        train_loss += loss.item()

        
        #Aggregation
        embeddings = torch.tensor(embeddings.cpu().detach().numpy())
        latents.append(embeddings) 
        

        model.zero_grad()
        loss.backward()
        optimizer.step()
        
    train_loss /= len(loader)
    latents = torch.cat(latents, dim=0)

    return model, train_loss, total_labels, total_preds, latents, probabilities

def validation (loader, model, loss_fn, task_type = None,
                gene2vec_flag = True, device = "cuda"):
    """
    @Ala and @Macualy provide description
    """
    val_loss = 0
    total_preds = []
    total_labels = []
    probabilities = []

    
    model.eval()
    with torch.no_grad():
        for batch in loader: 

            if gene2vec_flag:
                batch_inputs, batch_masks, gene2vec_embeddings, labels =  batch[0].to(device) , batch[1].to(device), batch[2].to(device), batch[3].to(device)
                embeddings, _ , preds = model(batch_inputs, batch_masks, gene2vec = gene2vec_embeddings)




            else:
                batch_inputs, batch_masks , labels =  batch[0].to(device) , batch[1].to(device), batch[2].to(device)
                embeddings, _ , preds = model(batch_inputs, batch_masks)

                
                
            if task_type == "regression":
            
                preds = preds.squeeze().float()
                labels = labels.squeeze().float()

                loss = loss_fn(preds, labels)


                total_preds.extend(preds.cpu().detach())
                total_labels.extend(labels.cpu().detach())        

        
            elif task_type == "classification":

                loss = loss_fn(preds, labels)

                total_preds.extend(preds.argmax(1).type(torch.int).to('cpu').numpy())
                total_labels.extend(labels.type(torch.int).to('cpu').numpy())
                probas = F.softmax(preds, dim=1)
                probabilities.extend(probas.detach().cpu().numpy())

            elif task_type == "multilabel":

                preds = preds.to(torch.float32)
                labels = labels.to(torch.float32)


                loss = loss_fn(preds, labels)

                total_labels.extend(labels.cpu().type(torch.int).numpy().tolist())
                total_preds.extend((preds > 0.5).type(torch.int).cpu().numpy().tolist())
                probas = torch.sigmoid(preds)
                probabilities.extend(probas.detach().cpu().numpy())

            
            val_loss += loss.item()
                

    val_loss /= len(loader)

    return model, val_loss, total_labels, total_preds, probabilities

def test(loader, model, loss_fn, task_type = None, gene2vec_flag = True, device = "cuda"):
    """
    @Ala and @Macualy provide description
    """
    test_loss = 0
    total_preds = []
    total_labels = []
    latents = []
    probabilities = []

    model.eval()
    with torch.no_grad():
        for batch in loader: 

            if gene2vec_flag:
                batch_inputs, batch_masks, gene2vec_embeddings, labels =  batch[0].to(device) , batch[1].to(device), batch[2].to(device), batch[3].to(device)
                embeddings, _ , preds = model(batch_inputs, batch_masks, gene2vec = gene2vec_embeddings)
                


            else:
                batch_inputs, batch_masks, labels =  batch[0].to(device) , batch[1].to(device), batch[2].to(device)
                embeddings, _ , preds = model(batch_inputs, batch_masks)      
            
            
            if task_type == "regression":
            
                preds = preds.squeeze().float()
                labels = labels.squeeze().float()

                loss = loss_fn(preds, labels)
 

                total_preds.extend(preds.cpu().detach())
                total_labels.extend(labels.cpu().detach())        

        
            elif task_type == "classification":

                loss = loss_fn(preds, labels)

                total_preds.extend(preds.argmax(1).type(torch.int).to('cpu').numpy())
                total_labels.extend(labels.type(torch.int).to('cpu').numpy())
                # softmax to get probabilities
                probas = F.softmax(preds, dim=1)
                probabilities.extend(probas.detach().cpu().numpy())

            
            elif task_type == "multilabel":

                preds = preds.to(torch.float32)
                labels = labels.to(torch.float32)


                loss = loss_fn(preds, labels)

                total_labels.extend(labels.cpu().type(torch.int).numpy().tolist())
                total_preds.extend((preds > 0.5).type(torch.int).cpu().numpy().tolist())
                #sigmoid to get probabilities for multilabel case
                probas = torch.sigmoid(preds)
                probabilities.extend(probas.detach().cpu().numpy())

            
            test_loss += loss.item()
            

            embeddings = torch.tensor(embeddings.cpu().detach().numpy())
            latents.append(embeddings)

    test_loss /= len(loader)
    latents = torch.cat(latents, dim=0)
       
    return model, test_loss, total_labels, total_preds, latents, probabilities

def get_metrics(y_true , y_pred, y_prob, history,  val_type = "Train",
                task_type = "classification"):
    
    """
    @Ala and @Macualy provide description
    """
    if task_type == "classification" or task_type == "multilabel":
    
        average = "samples" if task_type == "multilabel" else "weighted"
    
        acc= accuracy_score(y_true , y_pred)
        f1 = f1_score(y_true , y_pred, average=average, zero_division=np.nan)
        prec = precision_score(y_true , y_pred, average=average, zero_division=np.nan)
        rec = recall_score(y_true , y_pred, average=average, zero_division=np.nan)


        # Calculate ROC AUC
        if len(np.unique(y_true)) == 2:
            roc_auc = roc_auc_score(y_true, y_pred)  
            aupr = average_precision_score(y_true, y_pred, average="weighted")
        else:
            roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            aupr = average_precision_score(y_true, y_prob, average="weighted")



        history[val_type]["Accuracy"].append(acc)
        history[val_type]["F1"].append(f1)
        history[val_type]["Precision"].append(prec)
        history[val_type]["Recall"].append(rec)
        history[val_type]["ROC AUC"].append(roc_auc)
        history[val_type]["AUPR"].append(aupr)
        
        return acc, f1 , prec , rec, roc_auc, aupr
        
    else:
        
        
        s_corrcoef = spearmanr(y_true, y_pred)[0]
        p_corrcoef = pearsonr(y_true, y_pred)[0]
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)        
        
        history[val_type]["S_Correlation"].append(s_corrcoef)
        history[val_type]["P_Correlation"].append(p_corrcoef)
        history[val_type]["MAE"].append(mae)
        history[val_type]["MSE"].append(mse)
        history[val_type]["R2"].append(r2)
        
        
        return s_corrcoef,p_corrcoef, mae, mse, r2

def plot_latent(latents, labels, epoch, class_map = None,
                task_name= "subloc", validation_type="train", model_type="bert"):
    """
    @Ala and @Macualy provide description
    """
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
                        label=f'{cl}')
        plt.legend()
        
    else:
        plt.scatter(latents_tsne[:, 0], latents_tsne[:, 1])

        

    plt.title(f'Epoch {epoch}')
    task_path = f"data/{task_name}/{model_type}/"
    if not os.path.exists(task_path):
        os.makedirs(task_path)

    validation_path = f"{task_path}/{validation_type}"
    if not os.path.exists(validation_path):
        os.makedirs(validation_path)

    plt.savefig(f"{validation_path}/latent_{epoch}.png")
    plt.close()

def plot_metrics(data_path, task_name, model_type, best_epoch_num, task_type, k_fold):
    """
    @Ala and @Macualy provide description
    """
    if task_type == "regression":
        df = pd.read_csv(f'{data_path}/{task_name}/{model_type}/metrics_{task_name}_fold_{k_fold}.csv')
        df_filtered = df[df['epoch'] <= best_epoch_num]
        plt.figure(figsize=(12, 5))

        # Plot for Loss
        plt.subplot(1, 2, 1)
        plt.plot(df_filtered['epoch'], df_filtered['train_loss'], label='Train Loss', marker='o')
        plt.plot(df_filtered['epoch'], df_filtered['val_loss'], label='Validation Loss', marker='o')
        # plt.plot(df_filtered['epoch'], df_filtered['test_loss'], label='Test Loss', marker='o')
        plt.title('Loss across Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Plot for Correlation Coefficients
        plt.subplot(1, 2, 2)
        plt.plot(df_filtered['epoch'], df_filtered['s_train_corr'], label='Train Correlation', marker='o')
        plt.plot(df_filtered['epoch'], df_filtered['s_val_corr'], label='Validation Correlation', marker='o')
        # plt.plot(df_filtered['epoch'], df_filtered['test_corr'], label='Test Correlation', marker='o')
        plt.title('Correlation Coefficients across Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Correlation Coefficient')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f'{data_path}/{task_name}/{model_type}/finetuning_loss_{task_name}.png')
        plt.show()

    elif task_type == "classification" or task_type == "multilabel":
        df = pd.read_csv(f'{data_path}/{task_name}/{model_type}/metrics_{task_name}_fold_{k_fold}.csv')
        df_filtered = df[df['epoch'] <= best_epoch_num]
        plt.figure(figsize=(12, 5))

        # Plot for Loss
        plt.subplot(1, 2, 1)
        plt.plot(df_filtered['epoch'], df_filtered['train_loss'], label='Train Loss', marker='o')
        plt.plot(df_filtered['epoch'], df_filtered['val_loss'], label='Validation Loss', marker='o')
        # plt.plot(df_filtered['epoch'], df_filtered['test_loss'], label='Test Loss', marker='o')
        plt.title('Loss across Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Plot for Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(df_filtered['epoch'], df_filtered['train_acc'], label='Train Accuracy', marker='o')
        plt.plot(df_filtered['epoch'], df_filtered['val_acc'], label='Validation Accuracy', marker='o')
        # plt.plot(df_filtered['epoch'], df_filtered['test_acc'], label='Test Accuracy', marker='o')
        plt.title('Accuracy across Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f'{data_path}/{task_name}/{model_type}/finetuning_loss_{task_name}.png')
        plt.show()

def plot_GO_accuracy_histogram(labels, preds, level=2):
    """
    @Ala and @Macualy provide description
    """
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


def save_finetuned_embeddings(genes, pool = "cls", max_length= 100, batch_size =100, drop_rate =0.1,
                gene2vec_flag = True, gene2vec_hidden = 200, device = "cuda",
                task_type = "classification", n_labels = 3 , model_name = "microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract", task_name = 'Subcellular_location', model_type='Pubmed_large'): 


    model = FineTunedBERT(pool= pool, model_name = model_name, bert_state_dict=None, task_type = task_type, n_labels = n_labels,
                          drop_rate = drop_rate, 
                          gene2vec_flag= gene2vec_flag,
                          gene2vec_hidden = gene2vec_hidden, device ="cuda").to(device)

    # optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
    state_dict = torch.load(f'{data_path}/{task_name}/{model_type}/best_model_{task_name}.pth')
    model.load_state_dict(state_dict)

    # Tokenize the gene summaries
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    encoded_summaries = tokenizer.batch_encode_plus(genes["Summary"].tolist(), 
                                                   max_length=max_length, 
                                                   padding="max_length",
                                                   truncation=True,
                                                   return_tensors="pt")

    # DataLoader for all genes
    all_dataset = TensorDataset(encoded_summaries["input_ids"], encoded_summaries["attention_mask"])
    all_data_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=False)

    # Store gene names separately
    all_gene_names = genes["Gene name"].tolist()

    # Get embeddings for all genes
    all_embeddings = []
    model.eval()
    with torch.no_grad():
        for idx, (inputs, masks) in enumerate(all_data_loader):
            embeddings, _, _ = model(inputs.to(device), masks.to(device))
            all_embeddings.append(embeddings.cpu().numpy())

    # Flatten embeddings list
    all_embeddings = np.vstack(all_embeddings)

    
    embeddings_filename = f'{data_path}/{task_name}/{model_type}/fine_tuned_embeddings_{task_name}.csv'
    embeddings_df = pd.DataFrame(all_embeddings)
    embeddings_df['gene_name'] = all_gene_names  

    
    embeddings_df.to_csv(embeddings_filename, index=False)
 

    embeddings_df = pd.concat([embeddings_df.iloc[:, -1], embeddings_df.iloc[:, :-1]], axis=1)
    embeddings_df.columns = [''] * len(embeddings_df.columns)
    embeddings_filename = f'{data_path}/{task_name}/{model_type}/fine_tuned_embeddings_{task_name}.csv'
    embeddings_df.to_csv(embeddings_filename, header=False, index=False)
    print(f'Fine-tuned embeddings saved to {embeddings_filename}')
    print(f'best epoch number: {best_epoch_num}')

