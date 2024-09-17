
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizerFast, XLNetTokenizer, AdamW
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import random
import re
import warnings
from tqdm import tqdm
import pickle
from scipy.stats import spearmanr
import time

from src.litgene import FineTunedBERT
from src.utils import get_metrics


def get_triplet_loss(anchEmbeddings, posEmbeddings, negEmbeddings, margin =0.5):

    triplet_loss = TripletMarginLoss(p=2, margin=margin)

    TLoss = triplet_loss(anchEmbeddings, posEmbeddings, negEmbeddings)

    return TLoss


def train(loader, model, loss_fn, optimizer, task_type= None,
          gene2vec_flag = False, device = "cuda",
          contrastive_flag = False, margin = 0.5, alpha=0.5):
    """
        task : conservation , sublocation, solubility
        task_type : regression or classification
    
    
    """
    
    train_loss = 0
    total_tloss = 0
    latents  = []

    total_preds = []
    total_labels = []

    
    model.train()
    for batch in tqdm(loader): 
        
        if task_type.lower() in ["unsupervised", "interaction"]:
            if gene2vec_flag:
                batch_inputs_a, batch_masks_a, gene2vec_a  =  batch[0].to(device) , batch[1].to(device), batch[2].to(device)
                batch_inputs_p, batch_masks_p, gene2vec_p  =  batch[3].to(device) , batch[4].to(device), batch[5].to(device)
                batch_inputs_n, batch_masks_n, gene2vec_n  =  batch[6].to(device) , batch[7].to(device), batch[8].to(device)
                

                embeddings, _ , _ = model(batch_inputs_a, batch_masks_a, gene2vec =gene2vec_a)
                embeddings_p, _ , _ = model(batch_inputs_p, batch_masks_p, gene2vec = gene2vec_p)
                embeddings_n, _ , _ = model(batch_inputs_n, batch_masks_n, gene2vec = gene2vec_n)               

            
            else:
                batch_inputs_a, batch_masks_a  =  batch[0].to(device) , batch[1].to(device)
                batch_inputs_p, batch_masks_p  =  batch[2].to(device) , batch[3].to(device)
                batch_inputs_n, batch_masks_n  =  batch[4].to(device) , batch[5].to(device)
                

                embeddings, _ , _ = model(batch_inputs_a, batch_masks_a)
                embeddings_p, _ , _ = model(batch_inputs_p, batch_masks_p)
                embeddings_n, _ , _ = model(batch_inputs_n, batch_masks_n)               
      
        
        elif task_type.lower() in ["classification", "multilabel", "regression"]:
            
            if gene2vec_flag and contrastive_flag:
                
                batch_inputs_a, batch_masks_a, gene2vec_a  =  batch[0].to(device) , batch[1].to(device), batch[2].to(device)
                batch_inputs_p, batch_masks_p, gene2vec_p  =  batch[3].to(device) , batch[4].to(device), batch[5].to(device)
                batch_inputs_n, batch_masks_n, gene2vec_n  =  batch[6].to(device) , batch[7].to(device), batch[8].to(device)
                labels = batch[9].to(device)

                embeddings, _ , preds = model(batch_inputs_a, batch_masks_a, gene2vec =gene2vec_a)
                embeddings_p, _ , _ = model(batch_inputs_p, batch_masks_p, gene2vec = gene2vec_p)
                embeddings_n, _ , _ = model(batch_inputs_n, batch_masks_n, gene2vec = gene2vec_n)               

            
            elif contrastive_flag:
                
                batch_inputs_a, batch_masks_a  =  batch[0].to(device) , batch[1].to(device)
                batch_inputs_p, batch_masks_p  =  batch[2].to(device) , batch[3].to(device)
                batch_inputs_n, batch_masks_n  =  batch[4].to(device) , batch[5].to(device)
                labels = batch[6].to(device)

                embeddings, _ , preds = model(batch_inputs_a, batch_masks_a )
                embeddings_p, _ , _ = model(batch_inputs_p, batch_masks_p )
                embeddings_n, _ , _ = model(batch_inputs_n, batch_masks_n )
            
            elif gene2vec_flag:
                batch_inputs, batch_masks, gene2vec_embeddings, labels =  batch[0].to(device) , batch[1].to(device), batch[2].to(device), batch[3].to(device)
                embeddings, _ , preds = model(batch_inputs, batch_masks, gene2vec = gene2vec_embeddings)
            
            else:
                batch_inputs, batch_masks , labels =  batch[0].to(device) , batch[1].to(device), batch[2].to(device)
                embeddings, _ , preds = model(batch_inputs, batch_masks)

        
        
        if task_type == "regression":
            
            preds = preds.squeeze().float()
            labels = labels.squeeze().float()
            
            if contrastive_flag:
                TLoss = get_triplet_loss(embeddings, embeddings_p, embeddings_n, margin=margin)
                bert_loss = loss_fn(preds, labels)
                
                loss = bert_loss + alpha*TLoss 
                total_tloss += TLoss
            
            else:
                loss = loss_fn(preds, labels)
            
            
            total_preds.extend(preds.cpu().detach())
            total_labels.extend(labels.cpu().detach())        

        
        elif task_type == "classification":
            
            if contrastive_flag:
                TLoss = get_triplet_loss(embeddings, embeddings_p, embeddings_n, margin=margin)
                bert_loss = loss_fn(preds, labels)
                
                loss = bert_loss + alpha*TLoss
                total_tloss += TLoss
            
            else:
#                 print(preds)
#                 print(labels)
                loss = loss_fn(preds, labels)
            
            total_preds.extend(preds.argmax(1).type(torch.int).to('cpu').numpy())
            total_labels.extend(labels.type(torch.int).to('cpu').numpy())

        
        
        elif task_type == "multilabel":

            preds = preds.to(torch.float32)
            labels = labels.to(torch.float32)

            loss = loss_fn(preds, labels)
            
            total_labels.extend(labels.cpu().type(torch.int).numpy().tolist())
            total_preds.extend((preds > 0.5).type(torch.int).cpu().numpy().tolist())
            
        elif task_type == "unsupervised":
            
            
            loss = loss_fn(embeddings, embeddings_p, embeddings_n)
            
            labels = batch[6]
            total_labels.extend(labels.type(torch.int).numpy().tolist())
        
        elif task_type =="interaction":
            
            embeddings_a = embeddings
            embeddings = torch.cat((embeddings, embeddings), dim=0)
            embeddings_interact = torch.cat((embeddings_p, embeddings_n), dim=0)
            
            preds = Fusion(embeddings, embeddings_interact)
            

            labels = torch.cat((torch.ones(embeddings_p.size(0),1),
                                torch.zeros(embeddings_n.size(0),1)),
                               dim=0).to(preds.device)

            
            #loss = loss_fn(preds, labels)
            loss = loss_fn[0](preds, labels) + loss_fn[1](embeddings, embeddings_p, embeddings_n)
            
            total_labels.extend(labels.cpu().type(torch.int).numpy().tolist())
            total_preds.extend((preds > 0.5).type(torch.int).cpu().numpy().tolist())

        
        
        train_loss += loss.item()

        
        #Aggregation
        
        embeddings = torch.tensor(embeddings.cpu().detach().numpy())
        latents.append(embeddings) 
        
        
        model.zero_grad()
        loss.backward()
        optimizer.step()
        
    train_loss /= len(loader)
    latents = torch.cat(latents, dim=0)
    
    if contrastive_flag:
        print(f"Triplet loss:{total_tloss/len(loader)}")

    return model, train_loss, total_labels, total_preds, latents


def validation (loader, model, loss_fn, task_type = None,
                gene2vec_flag = False, device = "cuda", 
                contrastive_flag = False, margin = 0.5, alpha=0.5):
    
    val_loss = 0 
    total_tloss = 0
    total_preds = []
    total_labels = []

    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader): 

            if task_type.lower() in ["unsupervised", "interaction"]:
                if gene2vec_flag:
                    batch_inputs_a, batch_masks_a, gene2vec_a  =  batch[0].to(device) , batch[1].to(device), batch[2].to(device)
                    batch_inputs_p, batch_masks_p, gene2vec_p  =  batch[3].to(device) , batch[4].to(device), batch[5].to(device)
                    batch_inputs_n, batch_masks_n, gene2vec_n  =  batch[6].to(device) , batch[7].to(device), batch[8].to(device)


                    embeddings, _ , _ = model(batch_inputs_a, batch_masks_a, gene2vec =gene2vec_a)
                    embeddings_p, _ , _ = model(batch_inputs_p, batch_masks_p, gene2vec = gene2vec_p)
                    embeddings_n, _ , _ = model(batch_inputs_n, batch_masks_n, gene2vec = gene2vec_n)               


                else:
                    batch_inputs_a, batch_masks_a  =  batch[0].to(device) , batch[1].to(device)
                    batch_inputs_p, batch_masks_p  =  batch[2].to(device) , batch[3].to(device)
                    batch_inputs_n, batch_masks_n  =  batch[4].to(device) , batch[5].to(device)


                    embeddings, _ , _ = model(batch_inputs_a, batch_masks_a)
                    embeddings_p, _ , _ = model(batch_inputs_p, batch_masks_p)
                    embeddings_n, _ , _ = model(batch_inputs_n, batch_masks_n)               


            elif task_type.lower() in ["classification", "multilabel", "regression"]:

                if gene2vec_flag and contrastive_flag:

                    batch_inputs_a, batch_masks_a, gene2vec_a  =  batch[0].to(device) , batch[1].to(device), batch[2].to(device)
                    batch_inputs_p, batch_masks_p, gene2vec_p  =  batch[3].to(device) , batch[4].to(device), batch[5].to(device)
                    batch_inputs_n, batch_masks_n, gene2vec_n  =  batch[6].to(device) , batch[7].to(device), batch[8].to(device)
                    labels = batch[9].to(device)

                    embeddings, _ , preds = model(batch_inputs_a, batch_masks_a, gene2vec =gene2vec_a)
                    embeddings_p, _ , _ = model(batch_inputs_p, batch_masks_p, gene2vec = gene2vec_p)
                    embeddings_n, _ , _ = model(batch_inputs_n, batch_masks_n, gene2vec = gene2vec_n)               


                elif contrastive_flag:

                    batch_inputs_a, batch_masks_a  =  batch[0].to(device) , batch[1].to(device)
                    batch_inputs_p, batch_masks_p  =  batch[2].to(device) , batch[3].to(device)
                    batch_inputs_n, batch_masks_n  =  batch[4].to(device) , batch[5].to(device)
                    labels = batch[6].to(device)

                    embeddings, _ , preds = model(batch_inputs_a, batch_masks_a )
                    embeddings_p, _ , _ = model(batch_inputs_p, batch_masks_p )
                    embeddings_n, _ , _ = model(batch_inputs_n, batch_masks_n )

                elif gene2vec_flag:
                    batch_inputs, batch_masks, gene2vec_embeddings, labels =  batch[0].to(device) , batch[1].to(device), batch[2].to(device), batch[3].to(device)
                    embeddings, _ , preds = model(batch_inputs, batch_masks, gene2vec = gene2vec_embeddings)

                else:
                    batch_inputs, batch_masks , labels =  batch[0].to(device) , batch[1].to(device), batch[2].to(device)
                    embeddings, _ , preds = model(batch_inputs, batch_masks)


                
            if task_type == "regression":
            
                preds = preds.squeeze().float()
                labels = labels.squeeze().float()

                if contrastive_flag:
                    TLoss = get_triplet_loss(embeddings, embeddings_p, embeddings_n, margin=margin)
                    bert_loss = loss_fn(preds, labels)

                    loss = bert_loss + alpha*TLoss
                    total_tloss +=TLoss.item ()

                else:
                    loss = loss_fn(preds, labels)


                total_preds.extend(preds.cpu().detach())
                total_labels.extend(labels.cpu().detach())        

        
            elif task_type == "classification":

                if contrastive_flag:
                    TLoss = get_triplet_loss(embeddings, embeddings_p, embeddings_n, margin=margin)
                    bert_loss = loss_fn(preds, labels)

                    loss = bert_loss + alpha*TLoss
                    total_tloss +=TLoss.item ()

                else:
                    loss = loss_fn(preds, labels)

                total_preds.extend(preds.argmax(1).type(torch.int).to('cpu').numpy())
                total_labels.extend(labels.type(torch.int).to('cpu').numpy())

            elif task_type == "multilabel":

                preds = preds.to(torch.float32)
                labels = labels.to(torch.float32)


                loss = loss_fn(preds, labels)

                total_labels.extend(labels.cpu().type(torch.int).numpy().tolist())
                total_preds.extend((preds > 0.5).type(torch.int).cpu().numpy().tolist())
            
            elif task_type == "unsupervised":
            
                loss = loss_fn(embeddings, embeddings_p, embeddings_n)
                
                labels = batch[6]
                total_labels.extend(labels.type(torch.int).numpy().tolist())
            
            elif task_type =="interaction":
                embeddings_a = embeddings
                
                embeddings = torch.cat((embeddings, embeddings), dim=0)
                embeddings_interact = torch.cat((embeddings_p, embeddings_n), dim=0)

                preds = Fusion(embeddings, embeddings_interact)
                labels = torch.cat((torch.ones(embeddings_p.size(0),1),
                                    torch.zeros(embeddings_n.size(0),1)),
                                   dim=0).to(preds.device)

#                 loss = loss_fn(preds, labels)
                
                loss = loss_fn[0](preds, labels) + loss_fn[1](embeddings_a, embeddings_p, embeddings_n)

                total_labels.extend(labels.cpu().type(torch.int).numpy().tolist())
                total_preds.extend((preds > 0.5).type(torch.int).cpu().numpy().tolist())
            
            
            
            val_loss += loss.item()
                

    val_loss /= len(loader)
    
    if contrastive_flag:
        print(f"Triplet loss:{total_tloss/len(loader)}")

    return model, val_loss, total_labels, total_preds


def test(loader, model, loss_fn, task_type = None,
         gene2vec_flag = False, device = "cuda",
         contrastive_flag = False, margin = 0.5, alpha=0.5):
    
    test_loss = 0
    total_tloss = 0
    total_preds = []
    total_labels = []
    latents = []    
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader): 

            if task_type.lower() in ["unsupervised", "interaction"]:
                if gene2vec_flag:
                    batch_inputs_a, batch_masks_a, gene2vec_a  =  batch[0].to(device) , batch[1].to(device), batch[2].to(device)
                    batch_inputs_p, batch_masks_p, gene2vec_p  =  batch[3].to(device) , batch[4].to(device), batch[5].to(device)
                    batch_inputs_n, batch_masks_n, gene2vec_n  =  batch[6].to(device) , batch[7].to(device), batch[8].to(device)


                    embeddings, _ , _ = model(batch_inputs_a, batch_masks_a, gene2vec =gene2vec_a)
                    embeddings_p, _ , _ = model(batch_inputs_p, batch_masks_p, gene2vec = gene2vec_p)
                    embeddings_n, _ , _ = model(batch_inputs_n, batch_masks_n, gene2vec = gene2vec_n)               


                else:
                    batch_inputs_a, batch_masks_a  =  batch[0].to(device) , batch[1].to(device)
                    batch_inputs_p, batch_masks_p  =  batch[2].to(device) , batch[3].to(device)
                    batch_inputs_n, batch_masks_n  =  batch[4].to(device) , batch[5].to(device)


                    embeddings, _ , _ = model(batch_inputs_a, batch_masks_a)
                    embeddings_p, _ , _ = model(batch_inputs_p, batch_masks_p)
                    embeddings_n, _ , _ = model(batch_inputs_n, batch_masks_n)               


            elif task_type.lower() in ["classification", "multilabel", "regression"]:

                if gene2vec_flag and contrastive_flag:

                    batch_inputs_a, batch_masks_a, gene2vec_a  =  batch[0].to(device) , batch[1].to(device), batch[2].to(device)
                    batch_inputs_p, batch_masks_p, gene2vec_p  =  batch[3].to(device) , batch[4].to(device), batch[5].to(device)
                    batch_inputs_n, batch_masks_n, gene2vec_n  =  batch[6].to(device) , batch[7].to(device), batch[8].to(device)
                    labels = batch[9].to(device)

                    embeddings, _ , preds = model(batch_inputs_a, batch_masks_a, gene2vec =gene2vec_a)
                    embeddings_p, _ , _ = model(batch_inputs_p, batch_masks_p, gene2vec = gene2vec_p)
                    embeddings_n, _ , _ = model(batch_inputs_n, batch_masks_n, gene2vec = gene2vec_n)               


                elif contrastive_flag:

                    batch_inputs_a, batch_masks_a  =  batch[0].to(device) , batch[1].to(device)
                    batch_inputs_p, batch_masks_p  =  batch[2].to(device) , batch[3].to(device)
                    batch_inputs_n, batch_masks_n  =  batch[4].to(device) , batch[5].to(device)
                    labels = batch[6].to(device)

                    embeddings, _ , preds = model(batch_inputs_a, batch_masks_a )
                    embeddings_p, _ , _ = model(batch_inputs_p, batch_masks_p )
                    embeddings_n, _ , _ = model(batch_inputs_n, batch_masks_n )

                elif gene2vec_flag:
                    batch_inputs, batch_masks, gene2vec_embeddings, labels =  batch[0].to(device) , batch[1].to(device), batch[2].to(device), batch[3].to(device)
                    embeddings, _ , preds = model(batch_inputs, batch_masks, gene2vec = gene2vec_embeddings)

                else:
                    batch_inputs, batch_masks , labels =  batch[0].to(device) , batch[1].to(device), batch[2].to(device)
                    embeddings, _ , preds = model(batch_inputs, batch_masks)
            
            
            if task_type == "regression":
            
                preds = preds.squeeze().float()
                labels = labels.squeeze().float()

                if contrastive_flag:
                    TLoss = get_triplet_loss(embeddings, embeddings_p, embeddings_n, margin=margin)
                    bert_loss = loss_fn(preds, labels)

                    loss = bert_loss + alpha*TLoss
                    total_tloss+= TLoss.item()

                else:
                    loss = loss_fn(preds, labels)

 

                total_preds.extend(preds.cpu().detach())
                total_labels.extend(labels.cpu().detach())        

        
            elif task_type == "classification":

                if contrastive_flag:
                    TLoss = get_triplet_loss(embeddings, embeddings_p, embeddings_n, margin=margin)
                    bert_loss = loss_fn(preds, labels)

                    loss = bert_loss + alpha*TLoss
                    total_tloss+= TLoss.item()

                else:
                    loss = loss_fn(preds, labels)

                total_preds.extend(preds.argmax(1).type(torch.int).to('cpu').numpy())
                total_labels.extend(labels.type(torch.int).to('cpu').numpy())

            
            elif task_type == "multilabel":

                preds = preds.to(torch.float32)
                labels = labels.to(torch.float32)


                loss = loss_fn(preds, labels)

                total_labels.extend(labels.cpu().type(torch.int).numpy().tolist())
                total_preds.extend((preds > 0.5).type(torch.int).cpu().numpy().tolist())

            elif task_type == "unsupervised":
            
                loss = loss_fn(embeddings, embeddings_p, embeddings_n)
                
                labels = batch[6]
                total_labels.extend(labels.type(torch.int).numpy().tolist())
            
            
            elif task_type =="interaction":
                embeddings_a = embeddings

                embeddings = torch.cat((embeddings, embeddings), dim=0)
                embeddings_interact = torch.cat((embeddings_p, embeddings_n), dim=0)

                preds = Fusion(embeddings, embeddings_interact)
                labels = torch.cat((torch.ones(embeddings_p.size(0),1),
                                    torch.zeros(embeddings_n.size(0),1)),
                                   dim=0).to(preds.device)


#                 loss = loss_fn(preds, labels)
                loss = loss_fn[0](preds, labels) + loss_fn[1](embeddings_a, embeddings_p, embeddings_n)

                total_labels.extend(labels.cpu().type(torch.int).numpy().tolist())
                total_preds.extend((preds > 0.5).type(torch.int).cpu().numpy().tolist())

            
            
            test_loss += loss.item()
            

            embeddings = torch.tensor(embeddings.cpu().detach().numpy())
            latents.append(embeddings)

    test_loss /= len(loader)
    latents = torch.cat(latents, dim=0)
    
    if contrastive_flag:
        print(f"Triplet loss:{total_tloss/len(loader)}")
       
    return model, test_loss, total_labels, total_preds, latents


def retrieve_triplets(tokens_go, indices):
    #input_ids, attention_mask
        
    # Define tensors
#     bank_tensor = torch.tensor([[10, 5], [20, 2], [30, 19], [40, 2], [50, 22], [60, 9]])
#     indices_tensor = torch.tensor([[2, 4, -1], [5, -1, -1]])

    indices_flat = indices.flatten()
    indices_flat = indices_flat[indices_flat != -1]
    
#     retrieved_elements = bank_tensor[indices_flat]
    
    
#     valid_mask = indices != -1
    
#     valid_indices = indices_tensor[valid_mask]

#     retrieved_elements = bank_tensor[valid_indices]

    
    return tokens_go["input_ids"][indices_flat], tokens_go["attention_mask"][indices_flat], (indices != -1).sum(dim=1)


def CL_Trainer(
    train_loader, val_loader, tokens_go, epochs =5, lr = 5e-5, pool = "cls",
    model_name= "bert-base-cased", device = "cuda", batch_size =100, go_batch_size =100,
    temperature = 0.1, save_model_path = None):
    
    loss_fn = NTXentLoss(temperature= temperature)
    
    model = FineTunedBERT(pool= pool,
                          task_type = "unsupervised",
                          n_labels = 1,
                          model_name = model_name,
                          device = device).to(device)  
    model = nn.DataParallel(model)
    
    optimizer = AdamW(model.parameters(), lr=lr)
    
    optimal_loss = float('inf')
    epoch_train_loss, epoch_val_loss =0 , 0
    for epoch in range(epochs):
        start_time = time.time()
        
        print(f"Epoch {epoch+1} of {epochs}")
        print("-------------------------------")
        
        print("Training ...") 
        
#         train_loss = 0
        
        batch_train_loss = 0
        model.train()
        for batch_idx, batch in tqdm(enumerate(train_loader)): 
            optimizer.zero_grad()
            
            batch_inputs_a, batch_masks_a  =  batch[0], batch[1]#.to(device) , batch[1].to(device)
            positive_ind, negative_ind  =  batch[2], batch[3]#.to(device) , batch[3].to(device)

#             gene_embeddings, _ , _ = model(batch_inputs_a.to(device), batch_masks_a.to(device))
            
            
            inputs_p, masks_p, triplet_expander = retrieve_triplets(tokens_go, positive_ind)
            inputs_n, masks_n, _ = retrieve_triplets(tokens_go, negative_ind)
            
#             print("inputs_p:", inputs_p.shape)
#             print("masks_p:", masks_p.shape)
#             print("inputs_n:", inputs_n.shape)
#             print("masks_n:", masks_n.shape)
#             print("triplet_expander:", triplet_expander.shape)
#             print("gene_embeddings:", gene_embeddings.shape)
# #             print("gene_embeddings after:", t.shape)
#             print("triplet_expander:", triplet_expander)
            
#             gene_embeddings_extended = anchor = gene_embeddings[torch.repeat_interleave(
#                 torch.arange(gene_embeddings.shape[0]),
#                 triplet_expander)]
            
#             triplet_dataset = TensorDataset(inputs_p, masks_p, inputs_n, masks_n, triplet_expander)
            triplet_dataset = TensorDataset(inputs_p, masks_p, inputs_n, masks_n)#, gene_embeddings_extended)
            
            triplet_loader = DataLoader(triplet_dataset, batch_size=go_batch_size, shuffle=False)
            
#             tloss = 0
            mini_batch_train_loss = 0
            
            
            for triplet_batch in triplet_loader:
                batch_inputs_p, batch_masks_p = triplet_batch[0].to(device), triplet_batch[1].to(device)
                batch_inputs_n, batch_masks_n = triplet_batch[2].to(device), triplet_batch[3].to(device)
#                 batch_expander = triplet_loader[4].to(device)
#                 anchor = triplet_loader[4].to(device)
                
                #get embeddings
                #repeat the anchor embeddings to have the same size as positive and negative
#                 anchor = gene_embeddings[torch.repeat_interleave(torch.arange(gene_embeddings.shape[0]), batch_expander)]

                #triplet_batch = 2 
                #gene_embeddings -> 3*768
                #full_positive -> 7*768 
                #full_negative -> 7*768
                # [0,1,1,1,2,2,2]
            
                #anchor = gene_embeddings[0,1]

                #[1,200,200]
                
#                 anchor = gene_embeddings #triplet_batch[4].to(device)                
                anchor, _ , _ = model(batch_inputs_a.to(device), batch_masks_a.to(device))
                positive, _ , _ = model(batch_inputs_p, batch_masks_p)
                negative, _ , _ = model(batch_inputs_n, batch_masks_n)
                
                triplet_embeddings = torch.cat([anchor, positive, negative], dim=0).to(device)
                
#                 print("gene_embeddings:", gene_embeddings.shape)
#                 print(triplet_expander)
                
#                 print("anchor:", anchor.shape)
#                 print("positive:", positive.shape)
#                 print("negative:", negative.shape)
#                 print("triplet_embeddings:", triplet_embeddings.shape)
                
    
                N = positive.shape[0]
                a_ind = torch.arange(0, 1).repeat(N).to(device)
                p_ind = torch.arange(1,N+1).to(device)
                n_ind = torch.arange(N+1, 2*N +1).to(device)
    
    
#                 a_ind = torch.tensor(range(0, anchor.shape[0])).to(device)
#                 p_ind = torch.tensor(range(anchor.shape[0], anchor.shape[0] + positive.shape[0])).to(device)
#                 n_ind = torch.tensor(range(anchor.shape[0] + positive.shape[0],
#                                            anchor.shape[0] + positive.shape[0] + negative.shape[0])).to(device)
                tloss = loss_fn(triplet_embeddings, indices_tuple=(a_ind, p_ind, n_ind))
                #calculate loss
                mini_batch_train_loss += tloss.item()
                
        
                #tloss += mini_batch_train_loss.item()
            
                #Mini-Batch Gradient Descent
                model.zero_grad()
                tloss.backward()
                optimizer.step()

            batch_train_loss += (mini_batch_train_loss/len(triplet_loader))  
#             batch_train_loss += (tloss/len(triplet_loader))  
            
            #Save a checkpoint every 100 gene
            if batch_idx%1000 == 0:
                    torch.save({
                        'epoch': epoch,
                        'batch_idx': batch_idx,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': batch_train_loss },
                        f'{save_model_path}model_checkpoints/checkpoint_epoch_{epoch}_batch_{batch_idx}.pth')
            
        epoch_train_loss = batch_train_loss / len(train_loader) 
        
        
        print("Validation ...") 
        #do something
        

        batch_val_loss =0
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(val_loader)): 

                batch_inputs_a, batch_masks_a  =  batch[0],batch[1] #.to(device) , batch[1].to(device)
                positive_ind, negative_ind  =  batch[2],  batch[3] #.to(device) , batch[3].to(device)

#                 gene_embeddings, _ , _ = model(batch_inputs_a.to(device), batch_masks_a.to(device))


                inputs_p, masks_p, triplet_expander = retrieve_triplets(tokens_go, positive_ind)
                inputs_n, masks_n, _ = retrieve_triplets(tokens_go, negative_ind)
                
                
#                 gene_embeddings_extended = anchor = gene_embeddings[torch.repeat_interleave(
#                     torch.arange(gene_embeddings.shape[0]),
#                     triplet_expander)]

#                 triplet_dataset = TensorDataset(inputs_p, masks_p, inputs_n, masks_n, triplet_expander)
                triplet_dataset = TensorDataset(inputs_p, masks_p, inputs_n, masks_n)#, gene_embeddings_extended)
                triplet_loader = DataLoader(triplet_dataset, batch_size=go_batch_size, shuffle=False)

#                 vloss = 0
                mini_batch_val_loss =0
    
                for triplet_batch in triplet_loader:
                    batch_inputs_p, batch_masks_p = triplet_batch[0].to(device), triplet_batch[1].to(device)
                    batch_inputs_n, batch_masks_n = triplet_batch[2].to(device), triplet_batch[3].to(device)
#                     batch_expander = triplet_batch[4].to(device)

                    #get embeddings
                    #repeat the anchor embeddings to have the same size as positive and negative
#                     anchor = gene_embeddings[torch.repeat_interleave(torch.arange(gene_embeddings.shape[0]), batch_expander)]

#                     anchor = triplet_batch[4].to(device)
                    anchor, _ , _ = model(batch_inputs_a.to(device), batch_masks_a.to(device))
                    positive, _ , _ = model(batch_inputs_p, batch_masks_p)
                    negative, _ , _ = model(batch_inputs_n, batch_masks_n)


                    triplet_embeddings = torch.cat([anchor, positive, negative], dim=0).to(device)

#                     print("gene_embeddings:", gene_embeddings.shape)
#                     print(triplet_expander)

#                     print("anchor:", anchor.shape)
#                     print("positive:", positive.shape)
#                     print("negative:", negative.shape)
#                     print("triplet_embeddings:", triplet_embeddings.shape)
                    

#                     a_ind = torch.tensor(range(0, anchor.shape[0])).to(device)
#                     p_ind = torch.tensor(range(anchor.shape[0], anchor.shape[0] + positive.shape[0])).to(device)
#                     n_ind = torch.tensor(range(anchor.shape[0] + positive.shape[0],
#                                                anchor.shape[0] + positive.shape[0] + negative.shape[0])).to(device)

                    N = positive.shape[0]
                    a_ind = torch.arange(0, 1).repeat(N).to(device)
                    p_ind = torch.arange(1,N+1).to(device)
                    n_ind = torch.arange(N+1, 2*N +1).to(device)
                    #calculate loss
                    vloss = loss_fn(triplet_embeddings, indices_tuple=(a_ind, p_ind, n_ind))
                    mini_batch_val_loss += vloss.item()

#                     vloss += mini_batch_val_loss.item()
                    
                batch_val_loss += (mini_batch_val_loss/len(triplet_loader))
            epoch_val_loss = batch_val_loss / len(val_loader)
            
        #Save BestModel as pickle
        if optimal_loss > epoch_val_loss:
            optimal_loss = epoch_val_loss
            print(f"Best Model was Found at epoch {epoch}.")
            
            with open(save_model_path+f"pickles/BestModel.pth", 'wb') as file:
                torch.save(model, file)
        else:
            with open(save_model_path+f"pickles/Model{str(epoch)}.pth", 'wb') as file:
                torch.save(model, file)
        
        
        print(f'\tET: {(time.time() - start_time):.3f} Seconds')
        print(f'Train Loss: {epoch_train_loss:.5f}')
        print(f'Val Loss: {epoch_val_loss:.5f}')


def trainer(epochs, train_loader=None, val_loader=None, test_loader=None,
            lr = 5e-5, pool = "cls", max_length= 100, drop_rate =0.1,
            gene2vec_flag = False, gene2vec_hidden = 200, task_type = "classification",
            n_labels = 3 ,model_name= "bert-base-cased",
            class_map = None, device = "cuda", save_model_path=None,
            contrastive_flag = False, margin = 0.5, alpha=0.5, load_model =None):
    
    
    """
        gene2vec_flag: if True then, the embeddings of gene2vec will be concat to GeneLLM embeddings.

        model_name: "xlnet-base-cased",
                    "microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract",
                    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                    "dmis-lab/biobert-base-cased-v1.1",
                    "bert-base-cased",
                    "bert-base-uncased"


        task_type = classification or regression
    """

    if task_type == "classification":
        print ("##########################################################")
        print (f"Currently running Single-Label Classification with {n_labels = }.")
        print ("##########################################################\n")

        
        history = {
            "Train":{"Accuracy":[], "F1":[], "Precision":[], "Recall":[]},
            "Val":{"Accuracy":[], "F1":[], "Precision":[], "Recall":[]},
            "Test":{"Accuracy":[], "F1":[], "Precision":[], "Recall":[]}}
        
        loss_fn = nn.CrossEntropyLoss()
        
    elif task_type == "interaction":
        
        print ("##########################################################")
        print (f"Currently running Gene-Interaction Classification with {n_labels = }.")
        print ("##########################################################\n")
        
        history = {
            "Train":{"Accuracy":[], "F1":[], "Precision":[], "Recall":[]},
            "Val":{"Accuracy":[], "F1":[], "Precision":[], "Recall":[]},
            "Test":{"Accuracy":[], "F1":[], "Precision":[], "Recall":[]}}
        
        loss_fn = [nn.BCELoss(), TripletMarginLoss(p=2, margin = margin)]

 
    elif task_type == "multilabel":
        print ("##########################################################")
        print (f"Currently running Multi-label Classification with {n_labels = }.")
        print ("##########################################################\n")

        history = {
            "Train":{"Accuracy":[], "F1":[], "Precision":[], "Recall":[]},
            "Val":{"Accuracy":[], "F1":[], "Precision":[], "Recall":[]},
            "Test":{"Accuracy":[], "F1":[], "Precision":[], "Recall":[]}}
                    
        loss_fn = MultiLabelFocalLoss()
#         loss_fn = nn.BCELoss()    
        
    elif task_type == "unsupervised":
        print ("#########################################")
        print (f"Currently running Unsupervised Learning.")
        print ("#########################################\n")

        history = {
            "Train":{"Accuracy":[], "F1":[], "Precision":[], "Recall":[]},
            "Val":{"Accuracy":[], "F1":[], "Precision":[], "Recall":[]},
            "Test":{"Accuracy":[], "F1":[], "Precision":[], "Recall":[]}}
            
        loss_fn = TripletMarginLoss(p=2, margin = margin)
        
    elif task_type == "regression":
        
        print ("\n###############################")
        print (f"Currently running Regression with {n_labels = }.")
        print ("###############################\n")
        
        history = {
            "Train":{"Correlation":[], "MAE":[], "MSE":[], "R2":[]},
            "Val":{"Correlation":[], "MAE":[], "MSE":[], "R2":[]},
            "Test":{"Correlation":[], "MAE":[], "MSE":[], "R2":[]}}
        
        loss_fn = nn.MSELoss()
            
    else:
        raise ValueError(f"task type error: {task_type}")
    

    if not load_model:
    
        model = FineTunedBERT(pool= pool, task_type = task_type, n_labels = n_labels,
                              drop_rate = drop_rate, model_name = model_name,
                              gene2vec_flag= gene2vec_flag,
                              gene2vec_hidden = gene2vec_hidden, device = device).to(device)            
    else:
        
        bert_state_dict = load_model.bert.state_dict()
        model_name = load_model.model_name
        pool = load_model.pool
        
        model = FineTunedBERT(pool= pool, task_type = task_type, n_labels = n_labels,
                              drop_rate = drop_rate, model_name = model_name,
                              gene2vec_flag= gene2vec_flag,
                              gene2vec_hidden = gene2vec_hidden,
                              device = device, bert_state_dict = bert_state_dict).to(device)            

    
    # model = nn.DataParallel(model, device_ids = [1,0])
    model = nn.DataParallel(model)
#     params_to_optimize = [
#         {'params': model.parameters()},  # model parameters
#         {'params': [loss_fn.alpha, loss_fn.gamma]},
#     ]
    
    
    if task_type == "interaction":
        
        global Fusion 
        Fusion = FusionModel(input_size=768,hidden_size=500,n_labels=n_labels).to(device)
        
        optimizer = AdamW(list(model.parameters()) + list(Fusion.parameters()), lr=lr)
        
    else:
        optimizer = AdamW(model.parameters(), lr=lr)
                
        
    
    best_pred = None
    best_model = model.state_dict()
    optimal_metric = -1
    optimal_loss = float('inf')


    for epoch in range(epochs):        
        start_time = time.time()
        
        print(f"Epoch {epoch+1} of {epochs}")
        print("-------------------------------")
        
        print("Training ...")
        
        model, train_loss, labels_train, pred_train, latents = train(train_loader, model, loss_fn, optimizer,
                                                                     task_type = task_type,
                                                                     gene2vec_flag = gene2vec_flag, device = device,
                                                                     contrastive_flag = contrastive_flag,
                                                                     margin = margin, alpha=alpha)

#         plot_latent(latents, labels_train,  epoch, validation_type="train")
      
        if val_loader is not None:
            print("Validation ...")
            model, val_loss, labels_val, pred_val  = validation (val_loader, model, loss_fn,
                                                                 task_type = task_type,
                                                                 gene2vec_flag = gene2vec_flag,device = device,
                                                                 contrastive_flag = contrastive_flag,
                                                                 margin = margin, alpha=alpha)
        
        print("Testing ...")
        model, test_loss, labels_test, pred_test, _ = test (test_loader, model, loss_fn,
                                                            task_type = task_type,
                                                            gene2vec_flag = gene2vec_flag, device = device,
                                                            contrastive_flag = contrastive_flag,
                                                            margin = margin, alpha=alpha)

        metrics_train  = get_metrics(labels_train , pred_train, history,
                                      val_type = "Train", task_type = task_type)
        
        if val_loader is not None: metrics_val = get_metrics(labels_val , pred_val, history,
                                                             val_type = "Val",task_type = task_type)

        metrics_test = get_metrics(labels_test , pred_test, history, val_type = "Test",
                                   task_type = task_type)

        if task_type in ["multilabel", "classification", "interaction"]:
            acc_train, f1_train, prec_train, rec_train = metrics_train
            acc_test, f1_test, prec_test, rec_test = metrics_test
            
            if val_loader is not None:  
                acc_val, f1_val, prec_val, rec_val = metrics_val
                
                if optimal_metric < acc_val:
                    optimal_metric = acc_val
                    best_pred = pred_test

                    if save_model_path is not None:
                        with open(save_model_path+f"best_model.pth", 'wb') as file:
                            torch.save(model, file)
                            #pickle.dump(model, file)
                
                
            else:
                if optimal_metric < acc_test:
                    optimal_metric = acc_test
                    best_pred = pred_test

                    if save_model_path is not None:
                        with open(save_model_path+f"best_model.pth", 'wb') as file:
                            torch.save(model, file)
            

            
            print(f'\tET: {(time.time() - start_time):.3f} Seconds')
            print(f'Train Loss: {train_loss:.5f}, Accuracy: {acc_train:.3f}, F1: {f1_train:.3f}, Precision: {prec_train:.3f}, Recall: {rec_train:.3f}')
            if val_loader is not None: 
                print(f'Val Loss: {val_loss:.5f}, Accuracy: {acc_val:.3f}, F1: {f1_val:.3f}, Precision: {prec_val:.3f}, Recall: {rec_val:.3f}')
            print(f'Test Loss: {test_loss:.5f}, Accuracy: {acc_test:.3f}, F1: {f1_test:.3f}, Precision: {prec_test:.3f}, Recall: {rec_test:.3f}')
    
    
        elif task_type == "unsupervised":
            
            print(f'\tET: {(time.time() - start_time):.3f} Seconds')
            print(f'Train Loss: {train_loss:.5f}')
            if val_loader is not None:
                print(f'Val Loss: {val_loss:.5f}')
            print(f'Test Loss: {test_loss:.5f}')

            
            if optimal_loss > train_loss:
                optimal_loss = train_loss
                best_model = model.state_dict()
                
        elif task_type == "regression":
            
            train_corr, train_mae, train_mse, train_r2 = metrics_train
            test_corr, test_mae, test_mse, test_r2 = metrics_test
            
            if val_loader is not None:
                val_corr, val_mae, val_mse, val_r2 = metrics_val
                
                if optimal_metric < val_corr:
                    optimal_metric = val_corr
                    best_pred = pred_test
                    
                    if save_model_path is not None:
                        with open(save_model_path+f"best_model.pth", 'wb') as file:
                            torch.save(model, file)

            else:
                if optimal_metric < test_corr:
                    optimal_metric = test_corr
                    best_pred = pred_test
                    
                    if save_model_path is not None:
                        with open(save_model_path+f"best_model.pth", 'wb') as file:
                            torch.save(model, file)

            
            print(f'\tET: {(time.time() - start_time):.3f} Seconds')
            print(f'\tTrain Loss: {train_loss:.5f}, corrcoef: {train_corr:.3f}, MAE: {train_mae:.3f}, MSE: {train_mse:.3f}, R2: {train_r2:.3f}')
            if val_loader is not None:
                print(f'\tVal Loss: {val_loss:.5f}, corrcoef: {val_corr:.3f}, MAE: {val_mae:.3f}, MSE: {val_mse:.3f}, R2: {val_r2:.3f}')
            print(f'\tTest Loss: {test_loss:.5f}, corrcoef: {test_corr:.3f}, MAE: {test_mae:.3f}, MSE: {test_mse:.3f}, R2: {test_r2:.3f}')

        else:
            raise ValueError(f"Incorrect task_type: {task_type}")
    
    
#         if save_model_path is not None:
            
#             with open(save_model_path+f"model_{str(epoch)}.pkl", 'wb') as file:
#                 pickle.dump(model, file)

#             torch.save(model, save_model_path+f"state_dict_{str(epoch)}.pth")
    
    return model, history, labels_test, best_pred