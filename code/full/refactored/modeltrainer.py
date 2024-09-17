###########################################################
# Project: GeneLLM
# File: model.py
# License: MIT
# Code Authors: Jararaweh A., Macualay O.S, Arredondo D., & 
#               Virupakshappa K.
###########################################################
from model import FineTunedBERT , MultiLabelFocalLoss
from utils import train, validation, test, get_metrics, plot_latent,plot_metrics
from transformers import AdamW
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import torch
import pandas as pd
import copy
import os
from data_processor import loading_data, process_data
import shutil
import gc
from transformers import BertTokenizerFast
from transformers import XLNetTokenizer
from torch.utils.data import DataLoader, TensorDataset


def trainer(epochs, genes, train_loader, val_loader, test_loader, k_fold,
                lr = 1e-5, pool = "cls", max_length= 100, batch_size =100, drop_rate =0.1,
                gene2vec_flag = True, gene2vec_hidden = 200, device = "cuda",
                task_type = "classification", n_labels = 3 , model_name = "microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract", task_name = 'Subcellular_location', model_type ='Pubmed_large'):
    
    
    """
        gene2vec_flag: if True then, the embeddings of gene2vec will be concat to GeneLLM embeddings.

        model_type: "xlnet-base-cased",
                    "microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract",
                    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                    "dmis-lab/biobert-base-cased-v1.1",
                    "bert-base-cased",
                    "bert-base-uncased"


        task_type = classification or regression
    
    
    """
    
    

    #subcell : {0:'Cytoplasm', 1:'Nucleus', 2:'Cell membrane'}
    #Sol: {0:'Membrane', 1:'Soluble'}
    #cons: {0:}
    global class_map
    if task_type == "classification":
        history = {
            "Train":{"Accuracy":[], "F1":[], "Precision":[], "Recall":[], "ROC AUC":[], "AUPR":[]},
            "Val":{"Accuracy":[], "F1":[], "Precision":[], "Recall":[], "ROC AUC":[], "AUPR":[]},
            "Test":{"Accuracy":[], "F1":[], "Precision":[], "Recall":[], "ROC AUC":[], "AUPR":[]}}
        
        unique_values = genes.iloc[:, 3].unique()
        
        class_map = {i: value for i, value in enumerate(unique_values)}
        print ("\n#############################")
        print (f"Currently running {task_name}.")
        print ("#############################\n")


    elif task_type == "multilabel":
        history = {
            "Train":{"Accuracy":[], "F1":[], "Precision":[], "Recall":[], "ROC AUC":[], "AUPR":[]},
            "Val":{"Accuracy":[], "F1":[], "Precision":[], "Recall":[], "ROC AUC":[], "AUPR":[]},
            "Test":{"Accuracy":[], "F1":[], "Precision":[], "Recall":[], "ROC AUC":[], "AUPR":[]}}
        
            
        print ("\n#############################")
        print ("Currently running {task_name}.")
        print ("#############################\n")          



    elif task_type == "regression":
        history = {
            "Train":{"S_Correlation":[], "P_Correlation":[], "MAE":[], "MSE":[], "R2":[]}, 
            "Val":{"S_Correlation":[], "P_Correlation":[], "MAE":[], "MSE":[], "R2":[]}, 
            "Test":{"S_Correlation":[], "P_Correlation":[], "MAE":[], "MSE":[], "R2":[]}}
        
        if n_labels == 1:
            
            class_map = None
            print ("\n###############################")
            print (f"Currently running {task_name}.")
            print ("###############################\n")
        
    

    if task_type == "regression":
        loss_fn = nn.MSELoss()
        
    elif task_type == "classification":
        loss_fn = nn.CrossEntropyLoss()
        
    elif task_type == "multilabel":
        loss_fn = MultiLabelFocalLoss()
        # loss_fn = nn.BCELoss()
    else:
        raise ValueError(f"task type error: {task_type}")



    model = FineTunedBERT(pool= pool, model_name = model_name, bert_state_dict=None, task_type = task_type, n_labels = n_labels,
                          drop_rate = drop_rate, 
                          gene2vec_flag= gene2vec_flag,
                          gene2vec_hidden = gene2vec_hidden, device ="cuda").to(device)

    
    optimizer = AdamW(model.parameters(), lr=lr)
    

    best_pred = None
    optimal_acc = -1
    global best_model_state
    best_model_state = None
    best_val_metric = 0.0 if task_type in ["classification", "multilabel"] else float('-inf')




    
    
    for epoch in range(epochs):
        start_time = time.time()
        
        print(f"Epoch {epoch+1} of {epochs}")
        print("-------------------------------")
        
        print("Training ...")
        
        model, train_loss, labels_train, pred_train, latents, probability_train = train(train_loader, model, loss_fn, optimizer,
                                                                     task_type = task_type,
                                                                     gene2vec_flag = gene2vec_flag,
                                                                     device = device)
        print(latents.size())
        plot_latent(latents, labels_train,  epoch, class_map, task_name, validation_type="train", model_type=model_type)
        
        
        
        print("Validation ...")
        model, val_loss, labels_val, pred_val, probability_val  = validation (val_loader, model, loss_fn,
                                                             task_type = task_type,
                                                             gene2vec_flag = gene2vec_flag,
                                                             device = device)
        
        # model, test_loss, labels_test, pred_test,_, probability_test = test(test_loader, model, loss_fn,
        #                                                 task_type = task_type,
        #                                                 gene2vec_flag = gene2vec_flag,
        #                                                 device = device)
        

        

        metrics_train  = get_metrics(labels_train , pred_train, probability_train, history,
                                      val_type = "Train", task_type = task_type)
        
        metrics_val = get_metrics(labels_val , pred_val, probability_val, history,
                                  val_type = "Val",task_type = task_type)
        

        
    
        # metrics_test = get_metrics(labels_test , pred_test, probability_test,  history, val_type = "Test",
        #                            task_type = task_type)
    
        
        



        if task_type == "classification" or task_type == "multilabel":
            acc_train, f1_train, prec_train, rec_train, roc_auc_train, aupr_train = metrics_train
            acc_val, f1_val, prec_val, rec_val, roc_auc_val, aupr_val = metrics_val
            # acc_test, f1_test, prec_test, rec_test = metrics_test

            

            print(f'\tET: {round(time.time() - start_time,2)} Seconds')
            print(f'Train Loss: {round(train_loss,4)}, Accuracy: {round(acc_train,4)}, F1: {round(f1_train,4)}, Precision: {round(prec_train,4)}, Recall: {round(rec_train,4)}, ROC AUC: {round(roc_auc_train,4)}, AUPR: {round(aupr_train,4)}')
            print(f'Val Loss: {round(val_loss,4)}, Accuracy: {round(acc_val,4)}, F1: {round(f1_val,4)}, Precision: {round(prec_val,4)}, Recall: {round(rec_val,4)}, ROC AUC: {round(roc_auc_val,4)}, AUPR: {round(aupr_val,4)}')
            # print(f'Test Loss: {round(test_loss,4)}, Accuracy: {round(acc_test,4)}, F1: {round(f1_test,4)}, Precision: {round(prec_test,4)}, Recall: {round(rec_test,4)}')
    
            with open(f'{data_path}/{task_name}/{model_type}/metrics_{task_name}_fold_{k_fold}.csv', 'a') as f:
                if epoch == 0:
                    f.write("epoch,fold,train_loss,train_acc,train_f1,train_prec,train_rec,train_roc_auc,train_aupr,val_loss,val_acc,val_f1,val_prec,val_rec,val_roc_auc, val_aupr\n")
                f.write(f"{epoch+1},{k_fold},{train_loss},{acc_train},{f1_train},{prec_train},{rec_train},{roc_auc_train},{aupr_train},{val_loss},{acc_val},{f1_val},{prec_val},{rec_val},{roc_auc_val}, {aupr_val}\n")

        else:

            s_train_corr, p_train_corr, mae_train, mse_train, r2_train = metrics_train
            s_val_corr, p_val_corr, mae_val, mse_val, r2_val = metrics_val
            # s_test_corr, p_test_corr, mae_test, mse_test, r2_test = metrics_test
            

            
            print(f'\tET: {round(time.time() - start_time,2)} Seconds')
            print(f'\tTrain Loss: {round(train_loss,4)}, s_corrcoef: {round(s_train_corr,4)}, p_corrcoef: {round(p_train_corr,4)}, MAE: {round(mae_train,4)}, MSE: {round(mse_train,4)}, R2: {round(r2_train,4)}')
            print(f'\tVal Loss: {round(val_loss,4)}, s_corrcoef: {round(s_val_corr,4)}, p_corrcoef: {round(p_val_corr,4)}, MAE: {round(mae_val,4)}, MSE: {round(mse_val,4)}, R2: {round(r2_val,4)}')
            # print(f'\tTest Loss: {round(test_loss,4)}, corrcoef: {round(test_corr,4)}, MAE: {round(mae_test,4)}, MSE: {round(mse_test,4)}, R2: {round(r2_test,4)}')
            
            with open(f'{data_path}/{task_name}/{model_type}/metrics_{task_name}_fold_{k_fold}.csv', 'a') as f:
                if epoch == 0:
                    f.write(f'epoch,fold,train_loss,s_train_corr,p_train_corr,train_mae,train_mse,train_r2,val_loss,s_val_corr,p_val_corr,val_mae,val_mse,val_r2\n')
                f.write(f"{epoch+1},{k_fold},{train_loss},{s_train_corr},{p_train_corr},{mae_train},{mse_train},{r2_train},{val_loss},{s_val_corr},{p_val_corr},{mae_val},{mse_val},{r2_val}\n")

        current_val_metric = s_val_corr if task_type == "regression" else roc_auc_val  
        if current_val_metric >= best_val_metric:
            best_val_metric = current_val_metric
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, f'{data_path}/{task_name}/{model_type}/best_model_{task_name}_{k_fold}.pth')
            global best_epoch_num
            best_epoch_num = epoch + 1
            print(f'best_epoch_num: {best_epoch_num}')



    plot_metrics(data_path, task_name, model_type, best_epoch_num, task_type, k_fold)

    # Evaluate on test set after all epochs

    best_model_path = f'{data_path}/{task_name}/{model_type}/best_model_{task_name}_{k_fold}.pth'
    model.load_state_dict(torch.load(best_model_path))
    model, test_loss, labels_test, pred_test,_, probability_test = test(test_loader, model, loss_fn,
                                                        task_type = task_type,
                                                        gene2vec_flag = gene2vec_flag,
                                                        device = device)
    
    metrics_test = get_metrics(labels_test , pred_test, probability_test,  history, val_type = "Test",
                                   task_type = task_type)
    
    if task_type == "classification" or task_type == "multilabel":
        acc_test, f1_test, prec_test, rec_test, roc_auc_test, aupr_test = metrics_test
        print(f'Test Loss: {round(test_loss,4)}, Accuracy: {round(acc_test,4)}, F1: {round(f1_test,4)}, Precision: {round(prec_test,4)}, Recall: {round(rec_test,4)}, ROC AUC: {round(roc_auc_test,4)}, AUPR: {round(aupr_test,4)}')
        with open(f'{data_path}/{task_name}/{model_type}/test_metrics_{task_name}.csv', 'a') as f:
            f.write(f"fold: {k_fold},best_epoch:{best_epoch_num},Accuracy: {acc_test},F1: {f1_test},Precision: {prec_test},Recall: {rec_test},ROC AUC: {roc_auc_test}, AUPR: {aupr_test}\n")

        return acc_test, f1_test, prec_test, rec_test, roc_auc_test

    else:
        s_test_corr, p_test_corr, mae_test, mse_test, r2_test = metrics_test
        print(f'\tTest Loss: {round(test_loss,4)}, s_corrcoef: {round(s_test_corr,4)}, p_corrcoef: {round(p_test_corr,4)}, MAE: {round(mae_test,4)}, MSE: {round(mse_test,4)}, R2: {round(r2_test,4)}')
        with open(f'{data_path}/{task_name}/{model_type}/test_metrics_{task_name}.csv', 'a') as f:
            f.write(f"fold: {k_fold},best_epoch:{best_epoch_num},s_corrcoef: {s_test_corr},p_corrcoef: {p_test_corr},MAE: {mae_test},MSE: {mse_test},R2: {r2_test}\n")
        
        bias_analysis(data_path, task_name, model_type, best_model_path, task_type, model_name, max_length, batch_size, gene2vec_flag, device, history, model, genes, loss_fn, k_fold)

        return s_test_corr, p_test_corr, mae_test, mse_test, r2_test
def bias_analysis(data_path, task_name, model_type, best_model_path, task_type, model_name, max_length, batch_size, gene2vec_flag, device, history, model, genes, loss_fn, k_fold):
    """
    @Ala and @Macualy provide description
    """
    print("Running bias analysis ...")
    
    genes['Word_Count'] = genes['Summary'].apply(lambda summary: len(summary.split()))
    median_count = genes['Word_Count'].median()
    min_count = genes['Word_Count'].min()
    max_count = genes['Word_Count'].max()
    print(f"Median word count: {median_count}")
    print(f"Minimum word count: {min_count}")
    print(f"Maximum word count: {max_count}")
    plt.figure(figsize=(10, 6))
    plt.hist(genes['Word_Count'], bins=30, color='blue', alpha=0.7)
    plt.title('Histogram of Word Counts')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    df_test_genes = pd.read_csv(f'{data_path}/{task_name}/{model_type}/folds/test_gene_names_fold_{k_fold}.csv')
    #merge test genes with genes df
    test_genes = df_test_genes['g_name'].tolist()
    genes = genes[genes['Gene name'].isin(test_genes)]
    genes = genes.reset_index(drop=True)
    

    threshold = [i for i in range(30, 80, 10)]

    for i in threshold:
        genes_df = genes.copy()
        genes_df['Count'] = genes_df['Word_Count'].apply(lambda word_count: 'low count' if word_count < i else 'high count' if word_count > 112 else 'medium count')
        #subset genes with low word count
        genes_low = genes_df[genes_df['Count'] == 'low count']
        #subset genes with high word count
        genes_high = genes_df[genes_df['Count'] == 'high count']

        #get the number of genes with low word count
        num_genes_low = genes_low.shape[0]
        print(f"Number of genes with low word count: {num_genes_low}")
        #get the number of genes with high word count
        num_genes_high = genes_high.shape[0]
        print(f"Number of genes with high word count: {num_genes_high}")

        count_df = [genes_low, genes_high]
        count_type = ['low', 'high']

        for texts, count in zip(count_df, count_type):
            print(f"Currently running {count} word count")


            if "xlnet" in model_name:
                tokenizer = XLNetTokenizer.from_pretrained(model_name)

            else:    
                tokenizer = BertTokenizerFast.from_pretrained(model_name)


            sentences, labels = texts["Summary"].tolist() , texts["Label"].tolist()
            g_index, g_name = texts.index.tolist() , texts["Gene name"].tolist()
            
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

            print("Adding Gene2Vec data ...")
            if gene2vec_flag:
                
                # unirep_df = pd.read_csv('/mnt/data/macaulay/datas/training_gene_embeddings.csv')
                # Gene2Vec = unirep_df.set_index('Gene').T.to_dict('list')
                            
                # tokens_df = tokens_df[tokens_df['g_name'].isin(set(Gene2Vec.keys()) & set(tokens_df["g_name"]))]
                
                # tokens_df["gene2vec"] = tokens_df["g_name"].apply(lambda name: 
                #                                                     Gene2Vec[name])# if name in Gene2Vec.keys() else None )

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

            test_tokens = tokens_df[tokens_df['g_name'].isin(test_genes)] # test genes is a list

            test_tokens = test_tokens.reset_index(drop=True)

            if gene2vec_flag:

                test_dataset = TensorDataset(torch.tensor(test_tokens["input_ids"].tolist()),
                                        torch.tensor(test_tokens["attention_mask"].tolist()),
                                        torch.tensor(test_tokens["gene2vec"]),
                                        torch.tensor(test_tokens["labels"]),
                                        torch.tensor(test_tokens["g_index"]))


            else:

                test_dataset = TensorDataset(torch.tensor(test_tokens["input_ids"].tolist()),
                                        torch.tensor(test_tokens["attention_mask"].tolist()),
                                        torch.tensor(test_tokens["labels"]),
                                        torch.tensor(test_tokens["g_index"]))


            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            
            model.load_state_dict(torch.load(best_model_path))
            model, test_loss, labels_test, pred_test,_, probability_test = test(test_loader, model, loss_fn,
                                                                task_type = task_type,
                                                                gene2vec_flag = gene2vec_flag,
                                                                device = device)


            metrics_test = get_metrics(labels_test , pred_test, probability_test, history, val_type = "Test",
                                        task_type = task_type)
            
            if task_type == "classification" or task_type == "multilabel":
                acc_test, f1_test, prec_test, rec_test, roc_auc_test = metrics_test
                print(f'BIAS==== word_threshold: {i}, count_type: {count}, Accuracy: {round(acc_test,4)}, F1: {round(f1_test,4)}, Precision: {round(prec_test,4)}, Recall: {round(rec_test,4)}, ROC AUC: {round(roc_auc_test,4)}')
                with open(f'{data_path}/{task_name}/{model_type}/BIAS_test_metrics_{task_name}_{gene2vec_flag}_{k_fold}.csv', 'a') as f:
                    f.write(f'gene2vec_flag: {gene2vec_flag}\n')
                    f.write(f"word_threshold: {i}, count_type: {count},Accuracy: {acc_test},F1: {f1_test},Precision: {prec_test},Recall: {rec_test},ROC AUC: {roc_auc_test}\n")

            else:
                s_test_corr, p_test_corr, mae_test, mse_test, r2_test = metrics_test
                print(f'BIAS ==== word_threshold: {i}, count_type: {count}, s_corrcoef: {round(s_test_corr,4)}, p_corrcoef: {round(p_test_corr,4)}, MAE: {round(mae_test,4)}, MSE: {round(mse_test,4)}, R2: {round(r2_test,4)}')
                with open(f'{data_path}/{task_name}/{model_type}/BIAS_test_metrics_{task_name}_{gene2vec_flag}_{k_fold}.csv', 'a') as f:
                    f.write(f'gene2vec_flag: {gene2vec_flag}\n')
                    f.write(f"word_threshold: {i}, count_type: {count},s_corrcoef: {round(s_test_corr,4)}, p_corrcoef: {round(p_test_corr,4)},MAE: {mae_test},MSE: {mse_test},R2: {r2_test}\n")

    
    model_list = [#"xlnet-base-cased",
        # "microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract",
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        #"dmis-lab/biobert-base-cased-v1.1",
        # "bert-base-cased",
        # "bert-base-uncased"
        ]
    
    model_nicks = [#'xlnet',
                    # 'Pubmed_large',
                    'Pubmed_base',
                    #'Biobert_base',
                    # 'Bertbase_cased',
                    # 'Bertbase_uncased'
                    ]
    # Path to save all output
    #global data_path
    #data_path = "/data/macaulay/GeneLLM_MACAULAY/OUTPUTS1"

    gene2vec_flag = False
    if os.path.exists(f"{data_path}/{task_name}"):
        shutil.rmtree(f"{data_path}/{task_name}")
    os.makedirs(f"{data_path}/{task_name}", exist_ok=True)



    for model_name, model_type in zip(model_list, model_nicks):
        
        print(f'model name: {model_name}')
        print(f'model type: {model_type}')


        #########################################----------Prepare Data----------------######################################################################

        global gene_loaded_data
        gene_loaded_data, n_labels = loading_data(input_data_path, task_type)
        print(f'Number of {task_name} labels: {n_labels}')
        print(f'Bert Variant: {model_type}')
        display(gene_loaded_data)
        gene_loaded_data.to_csv(f'gene_loaded_data_{task_name}.csv', index=False)
        # time.sleep(500000)

        #########################################----------Create necessary directories----------------######################################################################


        os.makedirs(f"{data_path}/{task_name}/{model_type}", exist_ok=True)
        os.makedirs(f"{data_path}/{task_name}/{model_type}/enrichment_analysis", exist_ok=True)

        ##############################################----------------Split to test & val (5 folds)----------------######################################################################################

        max_length = 512
        batch_size = 40

        train_loader, val_loader, test_loader = process_data(gene_loaded_data, max_length, batch_size, gene2vec_flag = gene2vec_flag, model_name = model_name, task_name = task_name, model_type=model_type)
        

        ####################################################------------Run Bert  for finetuning--------------################################################################################
        
        fold_accuracy, fold_f1, fold_precision, fold_recall, fold_roc_auc, fold_s_corr, fold_p_corr, fold_mae, fold_mse, fold_r2 = [], [], [], [], [], [], [], [], [], []
        
        for k_fold in range(5):
            tensor = None
            torch.cuda.empty_cache()
            gc.collect()
            test_gene_df = pd.read_csv(f"{data_path}/{task_name}/{model_type}/folds/test_gene_names_fold_{k_fold}.csv")
            test_genes = test_gene_df['g_name'].tolist()

            val_gene_df = pd.read_csv(f"{data_path}/{task_name}/{model_type}/folds/val_gene_names_fold_{k_fold}.csv")
            val_genes = val_gene_df['g_name'].tolist()


            epochs = 20
            lr = 1e-5
            max_length = 512
            batch_size = 20
            pool ="cls"
            drop_rate = 0.1
            gene2vec_hidden = 200
            device = "cuda"
            class_map = None
            n_labels = n_labels

                
            print(f"model :{model_type}, gene2vec: {gene2vec_flag}")
            
            train_loader, val_loader, test_loader = process_data(gene_loaded_data, max_length, batch_size,
                                                                    val_genes, test_genes,
                                                                    gene2vec_flag = gene2vec_flag,
                                                                    model_name = model_name, task_name = task_name, model_type=model_type)
            
            if task_type == "classification":
                acc_test, f1_test, prec_test, rec_test, roc_auc_test = trainer(
                                                            epochs, gene_loaded_data, train_loader, val_loader, test_loader,k_fold=k_fold,
                                                            lr=lr, pool=pool, max_length=max_length, drop_rate=drop_rate,
                                                            gene2vec_flag=gene2vec_flag, gene2vec_hidden=gene2vec_hidden,
                                                            device=device, task_type=task_type,n_labels = n_labels, model_name = model_name,
                                                            task_name=task_name, model_type=model_type
                                                            )
                fold_accuracy.append(acc_test)
                fold_f1.append(f1_test)
                fold_precision.append(prec_test)
                fold_recall.append(rec_test)
                fold_roc_auc.append(roc_auc_test)

                if k_fold == 4:
                    with open(f'{data_path}/{task_name}/{model_type}/5_fold_test_metrics_{task_name}.csv', 'a') as f:
                        f.write(f"mean_accuracy: {np.mean(fold_accuracy)}, mean_f1: {np.mean(fold_f1)}, mean_precision: {np.mean(fold_precision)}, mean_recall: {np.mean(fold_recall)}, mean_roc_auc: {np.mean(fold_roc_auc)}\n")
                        f.write(f"std_accuracy: {np.std(fold_accuracy)}, std_f1: {np.std(fold_f1)}, std_precision: {np.std(fold_precision)}, std_recall: {np.std(fold_recall)}, std_roc_auc: {np.std(fold_roc_auc)}\n")
            elif task_type == "regression":
                s_test_corr, p_test_corr, mae_test, mse_test, r2_test = trainer(
                                                            epochs, gene_loaded_data, train_loader, val_loader, test_loader,k_fold=k_fold,
                                                            lr=lr, pool=pool, max_length=max_length, drop_rate=drop_rate,
                                                            gene2vec_flag=gene2vec_flag, gene2vec_hidden=gene2vec_hidden,
                                                            device=device, task_type=task_type,n_labels = n_labels, model_name = model_name,
                                                            task_name=task_name, model_type=model_type
                                                            )
                fold_s_corr.append(s_test_corr)
                fold_p_corr.append(p_test_corr)
                fold_mae.append(mae_test)
                fold_mse.append(mse_test)
                fold_r2.append(r2_test)

                if k_fold == 4:
                    with open(f'{data_path}/{task_name}/{model_type}/5_fold_test_metrics_{task_name}.csv', 'a') as f:
                        f.write(f"mean_s_corr: {np.mean(fold_s_corr)}, mean_p_corr: {np.mean(fold_p_corr)}, mean_mae: {np.mean(fold_mae)}, mean_mse: {np.mean(fold_mse)}, mean_r2: {np.mean(fold_r2)}\n")
                        f.write(f"std_s_corr: {np.std(fold_s_corr)}, std_p_corr: {np.std(fold_p_corr)}, std_mae: {np.std(fold_mae)}, std_mse: {np.std(fold_mse)}, std_r2: {np.std(fold_r2)}\n")

