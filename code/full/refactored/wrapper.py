###########################################################
# Project: GeneLLM
# File: model.py
# License: MIT
# Code Authors: Jararaweh A., Macualay O.S, Arredondo D., & 
#               Virupakshappa K.
###########################################################
import os
import shutil
import json
from utils import save_finetuned_embeddings
from data_processor import loading_data, process_data
from modeltrainer import trainer
#from model import Logistic_Regression



def analyze(input_data_path, data_path, task_type, task_name):
    """
    @Ala and @Macualay provide description
    """
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
    #data_path = "/home/tailab/GeneLLM/data/OUTPUTS1"

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
        continue
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
                



