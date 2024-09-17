###########################################################
# Project: GeneLLM
# File: model.py
# License: MIT
# Code Authors: Jararaweh A., Macualay O.S, Arredondo D., & 
#               Virupakshappa K.
###########################################################
from wrapper import analyze
import argparse
import gc
import torch



def main(args):
    task_names = ['solubility',
    'DosageSensitivity', 
    'BivalentVsLys4',
    'BivalentVsNonMethylated',
    'Tf_range', 
    'TF_target_type',
    'subcellular_location',
    'phastcons'
                ]
    task_types = ['classification', 
        'classification', 
        'classification', 
        'classification', 
        'classification', 
        'classification', 
        'classification', 
        'regression'
                    ]
    for task_name , task_type  in zip(task_names, task_types):
        input_data_path = f'/home/tailab/GeneLLM/data/{task_name}.csv'
        data_path = f'/home/tailab/GeneLLM/data/OUTPUTS1/'
        analyze(input_data_path, data_path, task_type, task_name)
        
        tensor = None
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='visualize keywords.')
    parser.add_argument("--input-data-path",help="Supply data path to train", type=str,default='../../data/cellular_locations.csv')
    parser.add_argument("--task-type",help="Type of GeneLLM task", type=str, default='classification')
    parser.add_argument("--task-name",help="Name of GeneLLM task",type=str, default='subcellular_localization')
    args = parser.parse_args()
    main(args)
