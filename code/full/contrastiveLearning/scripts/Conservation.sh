#!/bin/bash

# Example command to run the training script with specific parameters
python src/run.py \
    --epochs 3 \
    --lr 3e-05 \
    --pool "mean" \
    --max_length 512 \
    --batch_size 100 \
    --model_name "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" \
    --data_path "data/Conservation.csv" \
    --task_type "regression" \
    --start_model  "checkpoints/litgene_go_0.pth"\
    --save_model_path "checkpoints/Conservation_" \
    --test_split_size 0.15 \
    --val_split_size 0.15
    

